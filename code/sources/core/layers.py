import math
import logging
from numpy.lib.function_base import select

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from exchangable_tensor.sp_layers import prepare_global_index


class Pool(nn.Module):
    def __init__(self, axis=0):
        super(Pool, self).__init__()
        self.axis = axis
        self.eps = 1e-16

    def forward(self, tens: Tensor) -> Tensor:
        '''
        Pool across either the rows or columns of a sparse matrix
        and map the pooled representation back to a tensor of the 
        same size as the values so that it can be concatenated with 
        the values tensor
        '''
        index = tens.coalesce().indices()  # index.long()
        output = torch.sparse.sum(tens, dim=(self.axis,)).to_dense()
        n = torch.sparse.sum(torch.sparse.FloatTensor(index, torch.ones_like(index[self.axis, :])),
                             dim=[self.axis]).to_dense().float() + self.eps
        return torch.index_select(output / n[:, None], 0, index[1-self.axis, :].long())


class PoolReduce(nn.Module):
    def __init__(self, axis=0):
        super(PoolReduce, self).__init__()
        self.axis = axis
        self.eps = 1e-16

    def forward(self, tens: Tensor) -> Tensor:
        '''
        Pool across either the rows or columns of a sparse matrix
        and map the pooled representation back to a tensor of the 
        same size as the values so that it can be concatenated with 
        the values tensor
        '''
        index = tens.coalesce().indices()
        n = torch.sparse.sum(torch.sparse.FloatTensor(index, torch.ones_like(index[self.axis, :])),
                             dim=[self.axis]).to_dense().float() + self.eps
        return torch.sparse.sum(tens, dim=(self.axis,)).to_dense() / n[:, None]


class Exchangable(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Exchangable, self).__init__()
        self.linear = torch.nn.Linear(in_dim * 4, out_dim)
        self.row = Pool(0)
        self.col = Pool(1)

    def forward(self, tens: Tensor) -> Tensor:
        '''
        '''
        values = tens.coalesce().values().float()
        indices = tens.coalesce().indices()
        both = values.mean(dim=0)[None, :].expand_as(values)
        inputs = torch.cat(
            [values, self.row(tens), self.col(tens), both], dim=1)
        output = F.dropout(F.leaky_relu(self.linear(inputs)))
        return torch.sparse_coo_tensor(indices, output)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd=256, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class GenericPool(torch.nn.Module):
    def __init__(self, f=lambda x: x.mean(axis=0, keepdim=True).expand_as(x)):
        super(GenericPool, self).__init__()
        self.f = f

    def forward(self, input, index):
        n = input.shape[0]
        u = torch.unique(index)
        y = torch.cat([self.f(input[index == i, ...]) for i in u], axis=0)
        reverse_idx = torch.cat(
            [torch.arange(n)[index == i, ...] for i in u], axis=0)
        return y[torch.argsort(reverse_idx), ...]


class TransformerPool(torch.nn.Module):
    def __init__(self, n_embd=128, n_head=4):
        super(TransformerPool, self).__init__()
        self.sa = SelfAttention(n_embd=n_embd, n_head=n_head)
        self.pool = GenericPool(
            f=lambda x: self.sa(x[None, :]).view(-1, n_embd))

    def forward(self, input, index):
        return self.pool(input, index)


class SparseExchangeableTransformer(nn.Module):
    """
    Sparse exchangable matrix layer
    """

    def __init__(self, features, n_heads, n_axes=2, bias=True):
        super(SparseExchangeableTransformer, self).__init__()
        self.pooling = nn.ModuleList(
            [TransformerPool(features, n_heads) for i in range(n_axes)])
        self.linear = nn.Linear(in_features=features * (2 + n_axes),
                                out_features=features, bias=bias)
        self.in_features = features
        self.out_features = features

    def forward(self, input, index):
        pooled = [pool_axis(input, index=index[:, i] if index is not None else None)
                  for i, pool_axis in enumerate(self.pooling)]
        pooled += [torch.mean(input, dim=0).expand_as(input)]
        stacked = torch.cat([input] + pooled, dim=1)
        activation = self.linear(stacked)
        return activation
