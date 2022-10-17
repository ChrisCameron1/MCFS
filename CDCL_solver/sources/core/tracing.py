import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class Pool(nn.Module):
    def __init__(self, axis=0):
        super(Pool, self).__init__()
        self.axis = axis

    def forward(self, tens: Tensor) -> Tensor:
        '''
        Pool across either the rows or columns of a sparse matrix
        and map the pooled representation back to a tensor of the 
        same size as the values so that it can be concatenated with 
        the values tensor
        '''
        index = tens.coalesce().indices()  # index.long()
        output = torch.sparse.sum(tens, dim=(self.axis,)).to_dense()
        return torch.index_select(output, 0, index[1-self.axis, :].long())


class PoolReduce(nn.Module):
    def __init__(self, axis=0):
        super(PoolReduce, self).__init__()
        self.axis = axis

    def forward(self, tens: Tensor) -> Tensor:
        '''
        Pool across either the rows or columns of a sparse matrix
        and map the pooled representation back to a tensor of the 
        same size as the values so that it can be concatenated with 
        the values tensor
        '''
        return torch.sparse.sum(tens, dim=(self.axis,)).to_dense()


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
        output = F.leaky_relu(self.linear(inputs))
        return torch.sparse_coo_tensor(indices, output)


model = torch.nn.Sequential(Exchangable(2, 3),
                            Exchangable(3, 3),
                            Exchangable(3, 2),
                            PoolReduce(0))  # 0 rows; 1 columns

# Create some example data
# example matrix
M = torch.tensor([[0., 1, 2, ],
                  [1., 0, 0, ],
                  [3, 3, 0],
                  [0, 1, 0.]])
# sparse version
indices = torch.nonzero(M, as_tuple=False)
values = M[indices[:, 0], indices[:, 1]][:, None]

T = torch.sparse.FloatTensor(indices.T, values)
# model(T)
# trace model
traced = torch.jit.script(model, (T))
traced.save('model-test.zip')
print(traced(T))
