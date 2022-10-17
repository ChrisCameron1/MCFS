# standard lib imports
import glob
import importlib
from datetime import date
from typing_extensions import clear_overloads
from mysql.connector.connection import MySQLProtocol
import numpy as np
import argparse
import yaml
import os
import pathlib
from shutil import copyfile
import pandas as pd
from typing import Optional
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


# torch imports
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

# pytorch lightning imports
import pytorch_lightning as pl
import torchmetrics as tm
from pytorch_lightning.callbacks import ModelCheckpoint
# layers
from model_evaluation import cdcl_evaluation
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO

import mysql.connector
import tempfile

from scripts.analysis_utils import runtime_scatter_plot, torch_to_cnf

import time


import base64

# class CustomCheckpointIO(CheckpointIO):
#     def save_checkpoint(self, checkpoint, path, storage_options=None):
        

#     def load_checkpoint(self, path, storage_options=None):
#         ...

#     def remove_checkpoint(self, path):
#         ...

def run_kcnfs(instance):
    instance = instance.replace('\n','')
    KCNFS_CALLSTRING = '/global/scratch/wildebeest/kcnfs-2006/kcnfs-2006 -nop'
    temp = tempfile.NamedTemporaryFile(delete=False)
    #print(f'{KCNFS_CALLSTRING} {instance}')
    filename = f'/tmp/{os.path.basename(instance)}'
    # Add random seed to the filename
    filename = filename + '_' + str(random.randint(0, 1000000))
    start = time.time()
    os.system(f'{KCNFS_CALLSTRING} {instance} > {filename}')
    end = time.time()
    elapsed_time = end - start
    decisions = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            decision_string = "Size of search tree  :"
            if decision_string in line:
                decisions = int(line.split(decision_string)[1].split(' ')[1])
                break
    return decisions, elapsed_time

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
        values = tens.coalesce().values()
        output = torch.sparse.sum(tens, dim=(self.axis,)).to_dense()
        n = torch.sparse.sum(torch.sparse_coo_tensor(index, torch.ones_like(values[:, 0:1])),
                             dim=(self.axis,)).to_dense()
        return torch.index_select(output / n, 0, index[1-self.axis, :].long())


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
        self.linear = torch.nn.Linear(in_dim * 4, out_dim)  # todo fix 'both'
        self.row = Pool(0)
        self.col = Pool(1)
        self.linear

    def forward(self, tens: Tensor) -> Tensor:
        '''
        '''
        values = tens.coalesce().values().float()
        indices = tens.coalesce().indices()
        # both = values.mean(dim=0)[None, :].expand_as(values) #todo fix 'both'
        inputs = torch.cat(
            [values, self.row(tens), self.col(tens), values.mean(dim=0)[None, :].expand_as(values)], dim=1)
        output = F.leaky_relu(self.linear(inputs))
        return torch.sparse_coo_tensor(indices, output)


class MCTSDataset(Dataset):
    """Monte Carlo Tree Search dataset."""

    def __init__(self, 
                paths, 
                best_action_loss=False, 
                value_dict=None, 
                softmax_counts=False, 
                q_value_head=False,
                weight_counts_by_q=False,
                temperature=1.0
                ):
        """
        Args:
            root_dir (string): Directory with all the problems.
            indices (list): List of problems to train on.
        """
        self.files = paths
        self.is_df = isinstance(paths, pd.DataFrame)
        self.best_action_loss = best_action_loss
        self.value_dict = value_dict
        self.softmax_counts = softmax_counts
        self.q_value_head = q_value_head
        self.weight_counts_by_q = weight_counts_by_q
        self.temperature = temperature

    def __len__(self):
        if self.is_df:
            return self.files.shape[0]
        else:
            return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            if self.is_df:
                f = self.files.iloc[idx,0]
            else:
                f = self.files[idx]

            # pickle_inputs.push_back(values);
            # pickle_inputs.push_back(index);
            # pickle_inputs.push_back(mcts_lit_probabilities);
            # pickle_inputs.push_back(unnormalized_reward);
            # pickle_inputs.push_back(mcts_lit_counts);

            data_point = torch.load(f)
            values = data_point[0]
            indices = data_point[1]
            literal_probs = data_point[2]
            q_vals = data_point[3]
            q_vals /= q_vals.sum() # normalize
            counts = data_point[4]
            counts /= counts.sum() # normalize

            if self.weight_counts_by_q:
                counts = counts * q_vals
                counts /= counts.sum()

            if self.softmax_counts:
                counts = torch.softmax(self.temperature * counts, dim=0)

            if self.q_value_head:
                counts = torch.cat([counts, q_vals], dim=0)

            if len(data_point) == 5:
                value = torch.tensor([1.0]).float()
            else:
                value = data_point[5]
                if value is None:
                    value = torch.tensor([1.0]).float()
                else:
                    value = torch.tensor(np.log2([float(value)])).float()

        except Exception as e:
            print(f'Skipping {self.files[idx]}')
            print(e)
            raise e
            #return None
        output = indices.T, values, counts.view((-1,2)), value
        #print(output)
        #exit()
        return output


def counts_loss_fn(pred, target, truncation_index=None):
    if not truncation_index:
        t = target.flatten()
        p = pred.flatten()
    else:
        # Get k biggest values from target
        t = target.flatten()
        top_indices = torch.topk(t, truncation_index, largest=True).indices
        t = t[top_indices]
        p = pred.flatten()[top_indices]
    return -(t * (p - torch.logsumexp(p, 0))).mean()

def value_loss_fn(pred,target):
    t = target.flatten()
    p = pred.flatten()
    return nn.MSELoss()(p, t)


class Resnet(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=False):
        super(Resnet, self).__init__()
        self.ex1 = Exchangable(in_dim, out_dim)
        self.ex2 = Exchangable(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d() if batch_norm else None

    def forward(self, tens: Tensor) -> Tensor:
        '''
        '''
        out = self.ex2(self.ex1(tens))
        indices = tens.coalesce().indices()
        inp_values = tens.coalesce().values().float()
        out_values = out.coalesce().values().float()
        final_values = F.leaky_relu(inp_values + out_values)
        if self.batch_norm is not None:
            final_values = self.batch_norm(final_values)
        return torch.sparse_coo_tensor(indices, final_values)

class Policy(nn.Module):
    def __init__(self, 
                units=128, 
                n_layers=8, 
                in_width=2, 
                resnet=False, 
                dropout=0.0, 
                use_value_net=True, 
                num_size_features=3,
                pool_before_mlp=True,
                num_heads=1,
                attention_pooling=False,
                use_value_head=False):
        super().__init__()
        self.input_width = 128+1
        inputs = [Exchangable(self.input_width, units)]
        self.num_size_features = num_size_features
        self.pool_before_mlp = pool_before_mlp
        self.attention_pooling = attention_pooling
        self.use_value_head = use_value_head

        self.embed =torch.nn.Embedding(2, embedding_dim = 128) 

        if resnet:
            print("Using RESNET")
            for _ in range(n_layers // 2):
                inputs.append(Resnet(units, units))
        else:
            for _ in range(n_layers):
                inputs.append(Exchangable(units, units))
        inputs.append(PoolReduce(0))

        self.exchangeable_model = torch.nn.Sequential(*inputs)
        self.counts_linear = torch.nn.Sequential(  # torch.nn.LayerNorm([units]),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, 2))

        self.q_value_head = torch.nn.Sequential(  # torch.nn.LayerNorm([units]),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, 2))

        # if attention_pooling:
        self.multihead_attn = nn.MultiheadAttention(int((units + num_size_features)), num_heads)
        self.value_linear = torch.nn.Sequential(  # torch.nn.LayerNorm([units]),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(units+num_size_features, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, 1))

        self.subsolver = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(units+num_size_features, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, 1))
        self._init()
        #self.use_value_net = use_value_net

    def _init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                torch.nn.init.kaiming_uniform_(
                    p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            else:
                torch.nn.init.zeros_(p)

    def shared_forward(self, X: Tensor) -> Tensor:
        values = X.coalesce().values().float()
        indices = X.coalesce().indices()
        new_values = torch.zeros((values.shape[0], 2), device=values.device)
        one = torch.tensor(1.,device=values.device).float()
        tmp = torch.minimum(values.flatten(), one)
        new_values[torch.arange(values.shape[0]), tmp.long()] = one
        X = torch.sparse_coo_tensor(indices, new_values)
        em = self.exchangeable_model(X)
        if self.num_size_features > 0:
            size_features = torch.tensor([float(values.shape[0]), float(indices[0].max().float()), float(indices[1].max().float())]).to(em.device)
            size_features /= 100 # shrink so on similar scale as other features
            # Copy size features column for num_vars rows
            size_features = torch.stack([size_features] * em.shape[0])
            # Append size features to every column of em
            em = torch.cat((em, size_features), -1)     
        return em

    def forward(self, X: Tensor) -> Tensor:
        values = X.coalesce().values().float()
        indices = X.coalesce().indices()
        # Replace self.embed with values
        new_values = self.embed(values.long()).squeeze(1)
        # append degree feature
        values_neg = values.clone()
        values_neg[values_neg == 0] = -1
        
        unique, counts = torch.unique(values_neg.flatten() * indices[1,:], return_counts=True)
        degree = counts[indices[1,:]]
        new_values = torch.cat((new_values, degree.unsqueeze(1)), dim=1)
        # new_values = torch.zeros((values.shape[0], 2), device=values.device)
        # one = torch.tensor(1.,device=values.device).float()
        # tmp = torch.minimum(values.flatten(), one)
        # new_values[torch.arange(values.shape[0]), tmp.long()] = one
        X = torch.sparse_coo_tensor(indices, new_values)
        em = self.exchangeable_model(X)
        counts = self.counts_linear(em)
        
        # TODO: This may cause tracing issues. Not sure if conditionals allowed. 
        # Might be okay because not conditional on input
        if self.num_size_features > 0:
            size_features = torch.tensor([float(values.shape[0]), float(indices[0].max().float()), float(indices[1].max().float())]).to(em.device)
            size_features /= 100 # shrink so on similar scale as other features
            # Copy size features column for num_vars rows
            size_features = torch.stack([size_features] * em.shape[0])
            # Append size features to every column of em
            em = torch.cat((em, size_features), -1)     
        if self.pool_before_mlp:
            if self.attention_pooling:
                # Add second dimension to em. This be broken for batching
                em = torch.unsqueeze(em, 1)#.to(em.device)
                # Split em in into key, value, query
                key, tail = torch.split(em, [1,em.shape[0]-1], dim=0)
                value = em
                query = em
                em, attn_output_weights = self.multihead_attn(key, value, query) # value, query must be same size. key has same embedding size and same as output size
            else:
                em = em.mean(0)

        value = torch.mean(torch.squeeze(self.value_linear(em)))
        if self.use_value_head:
            value_head = self.q_value_head(em)
            output = torch.cat([counts.flatten(), value_head.flatten(), value.unsqueeze(0)])#, dim=-1)
        else:
            output = torch.cat([counts.flatten(), value.unsqueeze(0)])#, dim=-1)
        return output


class CurriculumWeighter(object):
    def __init__(self, df, config):
        self.df = df
        self.reweight_freq = config['optimization'].get("reweight_freq", 100)
        self.costs = -np.sort(df.n_vars.unique())
        self.old_losses = np.zeros_like(self.costs)
        self.losses = np.zeros_like(self.costs)
        self.lookup = {k: i for i, k in enumerate(df.n_vars.unique())}
        self.weights = costs
        self.counts = np.zeros_like(self.costs)

    def update_weights(self, index, losses):
        n_vars = index.max(axis=0)[0][-1] + 1
        self.losses[self.lookup[n_vars]] += losses
    
    def reweight(self):
        pass

class FixedWeighted(object):
    def __init__(self, df, config, scale=1000, debug=False, sigmoid=True):
        self.use_sigmoid = sigmoid
        self.df = df
        _, value_counts = np.unique(df.n_vars, return_counts=True)
        self.n = value_counts.shape[0]
        self.base_weights = value_counts / value_counts.sum()
        self.count = 0
        self.weights = self._f(0)
        ws = config['optimization'].get("weight_scale", None) 
        scale = ws if ws is not None else scale
        self.scale = scale
        if debug:
            self.observed_indices = []
        self.debug = debug

    def _f(self, lam):
        x = np.arange(self.n)
        if self.use_sigmoid:
            sigmoid = lambda x: 1./(1. + np.exp(10 * (x - lam)))
            return sigmoid(x)
        else:
            mu = np.minimum(lam, self.n)
            sig = np.maximum(lam - self.n, 1)
            gauss = lambda x: np.exp(-(x - mu)**2 / (sig**2))
            return gauss(x)
    
    def update_weights(self, index, losses):
        if self.debug:
            self.observed_indices.append(float(index.max(axis=0)[0][-1]))
        self.count += 1
    
    def reweight(self):
        self.weights = self._f(self.count / self.scale)
        self.weights *= self.base_weights
        self.weights /= self.weights.sum()
        if self.debug:
            print(f"Average # of variables: {np.mean(self.observed_indices)}, count: {len(self.observed_indices)}")
            print("Weights:", np.round(self.weights, 2))
            self.observed_indices = []
        

class PolicyModel(pl.LightningModule):
    def __init__(self,
                module, 
                lr=1e-3, 
                sampler=None, 
                comet_exp=None,
                validation_cnfs=[],
                training_cnfs=[],
                validation_baseline=None, 
                training_baseline=None,
                traced_model_path=None, 
                freq=1,
                tracing_data=None,
                validation_cutoff=10,
                num_variables_for_rollout=None,
                value_loss=True,
                counts_loss=True,
                q_value_loss=False,
                subsolver_loss=False,
                value_loss_weight=0.0001,
                truncate_loss = None):

        super().__init__()
        self.module = module
        self.lr = lr
        # self.train_acc = tm.metrics.Accuracy()
        # self.valid_acc = tm.metrics.Accuracy()
        self.running_acc = 0.
        self.n = 0.
        self.curriculum_weights = sampler
        self.comet_exp = comet_exp
        self.value_loss_weight = value_loss_weight

        self.batch_n = 0.
        self.batch_acc = 0.
        self.batch_acc_topk =0.
        self.batch_loss = 0.
        self.batch_value_loss = 0.
        self.baseline_value_loss = 0.
        self.mean_baseline_loss = 0.
        self.mean_value = 0.
        self.batch_subsolver_loss = 0.

        self.valid_running_acc = 0.
        self.running_loss = 0.
        self.checkpoint_validation_loss = 0.
        self.last_iter_valid_loss = 0.
        self.valid_n = 0.
        self.running_value_loss = 0.
        self.running_counts_loss = 0.

        self.size_mean_baseline_loss = 0.
        self.size_n = {}
        self.size_mean = {}

        self.val_decisions = 0.

        self.topk = 10

        self.traced_model_path = traced_model_path
        self.validation_cnfs = validation_cnfs
        self.training_cnfs = training_cnfs
        self.validation_baseline = validation_baseline
        self.training_baseline = training_baseline
        self.validation_runtime = None
        self.comet_exp = comet_exp
        self.validation_counter = 0
        self.freq=freq

        self.tracing_data = tracing_data
        self.validation_cutoff = validation_cutoff
        self.num_variables_for_rollout = num_variables_for_rollout

        self.value_predictions = []
        self.value_targets = []

        self.value_loss = value_loss
        self.counts_loss = counts_loss
        self.q_value_loss = q_value_loss
        self.subsolver_loss = subsolver_loss

        self.truncate_loss = truncate_loss

    def forward(self, X: Tensor) -> Tensor:
        return self.module(X)

    def policy(self, X: Tensor) -> Tensor:
        return F.softmax(self(X).flatten()).view_as(X)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.lr)
        print(f"Actual LR: {self.lr}")
        #lr_lambda = lambda epoch: 1e-9 if epoch < 1 else self.lr
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=0, verbose=False)
        return optimizer#[optimizer], [scheduler]

    def _make_idx(self, idx):
        return torch.cat([torch.repeat_interleave(idx, 2)[:, None],
                          torch.repeat_interleave(torch.tensor([0, 1])[None, :],
                                                  idx.shape[0], 0).flatten()[:, None]],
                         axis=1).long()

    def _remap(self, index, shape_lookup):
        shape_lookup = shape_lookup.view((-1, 4))
        fn_indices = []
        start = 0
        for i in range(shape_lookup.shape[0]):
            idx = index[start:start + int(shape_lookup[i,0]),:]
            fn_indices.append(idx)
            start += int(shape_lookup[i,0])
        return fn_indices

    
    def training_step(self, batch, batch_idx):
        index, values, counts_target, value_target, shapes = batch
        fn_indices = self._remap(index, shapes)

        T = torch.sparse_coo_tensor(index.T, values)
        #print(index)
        output = self(T)
        counts_hat = output[:-1]
        counts_target = counts_target.flatten()
        # get size of counts_hat
        if self.q_value_loss:
            counts_hat_middle = int(counts_hat.shape[0]/2)
            counts_ce_hat = counts_hat[:counts_hat_middle]
            count_accuracy_hat = counts_hat[counts_hat_middle:]
            counts_ce_target = counts_target[:counts_hat_middle]
            count_accuracy_target = counts_target[counts_hat_middle:]
        else:
            counts_ce_hat = counts_hat
            counts_ce_target = counts_target

        value_hat = output[-1]
        value_loss = 0.
        if self.counts_loss:
            counts_loss = counts_loss_fn(counts_ce_hat, counts_ce_target, self.truncate_loss)
        else:
            counts_loss = 0.
        if self.q_value_loss:
            counts_accuracy_loss = counts_loss_fn(count_accuracy_hat, count_accuracy_target, self.truncate_loss)
        else:
            counts_accuracy_loss = 0.
        if self.value_loss:
            value_loss = value_loss_fn(value_hat, value_target)
        else:
            value_loss = 0.
        if self.subsolver_loss:
            shared_output = self.module.shared_forward(T)
            # Remove gradients from shared layers
            shared_output = shared_output.detach().requires_grad_()
            subsolver_hat = torch.mean(torch.squeeze(self.module.subsolver(shared_output))) # Bug here with matrix sizes
            subsolver_loss = value_loss_fn(subsolver_hat, value_target) # Test if normal value first
        else:
            subsolver_loss = 0.
        
        loss = counts_loss + (value_loss * self.value_loss_weight) + subsolver_loss + counts_accuracy_loss
        y_hat_pred = torch.tensor(torch.argmax(counts_hat))#.clone().detach()
        counts_target_pred = torch.tensor(torch.argmax(counts_target))#.clone().detach()
        if self.curriculum_weights is not None:
            self.curriculum_weights.weighter.update_weights(index, loss)
            if batch_idx % self.curriculum_weights.reweight_freq == 0:
                self.curriculum_weights.resample()

        flatten_target = counts_target.flatten()
        target_pred_topk = torch.topk(flatten_target,min(self.topk,flatten_target.shape[0])).indices

        acc = (y_hat_pred == counts_target_pred).float().mean()
        topk_acc = (y_hat_pred == target_pred_topk).float().sum() # TODO: Broken. Thing need to take sum and then mean across samples in batch.
        
        self.running_acc = self.running_acc * \
            self.n / (self.n+1) + acc / (self.n + 1)
        self.n += 1

        self.batch_acc = self.batch_acc * \
            self.batch_n / (self.batch_n+1) + acc / (self.batch_n + 1)
        self.batch_acc_topk = self.batch_acc_topk * \
            self.batch_n / (self.batch_n+1) + topk_acc / (self.batch_n + 1)
        self.batch_loss = self.batch_loss * \
            self.batch_n / (self.batch_n+1) + loss / (self.batch_n + 1)
        self.batch_value_loss = self.batch_value_loss * \
            self.batch_n / (self.batch_n+1) + value_loss / (self.batch_n + 1)
        self.batch_subsolver_loss = self.batch_subsolver_loss * \
            self.batch_n / (self.batch_n+1) + subsolver_loss / (self.batch_n + 1)
        self.mean_value = self.mean_value * \
            (self.n-1) / (self.n) + value_target.mean() / (self.n)
        self.baseline_value_loss = (value_target.mean() - self.mean_value)**2
        self.mean_baseline_loss = self.mean_baseline_loss * \
            (self.n-1) / (self.n) + self.baseline_value_loss / (self.n)


        # Check baseline that conditions on number of variables
        num_variables = counts_hat.shape[0]
        if num_variables not in self.size_n:
            self.size_n[num_variables] = 0
        if num_variables not in self.size_mean:
            self.size_mean[num_variables] = 0
        self.size_n[num_variables] += 1
        self.size_mean[num_variables] = self.size_mean[num_variables] * \
            self.size_n[num_variables] / (self.size_n[num_variables]+1) + value_target.mean() / (self.size_n[num_variables]+1)
        size_baseline_value_loss = (value_target.mean() - self.size_mean[num_variables])**2
        self.size_mean_baseline_loss = self.size_mean_baseline_loss * \
            (self.n-1) / (self.n) + size_baseline_value_loss / (self.n)
        self.batch_n += 1


        if self.n % 1000 == 0:
            self.comet_exp.log_metric('train_running_acc', self.running_acc)
            self.comet_exp.log_metric('train_acc', self.batch_acc)
            self.comet_exp.log_metric(f"train_acc_top{self.topk}", self.batch_acc_topk)
            self.comet_exp.log_metric('train_loss', self.batch_loss)
            self.comet_exp.log_metric('train_value_loss', min(2.0,self.batch_value_loss))
            self.comet_exp.log_metric('baseline_value_loss', self.mean_baseline_loss)
            self.comet_exp.log_metric('size_baseline_value_loss', self.size_mean_baseline_loss)
            self.comet_exp.log_metric('subsolver_value_loss', min(2.0,self.batch_subsolver_loss))

            self.log('running_value_loss', min(2.0, self.batch_value_loss), prog_bar=True)
            self.log('running_acc', self.batch_acc, prog_bar=True)
            self.log('baseline_value_loss', self.mean_baseline_loss, prog_bar=True)


            self.batch_n = 0.
            self.batch_acc = 0.
            self.batch_acc_topk = 0.
            self.batch_loss = 0.
            self.batch_value_loss = 0.
            self.batch_subsolver_loss = 0.

            

        # self.log('train_loss', loss, prog_bar=True)
        # self.log('train_acc', self.running_acc, prog_bar=True)
        return {'loss' : loss}

    def validation_step(self, batch, batch_idx):
        index, values, counts_target, value_target, shapes = batch
        
        fn_indices = self._remap(index, shapes)

        T = torch.sparse_coo_tensor(index.T, values)
        output = self(T)
        counts_hat = output[:-1]
        counts_target = counts_target.flatten()
        if self.q_value_loss:
            counts_hat_middle = int(counts_hat.shape[0]/2)
            counts_ce_hat = counts_hat[:counts_hat_middle]
            count_accuracy_hat = counts_hat[counts_hat_middle:]
            counts_ce_target = counts_target[:counts_hat_middle]
            count_accuracy_target = counts_target[counts_hat_middle:]
        else:
            counts_ce_hat = counts_hat
            counts_ce_target = counts_target
        
        value_hat = output[-1]
        try:
            value_loss = 0.
            if self.counts_loss:
                counts_loss = counts_loss_fn(counts_ce_hat, counts_ce_target, self.truncate_loss)
            else:
                counts_loss = 0.
            if self.q_value_loss:
                counts_accuracy_loss = counts_loss_fn(count_accuracy_hat, counts_ce_hat, self.truncate_loss)
            else:
                counts_accuracy_loss = 0.
            if self.value_loss:
                value_loss = value_loss_fn(value_hat, value_target)
            else:
                value_loss = 0.
            loss = counts_loss + (value_loss * self.value_loss_weight) + counts_accuracy_loss
        except:
            print("Getting loss failed")
            import pdb; pdb.set_trace()
        y_hat_pred = torch.argmax(counts_hat).clone().detach()
        target_pred = torch.argmax(counts_target).clone().detach()

        acc = (y_hat_pred == target_pred).float().mean()

        self.valid_running_acc = self.valid_running_acc * \
            self.valid_n / (self.valid_n+1) + acc / (self.valid_n + 1)
        self.running_loss = self.running_loss * \
            self.valid_n / (self.valid_n+1) + loss / (self.valid_n + 1)
        self.running_value_loss = self.running_value_loss * \
            self.valid_n / (self.valid_n+1) + value_loss / (self.valid_n + 1)
        self.running_counts_loss = self.running_counts_loss * \
            self.valid_n / (self.valid_n+1) + counts_loss / (self.valid_n + 1)

        self.value_predictions.append(value_hat.cpu().numpy())
        self.value_targets.append(value_target.cpu().numpy())

        self.valid_n += 1

        return {'val_loss' : loss}

    def validation_epoch_end(self, validation_step_outputs):
        # for pred in validation_step_outputs:

        self.log('val_loss', self.running_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.valid_running_acc, prog_bar=True, on_epoch=True)
        self.log('val_value', self.running_value_loss, prog_bar=True, on_epoch=True)

        self.comet_exp.log_metric('val_acc', self.valid_running_acc)
        self.checkpoint_validation_loss = self.running_loss
        self.comet_exp.log_metric('val_loss', self.running_loss)
        self.comet_exp.log_metric('val_value', min(2.0, self.running_value_loss))
        self.comet_exp.log_metric('val_counts', self.running_counts_loss)

        if len(self.value_predictions) == len(self.value_targets):
            print("Warning: number of predictions and targets do not match")

        #  Save scatter plot to comet of validation predictions and targets.
        if len(self.value_predictions) == len(self.value_targets) and self.value_loss:
            plt.scatter(self.value_predictions, self.value_targets, alpha=0.1)
            # Plot line y = x over range of value_targets
            x = [min(self.value_targets), max(self.value_targets)] 
            plt.plot(x,x)
            plt.xlim = (min(self.value_targets), max(self.value_targets))
            plt.xlabel('Predictions')
            plt.ylabel('Targets') 
            # Save figure to file
            filename = f'./figures/val_predictions_targets.png'
            plt.savefig(filename)
            self.comet_exp.log_image(filename, name='value_predict_vs_target', step=self.validation_counter) 
            plt.clf()    
            

        # Log weights of model to comet
        weights = []
        for name in self.module.named_parameters():
                if 'weight' in name[0]:
                    weights.extend(name[1].detach().cpu().numpy().tolist())
        self.comet_exp.log_histogram_3d(weights, step=self.validation_counter)

        self.value_predictions = []
        self.value_targets = []
        self.valid_n = 0
        self.valid_running_acc = 0.
        self.last_iter_valid_loss = self.running_loss
        self.running_loss = 0.
        self.running_value_loss = 0.

        # Save torchscipt model for validation
        self.module.eval()
        self.module.cpu()
        traced = torch.jit.script(self.module, (self.tracing_data,))#torch.jit.script
        traced.save(self.traced_model_path)
        self.module.cuda()
     
        if self.validation_counter % self.freq == 0 and len(self.validation_cnfs) > 0:

            if self.validation_baseline is None:
                print("Getting kcnfs validation baseline..")
                self.validation_baseline = []
                self.validation_runtime = []
                for cnf in tqdm(self.validation_cnfs):
                    kcnfs_decisions, kcnfs_runtime = run_kcnfs(cnf)
                    self.validation_baseline.append(kcnfs_decisions)
                    self.validation_runtime.append(kcnfs_runtime)
            if self.training_baseline is None:
                print("Getting kcnfs training baseline..")
                self.training_baseline = []
                self.training_runtime = []
                for cnf in tqdm(self.training_cnfs):
                    kcnfs_decisions, kcnfs_runtime = run_kcnfs(cnf)
                    self.training_baseline.append(kcnfs_decisions)
                    self.training_runtime.append(kcnfs_runtime)

            unsat_decisions = branching_evaluation(self.validation_cnfs,
                                                    self.comet_exp,
                                                     self.validation_baseline, 
                                                     self.validation_runtime, 
                                                     self.traced_model_path, 
                                                     self.validation_counter,
                                                     setting='val')
            self.log('val_decisions', np.mean(unsat_decisions), prog_bar=True, on_epoch=True)
            self.val_decisions = np.mean(unsat_decisions)
            unsat_decisions = branching_evaluation(self.training_cnfs, 
                                                    self.comet_exp, 
                                                    self.training_baseline, 
                                                    self.training_runtime , 
                                                    self.traced_model_path, 
                                                    self.validation_counter, 
                                                    setting='train')
            self.log('train_decisions', np.mean(unsat_decisions), prog_bar=True, on_epoch=True)

            self.comet_exp.log_metric('kcnfs_runtime', np.mean(self.validation_runtime))
            self.comet_exp.log_metric('kcnfs_decisions', np.mean(self.validation_baseline))

        self.validation_counter+=1

    def on_save_checkpoint(self, checkpoint):
        print("Tracing saved checkpoint and pushing to database")
        # Save torchscript model for validation
        self.module.eval()
        self.module.cpu()
        traced = torch.jit.script(self.module, (self.tracing_data,))
        # Get directory of self.traced_model_path
        traced_directory  = os.path.dirname(self.traced_model_path)

        traced_file = os.path.join(traced_directory, f'epoch_{self.validation_counter}_valid_loss_{self.checkpoint_validation_loss:.7f}decisions_{self.val_decisions}.zip')
        traced.save(traced_file)
        self.module.cuda()


def branching_evaluation(cnfs, comet_exp, baseline, runtime, model_path, validation_counter, setting='validation'):

    decisions, runtimes, statuses = cdcl_evaluation(cnfs,
                model=model_path,
                data_dir='./',
                exec_dir='./',
                read_logs_only=False,
                rerun=True,
                depth_before_rollout=5,
                mcts=False
                )
    # Check if decisions same length as baseline
    assert len(decisions) == len(baseline)
    
    # Save decision comparison plot
    decisions_ratio = np.mean(np.array(baseline) / np.array(decisions))
    runtime_scatter_plot(nn=decisions, baseline=baseline, statuses=statuses,xlabel='KCNFs decisions', ylabel='NN+KCNFs decisions', title=f'kcnfs {decisions_ratio} times faster', unsat_only=True)
    temp_name = next(tempfile._get_candidate_names())
    figure_filename = os.path.join('/tmp/', temp_name) + '.png'
    plt.savefig(figure_filename)
    comet_exp.log_image(figure_filename, name=f'decisions_comparison_{setting}', step=validation_counter) 
    plt.clf()
    
    # Save runtime comparison plot
    runtime_ratio = np.mean(np.array(runtime) / np.array(runtimes))
    runtime_scatter_plot(nn=runtimes, baseline=runtime, statuses=statuses, xlabel='KCNFs (s)', ylabel='NN+KCNFs (s)', title=f'kcnfs {runtime_ratio} times faster', unsat_only=True)
    temp_name = next(tempfile._get_candidate_names())
    figure_filename = os.path.join('/tmp/', temp_name) + '.png'
    plt.savefig(figure_filename)
    comet_exp.log_image(figure_filename, name=f'runtime_comparison_{setting}', step=validation_counter) 
    plt.clf()

    sat_decisions = [x for i, x in enumerate(decisions) if statuses[i] == 'SAT']
    unsat_decisions = [x for i, x in enumerate(decisions) if statuses[i] == 'UNSAT']
    print(f'{setting} # branches: {np.mean(sat_decisions)} SAT, {np.mean(unsat_decisions)} UNSAT')
    comet_exp.log_metric(f'{setting}_sat_decisions', np.mean(sat_decisions))
    comet_exp.log_metric(f'{setting}_unsat_decisions', np.mean(unsat_decisions))
    comet_exp.log_metric(f'{setting}_decisions_ratio', decisions_ratio)
    comet_exp.log_metric(f'{setting}_runtime_ratio', runtime_ratio)
    return unsat_decisions

class ResetAccCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module.accuracy = 0.
        pl_module.n = 0

    
class ResetAccCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module.accuracy = 0.
        pl_module.n = 0


def to_valid_index(index):
    _, valid_index = torch.unique(index, return_inverse=True)
    return valid_index

class CurriculumSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    df: pd.DataFrame
    num_samples: int

    def __init__(self, df, weighter, config, num_samples: Optional[int]=None, generator=None) -> None:
        self.files = {s: df.loc[df.n_vars == s,:].index for s in np.sort(df.n_vars.unique())}
        self.subsampled_files = None
        self.weighter = weighter
        self.resample()
        self.generator = generator
        self._num_samples = num_samples
        self.reweight_freq = config['optimization'].get("reweight_freq", 1000)

    def resample(self):
        self.weighter.reweight()
        m = np.max([v.shape[0] for v in self.files.values()]) # get longest file list
        resampled_files = {s: np.random.choice(v, m, replace=True) for s, v in self.files.items()}
        selected = np.random.multinomial(m * len(self.files.keys()), self.weighter.weights / self.weighter.weights.sum())
        self.subsampled_files = []
        for k, n in zip(resampled_files.keys(), selected):
            self.subsampled_files += list(resampled_files[k][:n])
        self.subsampled_files = np.random.permutation(self.subsampled_files)
    
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.subsampled_files)
        return self._num_samples
    
    def __iter__(self):
        return iter(self.subsampled_files[:self.num_samples])

    def __len__(self):
        if self._num_samples is None:
            return len(self.subsampled_files)
        return self._num_samples

def reindex(batch):
    '''
    Custom collate function which builds a block sparse matrix from
    a set of sparse matrices. Given a list of sparse matrices:
    [A1, A2, ..., An]
    this function returns the indices and values of a matrix of the form:
    [A1          ]
    [   A2       ]
    [      ...   ]
    [          An]

    '''
    indices = []
    values = []
    counts_targets = []
    value_targets = []
    #masks = []
    shapes = []
    max_indices = torch.zeros((2)).long()
    for b in batch:
        if b is None:
            continue
        else:
            (i, v, t1, t2) = b
            # print(b)
            # exit()
        indices.append(i + max_indices[None, :])
        values.append(v)
        counts_targets.append(t1)
        value_targets.append(t2)
        #masks.append(m)
        # return offsets and shapes so we can reconstruct the original 
        # tensors.
        shapes.append(torch.cat([torch.tensor(indices[-1].shape), max_indices.clone()], dim=0))
        # update the offset tensor
        max_indices += torch.max(i, dim=0).values + 1
    return tuple([torch.cat(i, dim=0) for i in [indices, values, counts_targets, value_targets, shapes]])


def main(args):
    config = importlib.import_module(f"config.{args.config}").configuration
    COMET = importlib.import_module(f"config.{args.config}").COMET

    value_dict = None

    # Override any of the architecture parameters from the command line
    if args.lr is not None:
        config['optimization']['lr'] = args.lr
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.grad_batches is not None:
        config['optimization']['accumulate_grad_batches'] = int(args.grad_batches)
    if args.units is not None:
        config['model']['units'] = int(args.units)
    if args.layers is not None:
        config['model']['layers'] = int(args.layers)

    if args.use_mcts_db:
        mctsdb = mysql.connector.connect(
            user="username",
            password="password",
            host="host",
            port="4545"
        )
        #TODO: Add richer options for querying
        command = f"""SELECT D.bytes, D.value FROM DPLLSAT.Data D, DPLLSAT.Experiments E
                    WHERE D.experiment_id = E.experiment_id AND E.name = '{args.experiment_name}'"""
        cursor = mctsdb.cursor()
        cursor.execute(command)
        dataset = cursor.fetchall()
        
        #write to a tempfolder
        tmpdir = tempfile.mkdtemp()
        for data_point in dataset:
            point = data_point[0]
            value = data_point[1]
            if not value:
                print("Data point does not have value. Continuing...")
                continue
            temp_name = next(tempfile._get_candidate_names())
            pickle_data_bytes = base64.standard_b64decode(point)
            filename = f"{tmpdir}/{temp_name}.zip"
            with open(filename, 'wb') as f:
                f.write(pickle_data_bytes)
            value_dict[filename] = value

        # TODO: need to add value to the dataset

        train_path = tempdir
        valid_path = tempdir

    else:
        train_path = os.path.join(args.datapath,config["path_train"])
        valid_path = os.path.join(args.datapath,config["path_valid"])

    train_instances_file = None
    valid_instances_file = None
    solver_depths = {}
    if "train_instances" in config:
        train_instances_file = os.path.join(args.datapath,config["train_instances"])
    if "valid_instances" in config:
        valid_instances_file = os.path.join(args.datapath,config["valid_instances"])
    if 'subsolver_depths' in config:
        solver_depths = json.load(open(os.path.join(args.datapath,config['subsolver_depths'])))

    cp = args.cp

    print(f"Loading config: {args.config}. Training with the following configuration:")
    print(yaml.dump(config, sort_keys=True))
    checkpoint_path = f'checkpoints/{args.config}-{args.name}-{args.seed}/'
    tracing_path = f'traced/{args.config}-{args.name}-{args.seed}/'
    os.makedirs(tracing_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=10,
        save_last=True,
        verbose=True,
        monitor=args.valid_metric,
        mode='min',
        #period=1,
        #every_n_train_steps=100,
        #every_n_val_epochs=1,
        filename='{epoch}-{val_loss:.7f}-{val_decisions:.1f}'
    )

    # COMET setup
    experiment = Experiment(api_key=COMET['key'],
                project_name="BranchPrediction", workspace=COMET['un'],
                auto_output_logging=None)
    experiment.log_parameters(vars(args))
    experiment.set_name(args.name)
    # Log config to comet
    experiment.log_parameters(config)

    if train_instances_file is not None and valid_instances_file is not None:
        print(f'Reading instance from {train_instances_file} and {valid_instances_file}')
        train_files = open(train_instances_file,'r').read().splitlines()
        valid_files = open(valid_instances_file, 'r').read().splitlines()
        # If train files or valid files are empty, exit
        if len(train_files) == 0 or len(valid_files) == 0:
            print(f'No instances found in {train_path} or {valid_path}')
            exit()


        #print(train_files[0])
    elif train_path == valid_path:
        print(f'Reading instances from {train_path}')
        files = glob.glob(train_path + '/*.zip')
        random.shuffle(files)
        train_files = files[:int(0.9*len(files))]
        valid_files = files[int(0.9*len(files)):]
        
    else:
        train_files = glob.glob(train_path + '/*.zip')
        valid_files = glob.glob(valid_path + '/*.zip')

    # Randomly shuffle the train and valid files
    random.shuffle(train_files)
    random.shuffle(valid_files)
    
    #train_files = list(dat.loc[dat.n_vars == 48,"fn"])
    if args.train_lim is not None:
        random.shuffle(train_files)
        train_files = train_files[:args.train_lim]
    debug = False
    if debug:
        files = [sorted(glob.glob(i+'/data/000000/*.npz'))
            for i in sorted(glob.glob('./wildebeest/data/test_100k_*'))]
        # train_files = files[:int(len(files) * 0.9)]
        dataloader_old = DataLoader(MCTSDatasetOld(files[:int(len(files) * 0.9)]), 
                                    batch_size=config['optimization']['bs'],
                                    shuffle=True, num_workers=10, collate_fn=reindex)

    cs = None
    if config['optimization'].get('curriculum', False):
        print("Using curriculum learning")
        dat = pd.read_pickle('filelist.pkl')
        weighter = FixedWeighted(dat, config, debug=True, sigmoid=False)
        cs = CurriculumSampler(dat, weighter, config, num_samples=1000)
        dataloader = DataLoader(MCTSDataset(train_files, value_dict=value_dict,softmax_counts=args.softmax_counts,weight_counts_by_q=args.weight_counts_by_q, temperature=args.temperature,q_value_head=args.q_value_head), batch_size=config['optimization']['bs'],
                            sampler=cs, num_workers=10, collate_fn=reindex)
    elif args.upsample_by_depth:
        depth_weights = []
        if solver_depths:
            max_depth = max(solver_depths.values())
            for train_file in train_files:
                depth_weights.append(2 ** (max_depth - solver_depths[train_file]))
        else:
            for train_file in train_files:
                depth_weight = int(os.path.basename(train_file).split('_')[4]) # random_300_68060_100000_2950_CAPPED.zip
                if depth_weight <= 5:
                    depth_weights.append(2 ** (5 - depth_weight))
                else:
                    depth_weights.append(1)
        wrs = WeightedRandomSampler(depth_weights, len(depth_weights))
        dataloader = DataLoader(MCTSDataset(train_files, value_dict=value_dict, softmax_counts=args.softmax_counts,weight_counts_by_q=args.weight_counts_by_q, temperature=args.temperature,q_value_head=args.q_value_head), batch_size=config['optimization']['bs'],
                            sampler=wrs, shuffle=False, num_workers=10, collate_fn=reindex)
    else:
        cs = None
        dataloader = DataLoader(MCTSDataset(train_files, value_dict=value_dict, softmax_counts=args.softmax_counts,weight_counts_by_q=args.weight_counts_by_q, temperature=args.temperature,q_value_head=args.q_value_head), batch_size=config['optimization']['bs'],
                             shuffle=True, num_workers=10, collate_fn=reindex)
    if debug:
        for b_old in dataloader_old:
            break
        for b in dataloader:
            break
        import pdb; pdb.set_trace()
    
    random.shuffle(valid_files)
    if args.valid_lim is not None:
        valid_files = valid_files[:args.valid_lim]
    
    testloader = DataLoader(MCTSDataset(valid_files, value_dict=value_dict, softmax_counts=args.softmax_counts,weight_counts_by_q=args.weight_counts_by_q, temperature=args.temperature,q_value_head=args.q_value_head), batch_size=config['optimization']['bs'],
                            shuffle=False, num_workers=10, collate_fn=reindex)
    print(
        f"Starting training with {len(dataloader)} training examples & {len(testloader)} test examples.")


    # Get training and validation CNFs from depth 1 data points
    if solver_depths:
        train_cnf_files = [f for f in train_files if solver_depths[f] == 1][:args.num_cnfs]
        valid_cnf_files = [f for f in valid_files if solver_depths[f] == 1][:args.num_cnfs]
    else:
        print('Warning: no solver depths found, using depth 1 from filename.')
        train_cnf_files = []
        for f in train_files:
            if '_1_' in f:
                train_cnf_files.append(f)
        train_cnf_files = train_cnf_files[:args.num_cnfs]
        valid_cnf_files = []
        for f in valid_files:
            if '_1_' in f:
                valid_cnf_files.append(f)
        valid_cnf_files = valid_cnf_files[:args.num_cnfs]
    train_cnfs = []
    for train_file in tqdm(train_cnf_files, desc='Reading training CNFs'):
        data_point = torch.load(train_file)
        values = data_point[0]
        indices = data_point[1]
        train_cnfs.append(torch_to_cnf(values, indices, cnf_filename=train_file.replace('.zip', '.cnf')))
    
    validation_cnfs = []
    for valid_file in tqdm(valid_cnf_files, desc='Reading validation CNFs'):
        data_point = torch.load(valid_file)
        values = data_point[0]
        indices = data_point[1]
        validation_cnfs.append(torch_to_cnf(values, indices, cnf_filename=valid_file.replace('.zip', '.cnf')))


    # this is a little irritating -  pytorch lightning doesn't work with torchscript so we create 
    # the base model separately and then train it with pytorch lightning 
    base_model = Policy(n_layers=config["model"]["layers"], units=config["model"]["units"], 
                        in_width=2, resnet=config["model"].get('resnet', False), 
                        dropout=config["model"]["dropout"],
                        num_size_features= 3 if args.use_size_features else 0,
                        pool_before_mlp=args.pool_before_mlp,
                        attention_pooling=args.attention_pooling,
                        use_value_head=args.q_value_head)

    # Convert model to string and send to cometml
    model_str = str(base_model)#.to_string()
    experiment.set_model_graph(model_str)

    # make example data
    index = torch.tensor([[0, 0],[1,0],[20, 20]]).long()
    new_values = torch.tensor([[0.], [1.], [0.]]).float()
    tracing_data = torch.sparse_coo_tensor(index.T, new_values)
    # Move to GPU if available
    # if torch.cuda.is_available():
    #     tracing_data = tracing_data.cuda()
    #     base_model.cuda()
    traced = torch.jit.script(base_model, (tracing_data,))
    # save a base model #worked here
    traced.save(tracing_path + 'policy_model-no_training.zip')

    # if "validation_cnfs" in config:
    #     with open(config["validation_cnfs"],'r') as f:
    #             validation_cnfs = f.readlines()
    # else:
    #     validation_cnfs = []
    name = f'{args.config}-{args.seed}'
    traced_model_path = os.path.join(tracing_path, f'last_traced_{name}.zip')

    if cp is not None:
        print(f"Loading model from checkpoint {cp}")
        model = PolicyModel(base_model, lr=config["optimization"]["lr"]).load_from_checkpoint(cp, 
                            module=base_model, lr=config["optimization"]["lr"], 
                            sampler=cs, 
                            comet_exp=experiment, 
                            validation_cnfs=validation_cnfs, 
                            training_cnfs=train_cnfs,
                            traced_model_path=traced_model_path, 
                            tracing_data=tracing_data, 
                            validation_cutoff=args.valid_cutoff,
                            num_variables_for_rollout=args.num_variables_for_rollout,
                            value_loss = args.value_loss,
                            counts_loss = args.counts_loss,
                            q_value_loss = args.q_value_head,
                            subsolver_loss = args.subsolver_loss,
                            value_loss_weight = args.value_loss_weight,
                            truncate_loss = args.truncate_loss)
        traced = torch.jit.script(base_model, (tracing_data,))
        traced.save(tracing_path + f"policy_model-cp-{cp.split('/')[-1].split('ckpt')[0]}.zip")
    else:
        model = PolicyModel(base_model, lr=config["optimization"]["lr"], sampler=cs,
                            comet_exp=experiment, 
                            validation_cnfs=validation_cnfs,  
                            training_cnfs=train_cnfs,
                            traced_model_path=traced_model_path, 
                            tracing_data=tracing_data, 
                            validation_cutoff=args.valid_cutoff, 
                            num_variables_for_rollout=args.num_variables_for_rollout,
                            value_loss = args.value_loss,
                            q_value_loss = args.q_value_head,
                            counts_loss = args.counts_loss,
                            value_loss_weight = args.value_loss_weight,
                            truncate_loss = args.truncate_loss)

    name = f'{args.config}-{args.seed}'
    # Setup logger
    logger = TensorBoardLogger('tb_logs', name=name)
    # Save the current training script with the logs so we can always check if there 
    # are any modifications that need to be reproduced
    current_train_script = pathlib.Path(os.path.realpath(__file__))
    new_path = current_train_script.parents[0] / 'tb_logs' / name / f'version_{logger.version}'
    os.makedirs(new_path, exist_ok=True)
    copyfile(current_train_script, new_path / current_train_script.name)

    callbacks = [ResetAccCallback(),checkpoint_callback]

    if config["optimization"]["val_freq"] >= 1:
        trainer = pl.Trainer(gpus=1, callbacks=callbacks,
                            check_val_every_n_epoch=int(config["optimization"]["val_freq"]),
                            logger=logger, max_epochs=args.epochs, 
                            auto_lr_find=True, 
                            accumulate_grad_batches=config["optimization"]["accumulate_grad_batches"],
                            enable_progress_bar = False)
    else:
        trainer = pl.Trainer(gpus=1, callbacks=callbacks,
                            val_check_interval=config["optimization"]["val_freq"],
                            logger=logger, max_epochs=args.epochs, 
                            auto_lr_find=True,
                            accumulate_grad_batches=config["optimization"]["accumulate_grad_batches"],
                            enable_progress_bar = False)


    trainer.fit(model, dataloader, val_dataloaders=testloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train / evaluate")
    parser.add_argument('-c','--config', default='test', help='Config file to describe architecture and train/validation data. See ./config/test.py for a template')
    parser.add_argument('-d','--datapath', default='')
    parser.add_argument('--cp', default=None, help='Checkpoint to restart from. Checkpoint architecture must be same as --config')
    parser.add_argument('--valid_lim', type=int, default=None, help='Set to constrain the number of validation instanes used. If not set, full set is used.')
    parser.add_argument('--train_lim', type=int, default=None, help='Set to constrain the number of training instanes used. If not set, full set is used.')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--epochs', type=int, default=1000, help='Number of passes through training set before terminating')
    parser.add_argument('--name', '-n', default='no_name')
    parser.add_argument('--only_unsat', action='store_true', help="Filter out all SAT data.") # TODO: Set to true if parameter set
    parser.add_argument('--valid_metric', default='val_decisions', choices=['val_loss', 'val_decisions'], help="Which metric to to use for selecting checkpoints to save.") 
    parser.add_argument('--valid_cutoff', type=int, default=1000, help='Number of solver decisions before cutting off run')
    parser.add_argument('--num_variables_for_rollout', type=int, default=None, help='Size of problem where subsolver should be used')
    parser.add_argument('--value_loss', action='store_true', help='Use value net')
    parser.add_argument('--counts_loss', action='store_true', help='Use counts net')
    parser.add_argument('--counts_accuracy_loss', action='store_true', help='Use counts accuracy net')
    parser.add_argument('--subsolver_loss', action='store_true', help='Use subsolver net')
    parser.add_argument('--num_cnfs', type=int, default=20, help='Number of CNFs to use for training/validation')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--units', type=int, default=None, help='Number of units in hidden layers')
    parser.add_argument('--layers', type=int, default=None, help='Number of hidden layers')
    parser.add_argument('--grad_batches', type=float, default=None, help='Learning rate')
    parser.add_argument('--value_loss_weight', type=float, default=0.0001, help='Weight of value loss')

    # Database parameters
    parser.add_argument('--use_mcts_db', action='store_true', help='Use MCTS to get data')
    parser.add_argument('--experiment_name', action='store_true', help='Experiment name in database')

    # Architecture parameters
    parser.add_argument('--use_size_features', action='store_true', help='Use size features')
    parser.add_argument('--pool_before_mlp', action='store_true', help='Use mean before mlp')
    parser.add_argument('--attention_pooling', action='store_true', help='Use attention pooling')

    # Sampling params
    parser.add_argument('--upsample_by_depth', action='store_true', help='Sample inversely proportion to exponetial of depth')

    # Counts/loss parameters
    parser.add_argument('--truncate_loss', type=int, default=None, help='Truncate loss at this index')
    parser.add_argument('--softmax_counts', action='store_true', help='Use softmax on counts')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for softmax')
    parser.add_argument('--q_value_head', action='store_true', help='Use q values as counts')
    parser.add_argument('--weight_counts_by_q', action='store_true', help='Weight counts by q values')

    args = parser.parse_args()
    main(args)