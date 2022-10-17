# standard lib imports
from datetime import date
import numpy as np
from shutil import copyfile
import pandas as pd
from typing import Optional
import sys

# torch imports
from comet_ml import Experiment
import torch
from torch import Tensor

from train import PolicyModel, Policy

checkpoint_path = sys.argv[1]


base_model = Policy(n_layers=4, units=512,in_width=2,resnet=False)
model = PolicyModel.load_from_checkpoint(checkpoint_path, module=base_model)

# make example data
index = torch.tensor([[0, 0],
                    [1,0],
                    [20, 20]]).long()
new_values = torch.tensor([[0.], [1.], [0.]]).float()
tracing_data = torch.sparse_coo_tensor(index.T, new_values)


model.module.eval()
#setattr(model.module, '_is_full_backward_hook', list())
# remove_attributes = []
# for key, value in vars(model.module).items():
#     if value is None:
#         print('Removed attribute %s' % key)
#         setattr(model.module, key, bool)
#         remove_attributes.append(key)

# Convert to CPU
model.module.to('cpu')

# for key in remove_attributes:
#     setattr(model.module, key, False)
traced = torch.jit.script(model.module, (tracing_data,))
output_tracing_path = checkpoint_path.replace('ckpt','zip')
traced.save(output_tracing_path)