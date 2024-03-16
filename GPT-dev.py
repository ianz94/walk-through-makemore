import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel, so that we can make the best use of our GPU?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embed = 64 # how dimensional is the information of each token
n_head = 4
n_layer = 4
dropout = 0.0 # to reduce overfitting

torch.manual_seed(1337)

# training material
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

