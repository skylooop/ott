import os

# CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES = "4,5,6,7"
# CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import sys

sys.path.insert(0, "/home/jovyan/d-shlenskii/repos/skyloop/ott/src")

import warnings

warnings.filterwarnings('ignore')

import functools
import json
import pickle
from abc import abstractmethod
from argparse import ArgumentParser

from jaxtyping import ArrayLike, Float
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

import flax.linen as nn
import optax
from flax.struct import PyTreeNode

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

from ott2.neural.methods.flows.dynamics import LagrangianFlow
from ott2.neural.methods.nocc import NeuralOC
from ott2.neural.networks.resnet_d import ResNet_D

GLOBAL_KEY = jax.random.key(42) 

parser = ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--ckpt_name", type=str)
args = parser.parse_args()

config_path = args.config
CONFIG = OmegaConf.load(config_path)

SAVE_DIR = config_path[:-len(".yaml")]
MODEL_PATH = f"{SAVE_DIR}/model/{args.ckpt_name}"

# Data

class MyImageFolder(ImageFolder):
    def __getitem__(self, idx: int):
        return super().__getitem__(idx)[0].numpy()

# Network

class ResNetDwTime(nn.Module):
    size: int = 64 
    nlayers: int = 4
    nc: int = 3
    nfilter: int = 64
    nfilter_max: int = 512
    t_embedding_dim: int = 128

    @nn.compact
    def __call__(self, t, x, train=True):
        b_size = t.shape[0]

        x = x.reshape(-1, self.nc, self.size, self.size)

        t_emb = nn.Dense(self.size**2)(t[:, None]) # [b, size**2]
        t_emb = t_emb.reshape(b_size, 1, self.size, self.size)

        x_with_t = jnp.concatenate([x, t_emb], axis=1)
        return ResNet_D(
            size=self.size,
            nlayers=self.nlayers,
            nc=self.nc+1,
            nfilter=self.nfilter,
            nfilter_max=self.nfilter_max,
        )(x_with_t)

# Potential
class LagrangianPotentialFree(PyTreeNode):
    @abstractmethod
    def __call__(self, x):
        return 0.
potential = LagrangianPotentialFree()


# training params
batch_size = CONFIG["batch_size"]
n_iters = CONFIG["n_training_iters"]
collect_buffer_iters = CONFIG["collect_buffer_iters"]
update_potential_every = CONFIG["update_potential_every"]
eval_every = CONFIG["eval_every"]

# data preparation
img_size, nc = 64, 3
test_ratio = 0.1

## celeba female dataset
path = "/home/jovyan/nazar/celeba_female"
attrs_path = "/home/jovyan/nazar/list_attr_celeba.txt" 
transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
celeba_female_dataset = MyImageFolder(path, transform=transform)

with open(attrs_path, 'r') as f:
    lines = f.readlines()[1:]
idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
print("celeba", len(idx), len(lines), len(celeba_female_dataset))

test_size = int(len(idx) * test_ratio)
train_idx, test_idx = idx[:-test_size], idx[-test_size:]

train_celeba_female_dataset = Subset(celeba_female_dataset, train_idx)
test_celeba_female_dataset = Subset(celeba_female_dataset, test_idx)

celeba_female_dataset = Subset(celeba_female_dataset, idx)

test_batch_size = 256
test_celeba_female_loader = DataLoader(
    test_celeba_female_dataset,
    shuffle=False,
    batch_size=test_batch_size,
    num_workers=4,
)

# model

net = ResNetDwTime(size=img_size, nc=nc)
num_iterations = n_iters
noc = NeuralOC(
    input_dim=nc*img_size**2,
    value_model=net,
    optimizer=optax.chain(
        optax.clip(max_delta=1.),
        optax.adam(**CONFIG["optimizer"]),
    ),
    control_steps=30,
    reg_weight=CONFIG["reg_weight"],
    control_weight=CONFIG["control_weight"],
    acc_weight=0.,
    potential_weight=0.,
    flow=LagrangianFlow(sigma=0.1, potential=potential),
    key=GLOBAL_KEY,
    batch_size=batch_size,
    load_dir=None,
)
with open(MODEL_PATH, "rb") as file:
    params = pickle.load(file)
noc.state = noc.state.replace(params=params)


idx = 0
for inp_batch in tqdm(test_celeba_female_loader):
    img_size = inp_batch.shape[-1]
    sample_batch = noc.transport(inp_batch.numpy())[1][-1].x.reshape(-1, 3, img_size, img_size)
    for x in sample_batch:
        print(f"{x.shape}")
        im = Image.fromarray(np.asarray(x))
        im.save(f"samples/image_{idx}.jpeg")
        idx += 1