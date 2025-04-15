
import sys

sys.path.insert(0, "/home/jovyan/d-shlenskii/repos/skyloop/ott/src")

import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

import warnings

warnings.filterwarnings('ignore')

import json
from abc import abstractmethod

from jaxtyping import ArrayLike, Float

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
EVAL_PATH = "/home/jovyan/d-shlenskii/repos/skyloop/ott/docs/tutorials/neural/images_nocc_log"


# Data

class InfiniteLoaderWrapper:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.loader_it = iter(loader)
    
    def __iter__(self):
        self.loader_it = iter(self.loader)
        return self

    def __next__(self):
        try:
            batch = next(self.loader_it)
        except StopIteration:
            self.loader_it = iter(self.loader)
            batch = next(self.loader_it)
        return batch

class OTLoader:
    def __init__(
        self,
        src_ds: Dataset,
        trg_ds: Dataset,
        flatten_flag: bool = False,
        **torch_dataloader_kwargs,
    ):
        def collate_fn(batch: tuple[np.ndarray]):
            return np.stack(batch)

        self.src_loader = InfiniteLoaderWrapper(DataLoader(src_ds, collate_fn=collate_fn, **torch_dataloader_kwargs))
        self.trg_loader = InfiniteLoaderWrapper(DataLoader(trg_ds, collate_fn=collate_fn, **torch_dataloader_kwargs))
        self.flatten_flag = flatten_flag

    def __iter__(self):
        self.src_loader = iter(self.src_loader)
        self.trg_loader = iter(self.trg_loader)
        return self

    def __next__(self):
        src_batch = jnp.asarray(next(self.src_loader))
        tgt_batch = jnp.asarray(next(self.trg_loader))
        if self.flatten_flag:
            b_size = src_batch.shape[0]
            src_batch = src_batch.reshape(b_size, -1)
            tgt_batch = tgt_batch.reshape(b_size, -1)
        return {
            "src_lin": src_batch,
            "tgt_lin": tgt_batch,
        }

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


# training params
batch_size = 16
n_iters = 1_000_000
collect_buffer_iters = 10_000
update_potential_every = 4
eval_every = 2_000

# data preparation
img_size, nc = 64, 3

## anime dataset
path = "/home/jovyan/nazar/aligned_anime_faces"
transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
anime_dataset = MyImageFolder(path, transform=transform)

## celeba female dataset
path = "/home/jovyan/nazar/celeba_female"
attrs_path = "/home/jovyan/nazar/list_attr_celeba.txt" 
transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
celeba_female_dataset = MyImageFolder(path, transform=transform)

with open(attrs_path, 'r') as f:
    lines = f.readlines()[1:]
idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
print("celeba", len(idx), len(lines), len(celeba_female_dataset))

celeba_female_dataset = Subset(celeba_female_dataset, idx)

## loader
ot_loader = OTLoader(
    src_ds=anime_dataset,
    trg_ds=celeba_female_dataset,
    flatten_flag=True,
    shuffle=True,
    batch_size=batch_size,
)
ot_loader = iter(ot_loader)

potential_data_loader = iter(ot_loader)
potential = LagrangianPotentialFree()

# evaluation function
def callback(step, training_logs, transport):
    clear_output()
    pi0 = next(potential_data_loader)['src_lin']
    pi1 = next(potential_data_loader)['tgt_lin']

    cost, trajs = transport(pi0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def to_range(image):
        image -= image.min()
        return image / image.max()

    axes[0].imshow(to_range(pi0[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[0].set_title('Source')
    axes[0].axis('off')

    axes[1].imshow(to_range(trajs[-1].x[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[1].set_title('SourceMapped')
    axes[1].axis('off')

    axes[2].imshow(to_range(pi1[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[2].set_title('Target')
    axes[2].axis('off')
    
    plt.savefig(f"{EVAL_PATH}/step_{step}.png")
     
net = ResNetDwTime(size=img_size, nc=nc)
num_iterations = n_iters
lr_schedule = optax.cosine_decay_schedule(
    init_value=2e-5, decay_steps=num_iterations, alpha=1e-2
)
noc = NeuralOC(
    input_dim=nc*img_size**2, 
    value_model=net, 
    optimizer=optax.adam(learning_rate=lr_schedule), 
    control_steps=30,
    reg_weight=0.001,
    control_weight=0.1, 
    acc_weight=0.5,
    potential_weight=0., 
    flow=LagrangianFlow(sigma=0.1, potential=potential), 
    key=GLOBAL_KEY,
    batch_size=batch_size,
)

logs = noc(
    potential_data_loader,
    n_iters=num_iterations,
    rng=GLOBAL_KEY,
    callback=callback,
    collect_buffer_iters=collect_buffer_iters,
    update_potential_every=update_potential_every,
    eval_every=eval_every,
)

with open(f"{EVAL_PATH}/logs.json", 'w') as f:
    json.dump(logs, f)