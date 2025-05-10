import os

# CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES = "0,1"
# CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import sys

sys.path.insert(0, "/home/nazar/projects/hota_images/src")

import warnings

warnings.filterwarnings('ignore')

import functools
import json
from abc import abstractmethod
from argparse import ArgumentParser

from jaxtyping import ArrayLike, Float
from omegaconf import OmegaConf

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
parser.add_argument("--load_model", action="store_true")
args = parser.parse_args()

config_path = args.config
CONFIG = OmegaConf.load(config_path)

SAVE_DIR = config_path[:-len(".yaml")]
SAVE_GENS_DIR = f"{SAVE_DIR}/gens"
SAVE_MODEL_DIR = f"{SAVE_DIR}/model"
for _dir in [SAVE_DIR, SAVE_GENS_DIR, SAVE_MODEL_DIR]:
  if not os.path.exists(_dir):
      os.makedirs(_dir)

OmegaConf.save(config=CONFIG, f=f"{SAVE_DIR}/config.yaml")

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


# training params
batch_size = CONFIG["batch_size"]
n_iters = CONFIG["n_training_iters"]
collect_buffer_iters = CONFIG["collect_buffer_iters"]
update_potential_every = CONFIG["update_potential_every"]
eval_every = CONFIG["eval_every"]

# data preparation
img_size, nc = 64, 3
test_ratio = 0.1

## anime dataset
path = "/home/nazar/projects/aligned_anime_faces"
transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
anime_dataset = MyImageFolder(path, transform=transform)

idx = list(range(len(anime_dataset)))
test_size = int(len(idx) * test_ratio)
train_idx, test_idx = idx[:-test_size], idx[-test_size:]

train_anime_dataset = Subset(anime_dataset, train_idx)
test_anime_dataset = Subset(anime_dataset, test_idx)

## celeba female dataset
path = "/home/nazar/projects/celeba_female"
attrs_path = "/home/nazar/projects/list_attr_celeba.txt" 
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

## loader
train_ot_loader = OTLoader(
    src_ds=train_celeba_female_dataset,
    trg_ds=train_anime_dataset,
    flatten_flag=True,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
)
train_ot_loader = iter(train_ot_loader)

potential_data_loader = iter(train_ot_loader)
potential = LagrangianPotentialFree()

# evaluation function
import tools.jax_inception as inception
from tools.fid import (
    calculate_frechet_distance,
    get_loader_stats,
    get_pushed_loader_stats,
)

inception_net = inception.InceptionV3(pretrained=True)
rng = jax.random.PRNGKey(0)
inception_params = inception_net.init(rng, jnp.ones((1, 299, 299, 3))) # TODO: WHY?
inception_apply = jax.jit(functools.partial(inception_net.apply, train=False))

mu_path = "experiments/mu_data.npy"
sigma_path = "experiments/sigma_data.npy"
if not os.path.exists(mu_path) or not os.path.exists(sigma_path):
    test_anime_loader = DataLoader(
        test_anime_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
    )
    mu_data, sigma_data = get_loader_stats(test_anime_loader, inception_apply, inception_params, batch_size=128, n_epochs=1, verbose=True, classes=False)
    with open(mu_path, "wb") as file:
        np.save(file, mu_data)
    with open(sigma_path, "wb") as file:
        np.save(file, sigma_data)
else:
    with open(mu_path, "rb") as file:
        mu_data = np.load(file)
    with open(sigma_path, "rb") as file:
        sigma_data = np.load(file)

test_batch_size = 512
test_celeba_female_loader = DataLoader(
    test_celeba_female_dataset,
    shuffle=False,
    batch_size=test_batch_size,
    num_workers=4,
)

def callback(step, training_logs, transport):
    # # # Compute FID
    mu, sigma = get_pushed_loader_stats(
        transport, test_celeba_female_loader, inception_apply, inception_params, batch_size=test_batch_size, verbose=True, upgrade=False
    )
    fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
    with open(f"{SAVE_DIR}/FID.txt", "a") as file:
        file.write(f"{fid}\n")

    # Visualization
    ## sample batch
    pi0 = next(potential_data_loader)['src_lin']
    pi1 = next(potential_data_loader)['tgt_lin']

    ## generate batch
    cost, trajs = transport(pi0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ## plot samples
    def to_range(image):
        image -= image.min()
        return jnp.clip(image / image.max(), 0., 1.)

    axes[0].imshow(to_range(pi0[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[0].set_title('Source')
    axes[0].axis('off')

    axes[1].imshow(to_range(trajs[-1].x[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[1].set_title('SourceMapped')
    axes[1].axis('off')

    axes[2].imshow(to_range(pi1[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[2].set_title('Target')
    axes[2].axis('off')

    plt.savefig(f"{SAVE_GENS_DIR}/step_{step}.png")

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
    load_dir=SAVE_MODEL_DIR if args.load_model else None,
)

logs = noc(
    potential_data_loader,
    n_iters=num_iterations,
    rng=GLOBAL_KEY,
    callback=callback,
    collect_buffer_iters=collect_buffer_iters,
    update_potential_every=update_potential_every,
    eval_every=eval_every,
    save_dir=SAVE_MODEL_DIR,
)

with open(f"{SAVE_DIR}/logs.json", 'w') as f:
    json.dump(logs, f)
