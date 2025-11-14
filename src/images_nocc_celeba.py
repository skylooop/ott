import os

# CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES = "2,3"
# CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import sys

sys.path.insert(0, "/home/jovyan/d-shlenskii/repos/ott/src")
# sys.path.insert(0, "/home/jovyan/nazar/hota_image/src")

import warnings

warnings.filterwarnings('ignore')

import functools
import json
from abc import abstractmethod
from argparse import ArgumentParser

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
from ott2.neural.networks.resnet_d import ResNet_D, ConvBlock
import torch, gc
gc.collect()
torch.cuda.empty_cache()
jax.clear_caches()


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
    nlayers: int = CONFIG["n_layers"]
    nc: int = 3
    nfilter: int = 100
    nfilter_max: int = 512
    conv_block_cls= ConvBlock


    @nn.compact
    def __call__(self, t, x, train=True):
        b_size = t.shape[0]
        D = self.size

        x = x.reshape(-1, self.nc, self.size, self.size)

        ntf = CONFIG["n_time_freq"]
        t_pos = jnp.arange(1, ntf) * t
        t_pos = jnp.concatenate([jnp.sin(t_pos) / jnp.arange(1, ntf), jnp.cos(t_pos) / jnp.arange(1, ntf)], -1)

        b = nn.Dense(self.size ** 2)(t_pos) # [b, size**2]
        b = b.reshape(b_size, 1, self.size, self.size)

        x_with_t = jnp.concatenate([x, b], axis=1)
        
        return ResNet_D(
            size=self.size,
            nlayers=self.nlayers,
            nc=self.nc + 1,
            nfilter=self.nfilter,
            nfilter_max=self.nfilter_max,
        )(x_with_t)

# Potential
class LagrangianPotentialFree(PyTreeNode):
    @abstractmethod
    def __call__(self, x):
        return 0.


print(f"CONFIG")
print(CONFIG)
# training params
batch_size = CONFIG["batch_size"]
n_iters = CONFIG["n_training_iters"]
collect_buffer_iters = CONFIG["collect_buffer_iters"]
update_potential_every = CONFIG["update_potential_every"]
eval_every = CONFIG["eval_every"]

# data preparation
img_size, nc = 64, 3
test_ratio = 0.1

## celeba dataset (contains both males and females)
path = "/home/jovyan/nazar/celeba_female"  # Your folder with all CelebA images
attrs_path = "/home/jovyan/nazar/list_attr_celeba.txt"
transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
celeba_dataset = MyImageFolder(path, transform=transform)

with open(attrs_path, 'r') as f:
    lines = f.readlines()[1:]
    
# Get male and female indices
female_idx = [i for i, line in enumerate(lines) if len((parts := line.strip().split())) >= 41 and parts[20] == '-1']
male_idx = [i for i, line in enumerate(lines) if len((parts := line.strip().split())) >= 41 and parts[20] == '1']

print(f"Found {len(female_idx)} female images, {len(male_idx)} male images")
print("Total celeba images:", len(celeba_dataset))

# Split into train/test for both genders
test_size_female = int(len(female_idx) * test_ratio)
test_size_male = int(len(male_idx) * test_ratio)

train_female_idx, test_female_idx = female_idx[:-test_size_female], female_idx[-test_size_female:]
train_male_idx, test_male_idx = male_idx[:-test_size_male], male_idx[-test_size_male:]

print(f"Train - Female: {len(train_female_idx)}, Male: {len(train_male_idx)}")
print(f"Test - Female: {len(test_female_idx)}, Male: {len(test_male_idx)}")

# Create datasets
## SOURCE: Male faces (we'll transform these to female)
train_male_dataset = Subset(celeba_dataset, train_male_idx)
test_male_dataset = Subset(celeba_dataset, test_male_idx)

## TARGET: Female faces (target domain)
train_female_dataset = Subset(celeba_dataset, train_female_idx)
test_female_dataset = Subset(celeba_dataset, test_female_idx)

## loader - UPDATED FOR MALE->FEMALE
train_ot_loader = OTLoader(
    src_ds=train_male_dataset,      # SOURCE: Male faces
    trg_ds=train_female_dataset,    # TARGET: Female faces
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
    get_pushed_loader_stats_torch,
)

inception_net = inception.InceptionV3(pretrained=True)
rng = jax.random.PRNGKey(0)
inception_params = inception_net.init(rng, jnp.ones((1, 299, 299, 3))) # TODO: WHY?
inception_apply = jax.jit(functools.partial(inception_net.apply, train=False))

# UPDATED FID STATISTICS - Use FEMALE faces as target distribution
mu_path = "/home/jovyan/nazar/hota_image/src/experiments/mu_data_female_target.npy"
sigma_path = "/home/jovyan/nazar/hota_image/src/experiments/sigma_data_female_target.npy"
if not os.path.exists(mu_path) or not os.path.exists(sigma_path):
    print(f"{mu_path=} does not exists. Computing stats for FID (Female target)")
    test_female_loader = DataLoader(
        test_female_dataset,  # Use female faces as target distribution
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
    )
    mu_data, sigma_data = get_loader_stats(test_female_loader, inception_apply, inception_params, batch_size=128, n_epochs=1, verbose=True, classes=False)
    with open(mu_path, "wb") as file:
        np.save(file, mu_data)
    with open(sigma_path, "wb") as file:
        np.save(file, sigma_data)
else:
    with open(mu_path, "rb") as file:
        mu_data = np.load(file)
    with open(sigma_path, "rb") as file:
        sigma_data = np.load(file)

test_batch_size = 256
# UPDATED TEST LOADER - Use MALE faces as source for transformation
test_male_loader = DataLoader(
    test_male_dataset,  # We'll transform male faces to female
    shuffle=False,
    batch_size=test_batch_size,
    num_workers=4,
)

def callback(step, training_logs, transport):
    # Compute FID - pushing MALE faces to FEMALE domain
    mu, sigma = get_pushed_loader_stats(
        transport, test_male_loader, inception_apply, inception_params, batch_size=test_batch_size, verbose=True, upgrade=False
    )
    fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)

    print("FID (Male->Female):", fid)

    with open(f"{SAVE_DIR}/FID.txt", "a") as file:
        file.write(f"{fid}\n")

    # Visualization
    ## sample batch - now male source and female target
    male_batch = next(potential_data_loader)['src_lin']  # Male faces
    female_batch = next(potential_data_loader)['tgt_lin']  # Female faces

    ## generate batch - transform male to female
    cost, trajs = transport(male_batch)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ## plot samples
    def to_range(image):
        image = image * 0.5 + 0.5
        return jnp.clip(image, 0., 1.)

    # Row 1: Source males and their transformations
    axes[0, 0].imshow(to_range(male_batch[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[0, 0].set_title('Male Source')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(to_range(trajs[-1].x[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[0, 1].set_title('Male->Female')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(to_range(female_batch[0]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[0, 2].set_title('Target Female')
    axes[0, 2].axis('off')

    # Row 2: Additional samples for better visualization
    axes[1, 0].imshow(to_range(male_batch[1]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[1, 0].set_title('Male Source 2')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(to_range(trajs[-1].x[1]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[1, 1].set_title('Male->Female 2')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(to_range(female_batch[1]).reshape(nc, img_size, img_size).transpose(1, 2, 0))
    axes[1, 2].set_title('Target Female 2')
    axes[1, 2].axis('off')

    print("Male source stats:", male_batch[0].min(), male_batch[0].mean(), male_batch[0].max())
    print("Female target stats:", female_batch[0].min(), female_batch[0].mean(), female_batch[0].max())
    print("Transformed stats:", trajs[-1].x[0].min(), trajs[-1].x[0].mean(), trajs[-1].x[0].max())
    print()

    plt.tight_layout()
    plt.savefig(f"{SAVE_GENS_DIR}/step_{step}.png")
    plt.close()  # Important to avoid memory leaks

net = ResNetDwTime(size=img_size, nc=nc)
num_iterations = n_iters
noc = NeuralOC(
    input_dim=nc*img_size**2,
    value_model=net,
    optimizer=optax.chain(
        # optax.clip(max_delta=1.),
        optax.adam(**CONFIG["optimizer"]),
    ),
    control_steps=CONFIG["control_steps"],
    cost_mult = CONFIG["cost_mult"],
    backward_batch_size = CONFIG["backward_batch_size"], 
    use_dual_abs_mult = CONFIG["use_dual_abs_mult"],
    control_weight=CONFIG["control_weight"],
    potential_weight=CONFIG["potential_weight"],
    flow=LagrangianFlow(sigma=CONFIG["sigma"], potential=potential),
    key=GLOBAL_KEY,
    load_dir=SAVE_MODEL_DIR if args.load_model else None,
)

logs = noc(
    potential_data_loader,
    n_iters=num_iterations,
    collect_buffer_iters = collect_buffer_iters,
    buffer_update_size = CONFIG["buffer_update_size"],
    update_potential_every=update_potential_every,
    buffer_size=CONFIG["buffer_size"],
    rng=GLOBAL_KEY,
    callback=callback,
    eval_every=eval_every,
    save_dir=SAVE_MODEL_DIR,
)

with open(f"{SAVE_DIR}/logs.json", 'w') as f:
    json.dump(logs, f)