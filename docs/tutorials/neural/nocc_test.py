import shutup
shutup.please()

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import warnings
warnings.filterwarnings('ignore')

# import rootutils
# ROOT = rootutils.setup_root(indicator='README.md', search_from=os.path.abspath(''), pythonpath=True, cwd=True)

import jax
import jax.numpy as jnp
import optax
from jaxtyping import ArrayLike, Float
import numpy as np
# GLOBAL_KEY = jax.random.key(42)

import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use(['science', 'notebook'])

# Lagrangian Potentials

import sys

sys.path.append("/home/nazar/projects/loop_ott/src")

from ott2.neural.methods.lagrangian import *