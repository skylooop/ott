# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
from typing import Iterator, Literal, NamedTuple, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["create_gaussian_mixture_samplers", "Dataset", "GaussianMixture"]

from ott import utils

Name_t = Literal["simple", "circle", "square_five", "square_four"]


class Dataset(NamedTuple):
  r"""Samplers from source and target measures.

  Args:
    source_iter: loader for the source measure
    target_iter: loader for the target measure
  """
  source_iter: Iterator[jnp.ndarray]
  target_iter: Iterator[jnp.ndarray]


@dataclasses.dataclass
class GaussianMixture:
  """A mixture of Gaussians.

  Args:
    name: the name specifying the centers of the mixture components:

      - ``simple`` - data clustered in one center,
      - ``circle`` - two-dimensional Gaussians arranged on a circle,
      - ``square_five`` - two-dimensional Gaussians on a square with
        one Gaussian in the center, and
      - ``square_four`` - two-dimensional Gaussians in the corners of a
        rectangle

    batch_size: batch size of the samples
    rng: initial PRNG key
    scale: scale of the Gaussian means
    std: the standard deviation of the individual Gaussian samples
  """
  name: Name_t
  batch_size: int
  rng: jax.Array
  scale: float = 5.0
  std: float = 0.5

  def __post_init__(self) -> None:
    gaussian_centers = {
        "simple":
            np.array([[0, 0]]),
        "circle":
            np.array([
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            ]),
        "square_five":
            np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]),
        "square_four":
            np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    }
    if self.name not in gaussian_centers:
      raise ValueError(
          f"{self.name} is not a valid dataset for GaussianMixture"
      )
    self.centers = gaussian_centers[self.name]

  def __iter__(self) -> Iterator[jnp.array]:
    """Random sample generator from Gaussian mixture.

    Returns:
      A generator of samples from the Gaussian mixture.
    """
    return self._create_sample_generators()

  def _create_sample_generators(self) -> Iterator[jnp.array]:
    rng = self.rng
    while True:
      rng1, rng2, rng = jax.random.split(rng, 3)
      means = jax.random.choice(rng1, self.centers, (self.batch_size,))
      normal_samples = jax.random.normal(rng2, (self.batch_size, 2))
      samples = self.scale * means + (self.std ** 2) * normal_samples
      yield samples


def create_gaussian_mixture_samplers(
    name_source: Name_t,
    name_target: Name_t,
    train_batch_size: int = 2048,
    valid_batch_size: int = 2048,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
  """Gaussian samplers.

  Args:
    name_source: name of the source sampler
    name_target: name of the target sampler
    train_batch_size: the training batch size
    valid_batch_size: the validation batch size
    rng: initial PRNG key

  Returns:
    The dataset and dimension of the data.
  """
  rng = utils.default_prng_key(rng)
  rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
  train_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=train_batch_size, rng=rng1)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=train_batch_size, rng=rng2)
      )
  )
  valid_dataset = Dataset(
      source_iter=iter(
          GaussianMixture(name_source, batch_size=valid_batch_size, rng=rng3)
      ),
      target_iter=iter(
          GaussianMixture(name_target, batch_size=valid_batch_size, rng=rng4)
      )
  )
  dim_data = 2
  return train_dataset, valid_dataset, dim_data

class UniformLineDataset:
    def __init__(self, size, mean_left=-1., mean_right=1.,
                 height=2.):#, src_mean, tgt_mean):
        self.size = size
        self.mean_right = mean_right
        self.mean_left = mean_left
        self.height = height
        # self.src_mean = src_mean
        # self.tgt_mean = tgt_mean
        
    def __iter__(self):
        rng = jax.random.PRNGKey(42)
        while True:
            rng, sample_key = jax.random.split(rng, 2)
            yield UniformLineDataset._sample(sample_key, self.size, self.mean_left, self.mean_right, self.height)

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def _sample(key, batch_size, mean_left, mean_right, height):
        k1, k2, key = jax.random.split(key, 3)
        x1 = jax.random.uniform(k1, (batch_size, 1), minval=mean_left-0.25, maxval=mean_left)
        x2 = jax.random.uniform(k2, (batch_size, 1), minval=-height, maxval=height)
        x_0 = jnp.concatenate([x1, x2], axis=1)
        
        k1, k2, key = jax.random.split(key, 3)
        x1 = jax.random.uniform(k1, (batch_size, 1), minval=mean_right, maxval=mean_right+0.25)
        x2 = jax.random.uniform(k2, (batch_size, 1), minval=-height, maxval=height)
        x_1 = jnp.concatenate([x1, x2], axis=1)

        return {
            "src_lin": x_0,
            "tgt_lin": x_1
        }

@dataclasses.dataclass
class Gaussian:
    source_mean: float
    source_var: float

    target_mean: float
    target_var: float

    batch_size: int
    init_key: jax.random.PRNGKey

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
        A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator[jnp.array]:
        key = self.init_key
        while True:
            key1, key2, key = jax.random.split(key, 3)
            source_normal_samples = jax.random.normal(key1, [self.batch_size, 2])
            source_samples = self.source_mean + self.source_var * source_normal_samples

            target_normal_samples = jax.random.normal(key2, [self.batch_size, 2])
            target_samples = self.target_mean + self.target_var * target_normal_samples

            yield {"src_lin": source_samples, 'tgt_lin': target_samples}
            
def create_lagrangian_ds(geometry_str: str, batch_size: int, mean_left, mean_right, height, key=None):
  if geometry_str == "babymaze":
    return UniformLineDataset(size=batch_size)
    # variance = 0.1
    # source_mean = jnp.array([-1.5, 0.5])
    # target_mean = jnp.array([1.5, -0.0])
    
  elif geometry_str == "box":
    return UniformLineDataset(size=batch_size, mean_left=mean_left, mean_right=mean_right, height=height)
  
  elif geometry_str == "drunken_spider":
    return UniformLineDataset(size=batch_size, mean_left=mean_left, mean_right=mean_right, height=height)
  
  elif geometry_str == "vneck":
    return UniformLineDataset(size=batch_size, mean_left=mean_left, mean_right=mean_right, height=height)
    # variance = 0.15
    # source_mean = jnp.array([-2.5, 0.0])
    # target_mean = jnp.array([2.5, 0.0])
  
  elif geometry_str == "slit":
    return UniformLineDataset(size=batch_size, mean_left=mean_left, mean_right=mean_right, height=height)
    # variance = 0.1
    # source_mean = jnp.array([-1.0, 0.0])
    # target_mean = jnp.array([1.0, 0.0])
  
  elif geometry_str == "pipe":
    variance = 0.1
    source_mean = jnp.array([-1.0, 0.0])
    target_mean = jnp.array([1.0, 0.0])
    
  elif geometry_str == "stunnel":
    variance = 0.5
    source_mean = jnp.array([-11.0, -1.0])
    target_mean = jnp.array([11.0, -1.0])
    
  return Gaussian(source_mean=source_mean, source_var=variance,
                  target_mean=target_mean, target_var=variance, batch_size=batch_size, init_key=key)