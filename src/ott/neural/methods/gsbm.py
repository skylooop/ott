from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# import diffrax
from functools import partial
from flax.training import train_state
from flax import linen as nn
import optax
from ott import utils
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils
from flax.training import train_state
from collections import defaultdict
from jaxtyping import ArrayLike, Float, Key

__all__ = ["GSBM"]

Callback_t = Callable[[int, ], None]

def linear_interp1d(t: Float[ArrayLike, "discr_means_t B"], xt: Float[ArrayLike, "discr_means_t B D"], query_t):
    #t: Float[ArrayLike, "discr_means_t B"] = jnp.repeat(jnp.linspace(0, 1, xt.shape[0]), repeats=xt.shape[1], axis=-1)
    velocities = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10)[..., None]
    left = jnp.searchsorted(t[1:].T.ravel(), query_t.T.ravel(), side='left').reshape(t.shape)
    mask_l = jax.nn.one_hot(left, xt.shape[0]).permute(0, 2, 1).reshape(query_t.shape[0], xt.shape[0], xt.shape[1], 1)
    pass

class GaussianSpline(nn.Module):
    networks: Dict[str, nn.Module]
    
    def mean(self, time, **kwargs):
        return self.networks['spline_means'](time, **kwargs)
    
    def gamma(self, time, **kwargs):
        return self.networks['spline_gammas'](time, **kwargs)
    
    def __call__(self, t):
        inits = {
            "spline_mean": self.mean(t),
            "spline_gamma": self.gamma(t)
        }
        return inits

class EndPointSpline(nn.Module):
    shape: tuple = None
    init: ArrayLike = None
    x0: Float[ArrayLike, "1 B D"] = None
    x1: Float[ArrayLike, "1 B D"] = None
    spline_discr: Float[ArrayLike, "discr_means_t B"] = None
    
    @nn.compact
    def __call__(self, query_t):
        knots = self.param(
            'knots', lambda rng, shape: self.init, self.shape
        )
        query_t: Float[ArrayLike, "evalua_t B"] = jnp.repeat(query_t[:, None], repeats=knots.shape[0], axis=-1)
        xt = jnp.concatenate([self.x0, knots.transpose(1, 0, 2), self.x1], axis=0) # (T, B, D)
        yt = linear_interp1d(self.spline_discr, xt, query_t)
        return yt.transpose(1, 0, 2)
        
class StdSpline(nn.Module):
    sigma: float = 1.0
    
    @nn.compact
    def __call__(self, query_t):
        base = self.sigma * jnp.sqrt(query_t * (1 - query_t))
        xt = None
        pass
        

def cost_fn(potential):
    def loss_fn(xt):
        cost_potential = potential(xt)
        return cost_potential
    return loss_fn
                

class GSBM:
    def __init__(
        self,
        T_mean=15, # number of knots for mean spline
        T_gamma=30, # number of knots for gamma spline 
        sigma: float = 1.0,
        lr_mean: float = 0.2,
        lr_gamma: float = 0.1,
        potential = None
    ):
        self.T_mean = T_mean
        self.T_gamma = T_gamma
        self.sigma = sigma
        self.lr_mean = lr_mean
        self.lr_gamma = lr_gamma
        self.cost_fn = cost_fn(potential)
        
    def _initialize(self, x0: Float[ArrayLike, "B D"], x1: Float[ArrayLike, "B D"], rng: Key):
        mean_discrit = jnp.linspace(0, 1, self.T_mean) # discritization for mean spline
        gamma_discrit = jnp.linspace(0, 1, self.T_gamma)# discritization for gamma spline
        mean_x_t = (1 - mean_discrit[None, :, None]) * x0[:, None] + mean_discrit[None, :, None] * x1[:, None] # (B, T, D)
        gamma_t = jnp.zeros((x0.shape[0], self.T_gamma, 1)) # (B, S, 1)
        
        optimizer = optax.adam(learning_rate=3e-4) # TODO: Add different lrs for different parts
        spline_means = EndPointSpline(init=mean_x_t[:, 1:-1, :], shape=mean_x_t[:, 1:-1, :].shape, x0=x0[None], x1=x1[None],
                                      spline_discr=jnp.repeat(mean_discrit[:, None], x0.shape[0], -1))
        spline_gammas = StdSpline()
        network_def = GaussianSpline(networks={"spline_means": spline_means,
                                           "spline_gammas": spline_gammas})
        params = network_def.init(rng, mean_discrit)['params']
        self.state = train_state.TrainState.create(
            params=params,
            apply_fn=network_def.apply,
            tx=optimizer
        )
    
    @jax.jit
    def update_step(self, query_t):
        xt = self.sample_xt(query_t)
        ut = self.sample_ut(query_t, xt)
        
        def loss_fn(params):
            pass
        
        return self.state.replace(params=params)
    
    def sample_xt(self, query_t: Float[ArrayLike, "T"], N: int, **kwargs) -> Float[ArrayLike, "B points T dim"]:
        mean_t = self.state(t, method='mean', **kwargs)
        
    
    def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
      eps: float = 0.001,
      timesteps_interpolate: int = 100
  ) -> Dict[str, List[float]]:
            
        loop_key = utils.default_prng_key(rng)
        it = 0
        pbar = tqdm(loader, total=n_iters, colour='green', dynamic_ncols=True)
        x0 = next(loader)['src_lin']
        x1 = next(loader)['tgt_lin']
        training_logs = defaultdict()
        
        self._initialize(x0, x1, loop_key)
        
        for batch in pbar:
            src, tgt = batch["src_lin"], batch["tgt_lin"]
            it_key = jax.random.fold_in(loop_key, it)
            
            # GSBM related
            evaluation_timesteps = jnp.linspace(eps, 1-eps, timesteps_interpolate)
            self.state, logs = None
            # ...
            
            if it % 10_000 == 0 and it > 0 and callback is not None:
                callback(it, training_logs, self.transport)
                pbar.set_postfix({})
            
            it += 1
            if it >= n_iters:
                break

        return training_logs