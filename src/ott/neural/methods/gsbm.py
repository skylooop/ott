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
from flax.struct import PyTreeNode
from ott.neural.methods.lagrangian.lagrangian_potentials import *
__all__ = ["GSBM"]

Callback_t = Callable[[int, ], None]

def my_softplus(x, beta=1, threshold=20):
    # mirroring the pytorch implementation https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = jax.lax.select(x * beta < threshold, x, jax.numpy.ones_like(x))
    return jax.lax.select(x * beta < threshold, 1/beta * jax.numpy.log(1 + jax.numpy.exp(beta * x_safe)), x)

def linear_interp1d(t: Float[ArrayLike, "discr_means_t B"], xt: Float[ArrayLike, "discr_means_t B D"], query_t: Float[ArrayLike, "evalua_t B"]):
    #t: Float[ArrayLike, "discr_means_t B"] = jnp.repeat(jnp.linspace(0, 1, xt.shape[0]), repeats=xt.shape[1], axis=-1)
    velocities = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10)[..., None]
    left = jax.vmap(partial(jnp.searchsorted, side='left'), in_axes=(0, 0))(t[1:].T, query_t.T).T
    mask_l = jax.nn.one_hot(left, xt.shape[0]).transpose(0, 2, 1).reshape(query_t.shape[0], xt.shape[0], xt.shape[1], 1)
    
    T, N, D = xt.shape
    S = query_t.shape[0]
    t = t.reshape(1, T, N, 1)
    xt = xt.reshape(1, T, N, D)
    m = jnp.ones_like(t, dtype=bool)
    s = query_t.reshape(S, N, 1)
    
    x0 = jnp.sum(t * mask_l, axis=1)
    p0 = jnp.sum(xt * mask_l, axis=1)
    m0 = jnp.sum(velocities * mask_l[:, :-1], axis=1)
    
    t = s - x0
    return t * m0 + p0

class GaussianSpline(nn.Module):
    networks: Dict[str, nn.Module]
    
    def mean(self, time, **kwargs):
        return self.networks['spline_means'](time, **kwargs)
    
    def gamma(self, time, **kwargs):
        return self.networks['spline_gammas'](time, **kwargs)
    
    def get_xt(self) -> Float[ArrayLike, "T B D"]:
        return jnp.concatenate([self.networks['spline_means'].x0, self.networks['spline_means'].variables['params']['knots'],
                                self.networks['spline_means'].x1], axis=0)
    
    def get_gamma(self, **kwargs):
        return self.networks['spline_gammas'](self.networks['spline_gammas'].spline_discr[:, 0], **kwargs)
    
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
    #xt: Float[ArrayLike, "T B D"] = None
    
    @nn.compact
    def __call__(self, query_t: Float[ArrayLike, "evalua_t"]):
        knots = self.param(
            'knots', lambda rng, shape: self.init, self.shape
        )
        query_t: Float[ArrayLike, "evalua_t B"] = jnp.repeat(query_t[:, None], repeats=knots.shape[1], axis=-1)
        xt = jnp.concatenate([self.x0, knots, self.x1], axis=0) # (T, B, D)
        yt = linear_interp1d(self.spline_discr, xt, query_t)
        return yt.transpose(1, 0, 2)
        
class StdSpline(nn.Module):
    sigma: float = 1.0
    shape: tuple = None
    init: ArrayLike = None
    x0: Float[ArrayLike, "1 B D"] = None
    x1: Float[ArrayLike, "1 B D"] = None
    spline_discr: Float[ArrayLike, "discr_means_s B"] = None
    
    @nn.compact
    def __call__(self, query_t: Float[ArrayLike, "evalua_t"]):
        knots = self.param(
            'knots', lambda rng, shape: self.init, self.shape
        )
        base = self.sigma * jnp.sqrt(query_t * (1 - query_t))
        query_t: Float[ArrayLike, "evalua_t B"] = jnp.repeat(query_t[:, None], repeats=knots.shape[1], axis=-1)
        xt = jnp.concatenate([self.x0, knots, self.x1], axis=0) # (T, B, D)
        xt = linear_interp1d(self.spline_discr, xt, query_t).transpose(1, 0, 2)
        return base.reshape(1, -1, 1) * my_softplus(xt)
        
def map_nested_fn(fn):
  '''Recursively apply `fn` to key-value pairs of a nested dict.'''
  def map_fn(nested_dict):
      #temp = {k: fn(k, v) for k, v in nested_dict['params'].items()}
      return {"params": {k: fn(k, v) for k, v in nested_dict['params'].items()}}
    # return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
    #         for k, v in nested_dict.items()}
  return map_fn
        
class GSBM(PyTreeNode):
    T_mean: int # number of knots for mean spline
    T_gamma: int # number of knots for gamma spline 
    sigma: float
    lr_mean: float
    lr_gamma: float
    potential: LagrangianPotentialBase
    state: train_state.TrainState = None

    @classmethod
    def create(cls, 
            T_mean=15, # number of knots for mean spline
            T_gamma=30, # number of knots for gamma spline 
            sigma: float = 0.9,
            lr_mean: float = 3e-4,
            lr_gamma: float = 3e-4,
            potential=None):
        return cls(
            T_mean=T_mean, T_gamma=T_gamma, sigma=sigma, lr_mean=lr_mean, lr_gamma=lr_gamma, potential=potential
        )
     
    def _initialize(self, x0: Float[ArrayLike, "B D"], x1: Float[ArrayLike, "B D"], rng: Key):
        # x0 = next(loader)['src_lin']
        # x1 = next(loader)['tgt_lin']
        # gsbm = self._initialize(x0, x1, loop_key)
        
        mean_discrit = jnp.linspace(0, 1, self.T_mean) # discritization for mean spline
        gamma_discrit = jnp.linspace(0, 1, self.T_gamma)# discritization for gamma spline
        mean_x_t = (1 - mean_discrit[None, :, None]) * x0[:, None] + mean_discrit[None, :, None] * x1[:, None] # (B, T, D)
        gamma_t = jnp.zeros((x0.shape[0], self.T_gamma, 1)) # (B, T_gamma, 1)
        
        label_fn = map_nested_fn(lambda k, _: "means" if k == "networks_spline_means" else "gammas")
        means_scheduler = optax.linear_schedule(init_value=self.lr_mean, end_value=9e-1, transition_steps=10_000, transition_begin=4_000)
        gammas_scheduler = optax.linear_schedule(init_value=self.lr_gamma, end_value=9e-1, transition_steps=10_000,transition_begin=4_000 )
        optimizer = optax.multi_transform({"means": optax.adam(learning_rate=means_scheduler),
                                           "gammas": optax.adam(learning_rate=gammas_scheduler)}, label_fn)#optax.adam(learning_rate=3e-4)
        mean_x_t: Float[ArrayLike, "T B D"] = mean_x_t.transpose(1, 0, 2)
        gamma_t: Float[ArrayLike, "T B 1"] = gamma_t.transpose(1, 0, 2)
        spline_means = EndPointSpline(init=mean_x_t[1:-1, ...], shape=mean_x_t[1:-1, ...].shape, x0=mean_x_t[0][None], x1=mean_x_t[-1][None],
                                      spline_discr=jnp.repeat(mean_discrit[:, None], x0.shape[0], -1))#, xt=mean_x_t)
        spline_gammas = StdSpline(init=gamma_t[1:-1, ...], shape=gamma_t[1:-1, ...].shape, x0=gamma_t[0][None], x1=gamma_t[-1][None],
                                      spline_discr=jnp.repeat(gamma_discrit[:, None], x0.shape[0], -1))#, xt=gamma_t)
        
        network_def = GaussianSpline(networks={"spline_means": spline_means,
                                           "spline_gammas": spline_gammas})
        params = network_def.init(rng, jnp.linspace(0, 1, 100))
        state = train_state.TrainState.create(
            params=params,
            apply_fn=network_def.apply,
            tx=optimizer
        )
        return self.replace(state = state)
    
    @jax.jit
    def update_step(self, gsbm, query_t, rng):
        def loss_fn(params):
            xt = self.sample_xt(gsbm, query_t, rng, params=params)
            ut = self.sample_ut(gsbm, query_t, xt, params=params)
            potential_loss = jax.vmap(jax.vmap(gsbm.potential))(xt)
            scale = (0.5 / (self.sigma ** 2))
            cost_c = scale * (ut ** 2).sum(axis=-1)
            return (cost_c + potential_loss).mean()
        
        loss, grads = jax.value_and_grad(loss_fn)(gsbm.state.params)
        updated_state = gsbm.state.apply_gradients(grads=grads)
        return self.replace(state=updated_state), loss
    
    def sample_ut(self, gsbm, query_t: Float[ArrayLike, "evalua_t"], xt: Float[ArrayLike, "B evalua_t D"], params, **kwargs):
        mean, dmean = jax.jvp(
            partial(gsbm.state.apply_fn, method='mean', **kwargs),
            (params, query_t),
            (jax.tree_map(jnp.zeros_like, params), jnp.ones_like(query_t))
        )
        std, dstd = jax.jvp(
            partial(gsbm.state.apply_fn, method='gamma', **kwargs),
            (params, query_t),
            (jax.tree_map(jnp.zeros_like, params), jnp.ones_like(query_t))
        )
        a = (dstd - self.sigma ** 2 / (2 * std)) / std
        drift = dmean + a * (xt -  mean)
        return drift
        
    
    def sample_xt(self, gsbm, query_t: Float[ArrayLike, "evalua_t"], rng, params=None, **kwargs) -> Float[ArrayLike, "B points evlua_t D"]:
        # sample from gaussian
        mean_t: Float[ArrayLike, "B evalua_t D"] = gsbm.state.apply_fn(params, query_t, method='mean', **kwargs)
        std_t = gsbm.state.apply_fn(params, query_t, method='gamma', **kwargs)
        noise = jax.random.normal(rng, shape=mean_t.shape)
        xt = mean_t + noise * std_t
        return xt
    
    def get_final_xt(self):
        return self.state.apply_fn(self.state.params, method='get_xt')
    
    def get_final_gammas(self):
        return self.state.apply_fn(self.state.params, method='get_gamma')
    
    def __call__(  # noqa: D102
      self,
      #loader: Iterable[Dict[str, np.ndarray]],
      gsbm,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
      eps: float = 0.001,
      timesteps_interpolate: int = 100
  ) -> Dict[str, List[float]]:
            
        loop_key = utils.default_prng_key(rng)
        it = 0
        pbar = tqdm(range(n_iters + 1), total=n_iters, colour='green', dynamic_ncols=True)
        
        for _ in pbar:
            it_key = jax.random.fold_in(loop_key, it)
            
            # GSBM related
            evaluation_timesteps = jnp.linspace(eps, 1-eps, timesteps_interpolate)
            gsbm, loss = self.update_step(gsbm, evaluation_timesteps, it_key)
            # ...
            
            if it % 1_000 == 0 and it > 0:# and callback is not None:
                #callback(it, training_logs, self.transport)
                pbar.set_postfix({'loss': loss})
            
            it += 1
            if it >= n_iters:
                break

        return gsbm