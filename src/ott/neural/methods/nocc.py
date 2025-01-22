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

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

class TimedX(NamedTuple):
  t: Any
  x: Any

class NeuralOC:
  
  def __init__(
      self,
      input_dim: int,
      value_model: nn.Module,
      optimizer: Optional[optax.GradientTransformation],
      correction_model: nn.Module,
      correction_optimizer: Optional[optax.GradientTransformation],
      flow: dynamics.LagrangianFlow,
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
      key:Optional[jax.Array] = None,
      **kwargs: Any,
  ):
    self.value_model = value_model
    self.flow = flow
    self.time_sampler = time_sampler
    self.potential_weight = potential_weight
    self.control_weight = control_weight
    self.reg_weight = reg_weight
   
    key, init_key = jax.random.split(key, 2)
    params = value_model.init(
      init_key, 
      jnp.ones([1, 1]), 
      jnp.ones([1, input_dim]), 
      jnp.ones([1, input_dim])
    )
  
    self.state = train_state.TrainState.create(
      apply_fn=value_model.apply,
      params=params,
      tx=optimizer
    )

    key, init_key = jax.random.split(key, 2)
    corr_params = correction_model.init(
      init_key, 
      jnp.ones([1, 1]), 
      jnp.ones([1, input_dim]),
      jnp.ones([1, input_dim])
    )
  
    self.corr_state = train_state.TrainState.create(
      apply_fn=correction_model.apply,
      params=corr_params,
      tx=correction_optimizer
    )

    self.buffer = np.empty([1000_000, input_dim])
    self.buffer_size = 0 

    self.train_step_cost, self.train_step_with_potential = self._get_step_fn()

  def _get_step_fn(self) -> Callable:
      
      # def expectile_loss(diff: jnp.ndarray, expectile=0.98) -> jnp.ndarray:
      #     weight = jnp.where(diff >= 0, expectile, (1 - expectile))
      #     return weight * diff ** 2

      def corr_loss(corr_state, corr_params, key_t, state, source, target):
        bs = source.shape[0]
        t = self.time_sampler(key_t, bs)
        x_0, x_1 = source, target
        corr_fn = lambda t, x, x_0_: corr_state.apply_fn(corr_params, t, x, x_0_)
        x_t = self.flow.compute_xt(key_t, t, x_0, x_1, corr_fn)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()
        U_t = self.flow.compute_potential(t, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])
        dsdt, dsdx = dsdtdx_fn(state.params, t, x_t, x_0)

        loss = dsdt
        # loss = dsdt + 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1)

        return (loss).mean()

      def am_loss(state, params, key_t, corr_state, source, target):
        bs = source.shape[0]
        t = self.time_sampler(key_t, bs)
        # t, u0 = sample_t(u0, bs)
        x_0, x_1 = source, target
        # x_t = x_0 * (1 - t) + x_1 * t
        corr_fn = lambda t, x, x_0_: corr_state.apply_fn(corr_state.params, t, x, x_0_)
        x_t = self.flow.compute_xt(key_t, t, x_0, x_1, corr_fn)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()
        U_t = self.flow.compute_potential(t, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])
        dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_0)

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            fun = lambda __x: state.apply_fn(p,t,__x,x0).sum()
            return jnp.trace(jax.jacfwd(jax.jacrev(fun))(x))

        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        s_diff = dsdt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + D * laplacian(params, t, x_t, x_0).reshape(-1, 1)
        loss = jnp.abs(s_diff ** 2).mean()
        loss += (- dsdt + 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True)).mean() * self.reg_weight

        return loss * self.control_weight
      
      def am_loss_sample(state, params, key_t, sample):
        
        x_t = sample
        bs = sample.shape[0]
        t = self.time_sampler(key_t, bs)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()
        U_t = self.flow.compute_potential(t, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])
        dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_t)

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            fun = lambda __x: state.apply_fn(p,t,__x,x0).sum()
            return jnp.trace(jax.jacfwd(jax.jacrev(fun))(x))

        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        s_diff = dsdt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + D * laplacian(params, t, x_t, x_t).reshape(-1, 1)
        loss = jnp.abs(s_diff ** 2).mean()
        loss += (- dsdt + 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True)).mean() * self.reg_weight

        return loss * self.control_weight

      def potential_loss(state, params, key, steps_count, weight, source, target):
        bs = source.shape[0]
        t_0, t_1 = jnp.zeros([bs, 1]), jnp.ones([bs, 1])
        x_0, x_1 = source, target
        dt = 1.0 / steps_count

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        def move(carry, _):
          t_, x_, key_ = carry
          _, dsdx = dsdtdx_fn(state.params, t_, x_, x_0)
          At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
          sigma = self.flow.compute_sigma_t(t_)
          key_, key_s = jax.random.split(key_)
          x_ = x_ - dt * dsdx @ At_T + sigma * jax.random.normal(key_s, shape=x_.shape) * dt
          t_ = t_ + dt
          return (t_, x_, key_), x_
        
        _, result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(result[-1])

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0)).mean()
        reg_loss = 0

        # exp. reg

        # t = jax.random.randint(key, (1,), 0, steps_count-1).reshape(1)
        # x_t = result[t].reshape(*x_0.shape)
        # t = (t + jnp.zeros([bs, 1])).reshape(bs, 1)
        # dsdt, dsdx = dsdtdx_fn(state.params, t, x_t, x_0)
        # reg_loss = 0.5 * (dsdx * dsdx).sum(-1, keepdims=True).mean() - dsdt.mean()

        return (reg_loss + dual_loss)  * weight, result

      @jax.jit
      def train_step_cost(state, key, source, target, sample):
        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, sample)
        state = state.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 20, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)
        
        return state, loss, loss_potential, x_seq


      @jax.jit
      def train_step_with_potential(state, corr_state, key, source, target):
        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, corr_state, source, target)
        state = state.apply_gradients(grads=grads)
        
        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 20, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)

        grad_fn = jax.value_and_grad(corr_loss, argnums=1, has_aux=False)
        loss_corr, corr_grads = grad_fn(corr_state, corr_state.params, key, state, source, target)
        corr_state = corr_state.apply_gradients(grads=corr_grads)
        
        return state, corr_state, loss, loss_potential, x_seq
      
      
      return train_step_cost, train_step_with_potential
  

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
  ) -> Dict[str, List[float]]:
    
    loop_key = utils.default_prng_key(rng)
    training_logs = {"cost_loss": [], "potential_loss": []}
    it = 0
    pbar = tqdm(loader, total=n_iters, colour='green', dynamic_ncols=True)
    for batch in pbar:
      # batch = jtu.tree_map(jnp.asarray, batch)
    
      src, tgt = batch["src_lin"], batch["tgt_lin"]
      # src_cond = batch.get("src_condition")
      it_key = jax.random.fold_in(loop_key, it)

      if it > 10_000 and it % 2 == 0:
          x_sample = self.buffer[np.random.randint(0, self.buffer_size, src.shape[0])]
          self.state, loss, loss_potential, x_seq = self.train_step_cost(self.state, it_key, src, tgt, x_sample)
      else:
          self.state, self.corr_state, loss, loss_potential, x_seq = self.train_step_with_potential(self.state, self.corr_state, it_key, src, tgt)
      
      training_logs["potential_loss"].append(loss_potential)
      training_logs["cost_loss"].append(loss)

      x_seq = x_seq.reshape(-1, x_seq.shape[-1])
      self.buffer = np.roll(self.buffer, x_seq.shape[0], axis=0)
      self.buffer[:x_seq.shape[0]] = np.asarray(x_seq)
      self.buffer_size = min(self.buffer_size + x_seq.shape[0], 1000_000)

      
      if it % 10_000 == 0 and it > 0 and callback is not None:
        callback(it, training_logs, self.transport)
        pbar.set_postfix({"pot_loss": loss_potential,
                          "cost_loss": loss})
      it += 1
      if it >= n_iters:
        break

    return training_logs

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    
    dt = 1.0 / 20
    t_0 = 0.0
    n = 20
    loop_key = jax.random.PRNGKey(0)
  
    @jax.jit
    def inference(state, x_0):

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
      
      def move(carry, _):
        t_, x_, cost, key_ = carry
        u = dsdx_fn(state.params, t_ * jnp.ones([x.shape[0],1]), x_, x_0)
        At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
        U_t = self.flow.compute_potential(t_, x_)
        sigma = self.flow.compute_sigma_t(t_)
        key_, key_s = jax.random.split(key_)
        x_ = x_ - dt * u @ At_T + sigma * jax.random.normal(key_s, shape=x_.shape) * dt
        t_ = t_ + dt
        cost += 0.5 * ((u @ At_T) * u).sum(-1).mean() * dt + U_t.mean() * dt * self.potential_weight
        return (t_, x_, cost, key_), x_
          
      (_, _, cost, _), result = jax.lax.scan(move, (t_0, x_0, 0.0, loop_key), None, length=n)
      return cost, result
    
    cost, result = inference(self.state, x)
    x_seq = [TimedX(t=t_0, x=x)]

    for i in range(n):
      t_ = x_seq[-1].t + dt
      x_seq.append(TimedX(t=t_, x=result[i]))
      
    return cost, x_seq