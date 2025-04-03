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
from flax import struct

# import diffrax
from functools import partial
from flax.training import train_state
from flax import linen as nn
import optax
from ott2 import utils
from ott2.neural.methods.flows import dynamics
from ott2.solvers import utils as solver_utils
from flax.training import train_state

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

class TimedX(NamedTuple):
  t: Any
  x: Any


class Trajectory(struct.PyTreeNode):
    t: jnp.ndarray
    x: jnp.ndarray
    u: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray


class NeuralOC:
  
  def __init__(
      self,
      input_dim: int,
      value_model: nn.Module,
      optimizer: Optional[optax.GradientTransformation],
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

    self.state_target = train_state.TrainState.create(
      apply_fn=value_model.apply,
      params= jax.tree.map(lambda x: jnp.copy(x), params),
      tx=optax.identity()
    )

    self.buffer = np.empty([1000_000, input_dim])
    self.time_buffer = np.empty([1000_000])
    self.buffer_size = 0 

    self.train_step_cost, self.train_step_with_potential = self._get_step_fn()

  def _get_step_fn(self) -> Callable:
      

      def am_loss(state, params, key_t, source, target, state_target):
        bs = source.shape[0]
        t = self.time_sampler(key_t, bs)
        x_0, x_1 = source, target
        x_t = self.flow.compute_xt(key_t, t, x_0, x_1)
        
        return am_loss_sample(state, params, key_t, x_t, t, source, state_target)

      def am_loss_sample(state, params, key_t, sample, t_sample, source, state_target):
        x_t = sample
        t_sample = t_sample.reshape(-1, 1)
        t_0 = t_sample
        At_T = self.flow.compute_inverse_control_matrix(t_0, x_t).transpose()
        U_t = self.flow.compute_potential(t_0, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])
        # dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

        dsdt, dsdx = dsdtdx_fn(params, t_0, x_t, x_t)
        u1 = jax.lax.stop_gradient(dsdx)

        sigma = self.flow.compute_sigma_t(t_0)
        gamma = 0.99
        dt = 1.0 / 30

        t_1 = jnp.minimum(t_0 + dt, 1.0)
        x_1 = x_t - (t_1 - t_0) * u1 @ At_T + sigma * jax.random.normal(key_t, shape=x_t.shape) * (t_1 - t_0) 
        key_t, key = jax.random.split(key_t, 2)

        _, u2 = dsdtdx_fn(state.params, t_1, x_1, x_1)
        t_2 = jnp.minimum(t_1 + dt, 1.0)
        x_2 = x_1 - (t_2 - t_1) * u2 @ At_T + sigma * jax.random.normal(key_t, shape=x_t.shape) * (t_2 - t_1) 

        s_0 = state.apply_fn(params, t_0, x_t, x_t)
        s_1 = state.apply_fn(state_target.params, t_1, x_1, x_1)
        s_2 = state.apply_fn(state_target.params, t_2, x_2, x_2)

        U_1 = self.flow.compute_potential(t_1, x_1)
        r_1 = 0.5 * ((u1 @ At_T) * u1).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1)
        r_2 = 0.5 * ((u2 @ At_T) * u2).sum(-1, keepdims=True) + self.potential_weight * U_1.reshape(-1, 1)
        
        R_2 = r_2 * (t_2 - t_1) + gamma * s_2  
        R_1 = r_1 * (t_1 - t_0) + gamma * (0.5 * s_1 + 0.5 * R_2)

        s_diff =(R_1 - s_0)

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
          x_next = x_ - dt * dsdx @ At_T + sigma * jax.random.normal(key_s, shape=x_.shape) * dt
          t_next = t_ + dt
          return (t_next, x_next, key_), TimedX(t_, x_)
        
        (t_last, x_last, key_last), result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(x_last)

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0)).mean()
        reg_loss = 0

        return (reg_loss + dual_loss) * weight, result

      @jax.jit
      def train_step_cost(state, key, source, target, x_sample, t_sample, state_target):
        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, x_sample, t_sample, source, state_target)
        state = state.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 20, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)

        new_target_params = optax.incremental_update(state.params, state_target.params, 0.01)
        state_target = state_target.replace(params=new_target_params)
        
        return state, loss, loss_potential, x_seq, state_target


      @jax.jit
      def train_step_with_potential(state, key, source, target, state_target):
        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, source, target, state_target)
        state = state.apply_gradients(grads=grads)
        
        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 20, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)

        new_target_params = optax.incremental_update(state.params, state_target.params, 0.01)
        state_target = state_target.replace(params=new_target_params)
        
        return state, loss, loss_potential, x_seq, state_target
      
      
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

      if it > 10_000 and it % 4 != 0:
          indices = np.random.randint(0, self.buffer_size, src.shape[0])
          x_sample = self.buffer[indices]
          t_sample = self.time_buffer[indices]
          self.state, loss, loss_potential, tx_seq, self.state_target = self.train_step_cost(self.state, it_key, src, tgt, x_sample, t_sample, self.state_target)
      else:
          self.state, loss, loss_potential, tx_seq, self.state_target = self.train_step_with_potential(self.state, it_key, src, tgt, self.state_target)
      
      training_logs["potential_loss"].append(loss_potential)
      training_logs["cost_loss"].append(loss)

      x_seq = tx_seq.x.reshape(-1, tx_seq.x.shape[-1])
      t_seq = tx_seq.t.reshape(-1)
      self.buffer = np.roll(self.buffer, x_seq.shape[0], axis=0)
      self.buffer[:x_seq.shape[0]] = np.asarray(x_seq)
      self.buffer_size = min(self.buffer_size + x_seq.shape[0], 1000_000)

      self.time_buffer = np.roll(self.time_buffer, x_seq.shape[0], axis=0)
      self.time_buffer[:x_seq.shape[0]] = np.asarray(t_seq)
      
      if it % 5_000 == 0 and it > 0 and callback is not None:
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