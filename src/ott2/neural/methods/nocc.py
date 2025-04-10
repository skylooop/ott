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
from flax import struct
import optax
from ott2 import utils
from ott2.neural.methods.flows import dynamics
from ott2.solvers import utils as solver_utils
from flax.training import train_state

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

class TimedX(struct.PyTreeNode):
  t: jnp.ndarray
  x: jnp.ndarray

class NeuralOC:
  
  def __init__(
      self,
      input_dim: int,
      value_model: nn.Module,
      optimizer: Optional[optax.GradientTransformation],
      flow: dynamics.LagrangianFlow,
      control_steps: int,
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      acc_weight: float,
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
    self.acc_weight = acc_weight
    self.control_steps = control_steps
   
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

    self.target_state = train_state.TrainState.create(
      apply_fn=value_model.apply,
      params=jax.tree.map(lambda x: jnp.copy(x), params),
      tx=optax.identity()
    )

    self.x_buffer = np.empty([1000_000, input_dim])
    self.t_buffer = np.empty([1000_000])
    self.buffer_size = 0 

    self.train_step_cost, self.train_step_with_potential = self._get_step_fn()

  def _get_step_fn(self) -> Callable:
      
      def am_loss(state, params, key_t, source, target, target_state):
        bs = source.shape[0]
        t = self.time_sampler(key_t, bs)
        x_0, x_1 = source, target
        x_t = self.flow.compute_xt(key_t, t, x_0, x_1)
        
        return am_loss_sample(state, params, key_t, t, x_t, target_state)
      
      def am_loss_sample(state, params, key_t, t_sample, x_sample, target_state):
        
        x_t = x_sample
        t = t_sample.reshape(-1, 1)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()
        U_t = self.flow.compute_potential(t, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])
        # dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
        
        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_t)
        dsdt_tgt, dsdx_tgt = dsdtdx_fn(target_state.params, t, x_t, x_t)
        u = dsdx_tgt
        vt = dsdt_tgt

        def normalize(x):
          norm = jnp.linalg.norm(x) + 1e-5
          return x / norm

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def acceleration(p, t, x, x0):
            fun = lambda __t, __x: state.apply_fn(p,__t,__x,x0).sum()
            norm_rev = lambda __t, __x: normalize(jax.jacrev(fun, 1)(__t, __x))
            return jax.jacfwd(norm_rev, argnums=0)(t, x).squeeze()
        
        a = acceleration(params, t, x_t, x_t)
        a_tgt = acceleration(target_state.params, t, x_t, x_t)

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            fun = lambda __x: state.apply_fn(p,t,__x,x0).sum()
            return jnp.trace(jax.jacfwd(jax.jacrev(fun))(x))
        
        # vddx_target = laplacian(target_state.params, t, x_t, x_t).reshape(-1, 1)
        vddx = laplacian(params, t, x_t, x_t).reshape(-1, 1)
        a_cost_tgt = jnp.sqrt((a_tgt * a_tgt).sum(-1, keepdims=True)) * self.acc_weight
        # a_cost = jnp.sqrt((a * a).sum(-1, keepdims=True)) * self.acc_weight
        potential_cost = self.potential_weight * U_t.reshape(-1, 1)

        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        s_diff_1 = dsdt - 0.5 * ((u @ At_T) * u).sum(-1, keepdims=True) + a_cost_tgt + potential_cost + D * vddx
        s_diff_2 = vt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + a_cost_tgt + potential_cost + D * vddx
        loss = jnp.abs(s_diff_1 ** 2).mean() + jnp.abs(s_diff_2 ** 2).mean() 

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
        
        (_, x_last ,_), result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(x_last)

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0)).mean()
        reg_loss = 0

        return (reg_loss + dual_loss)  * weight, result

      @jax.jit
      def train_step_cost(state, key, source, target, t_sample, x_sample, target_state):
        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state)
        state = state.apply_gradients(grads=grads)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, self.control_steps, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)
        
        return state, loss, loss_potential, x_seq, target_state


      @jax.jit
      def train_step_with_potential(state, key, source, target, target_state):
        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, source, target, target_state)
        state = state.apply_gradients(grads=grads)
        
        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, self.control_steps, 1.0, source, target)
        state = state.apply_gradients(grads=potential_grads)

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)
        
        return state, loss, loss_potential, x_seq, target_state
      
      
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
          ids = np.random.randint(0, self.buffer_size, src.shape[0])
          t_sample = self.t_buffer[ids]
          x_sample = self.x_buffer[ids]
          self.state, loss, loss_potential, tx_seq, self.target_state = self.train_step_cost(self.state, it_key, src, tgt, t_sample, x_sample, self.target_state)
      else:
          self.state, loss, loss_potential, tx_seq, self.target_state = self.train_step_with_potential(self.state, it_key, src, tgt, self.target_state)
      
      training_logs["potential_loss"].append(loss_potential)
      training_logs["cost_loss"].append(loss)

      x_seq = tx_seq.x.reshape(-1, tx_seq.x.shape[-1])
      t_seq = tx_seq.t.reshape(-1)
      self.x_buffer = np.roll(self.x_buffer, x_seq.shape[0], axis=0)
      self.x_buffer[:x_seq.shape[0]] = np.asarray(x_seq)
      self.t_buffer = np.roll(self.t_buffer, x_seq.shape[0], axis=0)
      self.t_buffer[:x_seq.shape[0]] = np.asarray(t_seq)
      self.buffer_size = min(self.buffer_size + x_seq.shape[0], 100_000)

      
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
    
    dt = 1.0 / self.control_steps
    t_0 = 0.0
    n = self.control_steps
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