import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
NPROC = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

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
# import diffrax
import pickle
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import flashbax as fbx
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state

from ott2 import utils
from ott2.neural.methods.flows import dynamics
from ott2.solvers import utils as solver_utils

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

from jax.experimental import mesh_utils

# multigpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec

P = PartitionSpec
mesh = Mesh(mesh_utils.create_device_mesh((NPROC,)), axis_names=('data',))

def with_mesh(f):
    def wrapper(*args, **kwargs):
        with mesh:
            return f(*args, **kwargs)
    return wrapper

# buffer
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        dim: int,
        batch_size: int,
        seed: int=0
    ):
        buffer = fbx.make_item_buffer(
            min_length=1,
            max_length=capacity,
            sample_batch_size=batch_size,
            add_batches=True,
        )
        buffer = buffer.replace(
            init = jax.jit(buffer.init),
            add = jax.jit(buffer.add, donate_argnums=0),
            sample = jax.jit(buffer.sample),
            can_sample = jax.jit(buffer.can_sample),
        )

        init_sample = {"x": np.random.randn(dim), "t": np.random.randn()} 
        state = buffer.init(init_sample)

        self.buffer = buffer
        self.state = state
        self.rng = jax.random.key(seed)
    
    def append(self, x: np.ndarray, t: np.ndarray):
        self.state = self.buffer.add(
            self.state,
            {"x": x, "t": t}
        )
    
    def sample(self):
        self.rng, key = jax.random.split(self.rng)
        return self.buffer.sample(self.state, key).experience


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
      batch_size: int,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
      key:Optional[jax.Array] = None,
      load_dir: str = None,
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

    with mesh:
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
        tx=optax.identity(),
      )
      if not load_dir:
        pass
      elif not os.path.exists(load_dir):
        print(f"Path does not exist: {load_dir}")
      else:
        with open(f"{load_dir}/opt_state_latest.pkl", "rb") as file:
          opt_state = pickle.load(file)
        with open(f"{load_dir}/params_latest.pkl", "rb") as file:
          params = pickle.load(file)
        with open(f"{load_dir}/step_latest.pkl", "rb") as file:
          step = pickle.load(file)
        self.state = self.state.replace(params=params, opt_state=opt_state, step=step)
        self.target_state = self.target_state.replace(params=params, step=step)
        print("opt_state + params + step: loaded")


    self.buffer = TrajectoryBuffer(
      capacity=100_000,
      dim=input_dim,
      batch_size=batch_size,
    )
    if load_dir and os.path.exists(load_dir):
        with open(f"{load_dir}/buffer_state_latest.pkl", "rb") as file:
          buffer_state = pickle.load(file)
          self.buffer.state = buffer_state
          print("buffer_state: loaded")


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
        s_diff_1 = dsdt - 0.5 * (u * u).sum(-1, keepdims=True) + a_cost_tgt + potential_cost + D * vddx
        s_diff_2 = vt - 0.5 * (dsdx * dsdx).sum(-1, keepdims=True) + a_cost_tgt + potential_cost + D * vddx
        loss = jnp.abs(s_diff_1 ** 2).mean() + jnp.abs(s_diff_2 ** 2).mean() 

        loss += (- dsdt + 0.5 * (dsdx * dsdx).sum(-1, keepdims=True)).mean() * self.reg_weight

        return loss

      def potential_loss(state, params, key, steps_count, weight, source, target):
        bs = source.shape[0]
        t_0, t_1 = jnp.zeros([bs, 1]), jnp.ones([bs, 1])
        x_0, x_1 = source, target
        dt = 1.0 / steps_count

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        def move(carry, _):
          t_, x_, key_ = carry
          _, dsdx = dsdtdx_fn(state.params, t_, x_, x_0)
          sigma = self.flow.compute_sigma_t(t_)
          key_, key_s = jax.random.split(key_)
          x_next = x_ - dt * dsdx + sigma * jax.random.normal(key_s, shape=x_.shape) * dt**0.5
          t_next = t_ + dt
          return (t_next, x_next, key_), TimedX(t_, x_)
        
        (_, x_last ,_), result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(x_last)

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0)).mean()
        reg_loss = 0

        return (reg_loss + dual_loss)  * weight, result

      def loss_fn(state, params, key, t_sample, x_sample, target_state, source, target):
        control_loss_value = am_loss_sample(state, params, key, t_sample, x_sample, target_state)
        potential_loss_value, x_seq = potential_loss(state, params, key, self.control_steps, 1.0, source, target)
        loss_value = control_loss_value + potential_loss_value
        return loss_value, {
          "control_loss_value": control_loss_value,
          "potential_loss_value": potential_loss_value,
          "x_seq": x_seq,
        }

      def loss_with_potential_fn(state, params, key, target_state, source, target):
        control_loss_value = am_loss(state, params, key, source, target, target_state)
        potential_loss_value, x_seq = potential_loss(state, params, key, self.control_steps, 1.0, source, target)
        loss_value = control_loss_value + potential_loss_value
        return loss_value, {
          "control_loss_value": control_loss_value,
          "potential_loss_value": potential_loss_value,
          "x_seq": x_seq,
        }

      @with_mesh
      @jax.jit
      def train_step_cost(state, key, source, target, t_sample, x_sample, target_state):
        source = jax.lax.with_sharding_constraint(source, P('data'))
        target = jax.lax.with_sharding_constraint(target, P('data'))
        t_sample = jax.lax.with_sharding_constraint(t_sample, P('data'))
        x_sample = jax.lax.with_sharding_constraint(x_sample, P('data'))

        # grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        # (loss, info), grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state, source, target)
        # state = state.apply_gradients(grads=grads)
        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, self.control_steps, 1.0, source, target)

        g_norm_control = optax.global_norm(control_grads)
        g_norm_potential = optax.global_norm(potential_grads)
        scale = g_norm_potential / g_norm_control
        state = state.apply_gradients(grads=jax.tree.map(
          lambda gc, gp: gc * self.control_weight * scale + gp,
          control_grads,
          potential_grads
        ))

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)
        
        # return state, info["control_loss_value"], info["potential_loss_value"], info["x_seq"], target_state
        return state, loss, loss_potential, x_seq, target_state

      @with_mesh
      @jax.jit
      def train_step_with_potential(state, key, source, target, target_state):
        source = jax.lax.with_sharding_constraint(source, P('data'))
        target = jax.lax.with_sharding_constraint(target, P('data'))

        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, source, target, target_state)
        
        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, self.control_steps, 1.0, source, target)

        # grad_fn = jax.value_and_grad(loss_with_potential_fn, argnums=1, has_aux=True)
        # (loss, info), grads = grad_fn(state, state.params, key, target_state, source, target)
        # state = state.apply_gradients(grads=grads)

        g_norm_control = optax.global_norm(control_grads)
        g_norm_potential = optax.global_norm(potential_grads)
        scale = g_norm_potential / g_norm_control
        state = state.apply_gradients(grads=jax.tree.map(
          lambda gc, gp: gc * self.control_weight * scale + gp,
          control_grads,
          potential_grads
        ))

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)
        
        # return state, info["control_loss_value"], info["potential_loss_value"], info["x_seq"], target_state
        return state, loss, loss_potential, x_seq, target_state
      
      return train_step_cost, train_step_with_potential
  

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      collect_buffer_iters: int = 0,
      update_potential_every: int = 1,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
      eval_every: int = 5_000,
      save_dir: str = None,
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

      if it > collect_buffer_iters and it % update_potential_every != 0:
          _sample = self.buffer.sample()
          x_sample, t_sample = _sample["x"], _sample["t"]
          self.state, loss, loss_potential, tx_seq, self.target_state = self.train_step_cost(self.state, it_key, src, tgt, t_sample, x_sample, self.target_state)
      else:
          self.state, loss, loss_potential, tx_seq, self.target_state = self.train_step_with_potential(self.state, it_key, src, tgt, self.target_state)

      training_logs["potential_loss"].append(loss_potential.item())
      training_logs["cost_loss"].append(loss.item())

      x_seq = tx_seq.x.reshape(-1, tx_seq.x.shape[-1])
      t_seq = tx_seq.t.reshape(-1)
      self.buffer.append(x=x_seq, t=t_seq)

      if it % eval_every == 0 and it > 0 and callback is not None:
        callback(it, training_logs, self.transport)
        pbar.set_postfix({"pot_loss": loss_potential,
                          "cost_loss": loss})
        if save_dir is not None:
          self.save(save_dir, it=it)

      it += 1
      if it >= n_iters:
        break

    return training_logs

  def save(self, save_dir: str, it: int):
    with open(f"{save_dir}/opt_state_step_{it}.pkl", "wb") as file:
      pickle.dump(self.state.opt_state, file)
    with open(f"{save_dir}/params_step_{it}.pkl", "wb") as file:
      pickle.dump(self.state.params, file)
    with open(f"{save_dir}/opt_state_latest.pkl", "wb") as file:
      pickle.dump(self.state.opt_state, file)
    with open(f"{save_dir}/params_latest.pkl", "wb") as file:
      pickle.dump(self.state.params, file)
    with open(f"{save_dir}/step_latest.pkl", "wb") as file:
      pickle.dump(self.state.step, file)
    with open(f"{save_dir}/buffer_state_latest.pkl", "wb") as file:
      pickle.dump(self.buffer.state, file)

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
  
    @with_mesh
    @jax.jit
    def inference(state, x_0):

      the_ones = jnp.ones([x.shape[0],1])
      x_0 = jax.lax.with_sharding_constraint(x_0, P('data'))
      the_ones = jax.lax.with_sharding_constraint(the_ones, P('data'))

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
      
      def move(carry, _):
        t_, x_, cost, key_ = carry

        u = dsdx_fn(state.params, t_ * the_ones, x_, x_0)
        U_t = self.flow.compute_potential(t_, x_)
        sigma = self.flow.compute_sigma_t(t_)
        key_, key_s = jax.random.split(key_)
        x_ = x_ - dt * u + sigma * jax.random.normal(key_s, shape=x_.shape) * dt**0.5
        t_ = t_ + dt
        cost += 0.5 * (u * u).sum(-1).mean() * dt + U_t.mean() * dt * self.potential_weight
        return (t_, x_, cost, key_), x_

      (_, _, cost, _), result = jax.lax.scan(move, (t_0, x_0, 0.0, loop_key), None, length=n)
      return cost, result
    
    cost, result = inference(self.state, x)
    result = jax.lax.stop_gradient(result)

    x_seq = [TimedX(t=t_0, x=x)]

    for i in range(n):
      t_ = x_seq[-1].t + dt
      x_seq.append(TimedX(t=t_, x=result[i]))
      
    return cost, x_seq
