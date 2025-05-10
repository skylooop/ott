# import diffrax
import os
import pickle
from functools import partial

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

import flashbax as fbx
import lineax as lx
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

import diffrax
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state

from ott2 import utils
from ott2.neural.methods.flows import dynamics
from ott2.solvers import utils as solver_utils

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

# multigpu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

NPROC = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))

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
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      acc_weight: float,
      pretrain_steps: float = 5_000,
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
    self.pretrain_steps = pretrain_steps

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
        tx=optax.identity()
      )

    self.train_step_fast, self.train_step_cost = self._get_step_fn()
    self.inference = self.get_inference()


  def reset(
      self,
      optimizer: Optional[optax.GradientTransformation],
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      acc_weight: float
  ):
    
    self.potential_weight = potential_weight
    self.control_weight = control_weight
    self.reg_weight = reg_weight
    self.acc_weight = acc_weight
    self.pretrain_steps = 0
   
    params = self.state.params
  
    self.state = train_state.TrainState.create(
      apply_fn=self.value_model.apply,
      params=params,
      tx=optimizer
    )

    self.target_state = train_state.TrainState.create(
      apply_fn=self.value_model.apply,
      params=jax.tree.map(lambda x: jnp.copy(x), params),
      tx=optax.identity()
    )

    self.train_step_fast, self.train_step_cost = self._get_step_fn()
    self.inference = self.get_inference()


  def _get_step_fn(self) -> Callable:
      
      # def am_loss(state, params, key_t, source, target, target_state):
      #   bs = source.shape[0]
      #   t = self.time_sampler(key_t, bs)
      #   x_0, x_1 = source, target
      #   x_t = self.flow.compute_xt(key_t, t, x_0, x_1)
        
      #   return am_loss_sample(state, params, key_t, t, x_t, target_state, reg_weight=0)
      
      def am_loss_sample(state, params, key_t, t_sample, x_sample, target_state, reg_weight):
        
        x_t = x_sample
        t = t_sample.reshape(-1, 1)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_t)
        dsdt_tgt, dsdx_tgt = dsdtdx_fn(target_state.params, t, x_t, x_t)
        u = dsdx_tgt
        vt = dsdt_tgt

        dt = 1.0 / 100
        x_dt = x_t - jax.lax.stop_gradient(dsdx) * dt
        x_2dt = x_t - jax.lax.stop_gradient(dsdx) * dt * 2
        U_t = 0.4 * self.flow.compute_potential(t, x_t) + 0.3 * self.flow.compute_potential(t+dt, x_dt) + 0.3 * self.flow.compute_potential(t+dt*2, x_2dt)

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            fun = lambda __x: state.apply_fn(p,t,__x,x0).sum()
            return jnp.trace(jax.jacfwd(jax.jacrev(fun))(x))
        
        def normalize(x):
          norm = jnp.linalg.norm(x) + 1e-8
          return x / norm

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def acceleration(p, t, x, x0):
            fun = lambda __t, __x: state.apply_fn(p,__t,__x,x0).sum()
            dsdx_fn = jax.grad(fun, argnums=1)
            norm_rev = lambda __t, __x: normalize(jax.jacrev(fun, 1)(__t, __x))
            Dt, Dx = jax.jacfwd(norm_rev, argnums=[0, 1])(t, x)
            acc = Dt.squeeze() - Dx @ dsdx_fn(t, x)
            return acc
        
        a = acceleration(params, t, x_t, x_t)
        # a_tgt = acceleration(target_state.params, t, x_t, x_t)
        # a_cost_tgt = jnp.sqrt((a_tgt * a_tgt).reshape(x_t.shape[0], x_t.shape[1]).sum(-1, keepdims=True)) * self.acc_weight
        a_cost = jnp.sqrt((a * a).reshape(x_t.shape[0], x_t.shape[1]).sum(-1, keepdims=True) + 1e-8) * self.acc_weight

        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        s_diff_1 = dsdt - 0.5 * ((u @ At_T) * u).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + a_cost + D * laplacian(state.params, t, x_t, x_t).reshape(-1, 1)
        s_diff_2 = vt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + a_cost + D * laplacian(params, t, x_t, x_t).reshape(-1, 1)
        loss = jnp.abs(s_diff_1 ** 2).mean() + jnp.abs(s_diff_2 ** 2).mean()
        # loss += (- dsdt + 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + a_cost_tgt).mean() * reg_weight

        return loss

      def potential_loss(state, params, key, steps_count, weight, source, target):
        bs = source.shape[0]
        t_0, t_1 = jnp.zeros([bs, 1]), jnp.ones([bs, 1])
        x_0, x_1 = source, target
        dt = 1.0 / steps_count

        dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

        def move(carry, _):
          t_, x_, key_ = carry
          dsdx = dsdx_fn(state.params, t_, x_, x_0)
          sigma = self.flow.compute_sigma_t(t_)
          key_, key_s = jax.random.split(key_)
          x_next = x_ - dt * dsdx + sigma * jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
          t_next = t_ + dt

          # noise = jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
          # x_pred = x_ - dt * dsdx + sigma * noise
          # # Corrector step (Heun's method)
          # u_corr = dsdx_fn(state.params, t_ + dt, x_pred, x_0)
          # x_next = x_ - dt * 0.5 * (dsdx + u_corr) + sigma * noise  # Same noise for both steps
          # t_next = t_ + dt

          return (t_next, x_next, key_), TimedX(t_, x_)
        
        (_, x_last ,_), result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(x_last)

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0))
        dual_loss = (dual_loss.mean() * jnp.abs(dual_loss.mean()))
        
        return dual_loss * weight, result

      @with_mesh
      @jax.jit
      def train_step_cost(state, key, source, target, t_sample, x_sample, target_state, reg_weight, scale):
        source = jax.lax.with_sharding_constraint(source, P('data'))
        target = jax.lax.with_sharding_constraint(target, P('data'))
        t_sample = jax.lax.with_sharding_constraint(t_sample, P('data'))
        x_sample = jax.lax.with_sharding_constraint(x_sample, P('data'))

        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state, reg_weight)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 30, 1.0, source, target)

        g_norm_control = optax.global_norm(control_grads)
        g_norm_potential = optax.global_norm(potential_grads)
        # scale = jnp.clip(g_norm_potential / g_norm_control, min=0.01, max=10)
        scale_update = g_norm_potential / g_norm_control
        scale = scale_update * 0.1 + scale * 0.9

        state = state.apply_gradients(
          grads=jax.tree.map(lambda gc, gp: gc * scale * self.control_weight + gp, control_grads, potential_grads)
        )

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)

        return state, loss, loss_potential, x_seq, target_state, g_norm_control,  g_norm_potential, scale


      @with_mesh
      @jax.jit
      def train_step_fast(state, key, source, target, t_sample, x_sample, target_state, reg_weight, scale):
        t_sample = jax.lax.with_sharding_constraint(t_sample, P('data'))
        x_sample = jax.lax.with_sharding_constraint(x_sample, P('data'))

        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state, reg_weight)

        state = state.apply_gradients(
          grads=jax.tree.map(lambda gc: gc * scale * self.control_weight, control_grads)
        )

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)

        return state, loss, target_state
      
      
      return train_step_fast, train_step_cost
  

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
      eval_every: int = 5_000,
      save_dir: str = None,
  ) -> Dict[str, List[float]]:
    batch_size, input_dim = next(iter(loader))["src_lin"].shape
    self.buffer = TrajectoryBuffer(
      capacity=100_000,
      dim=input_dim,
      batch_size=batch_size,
    )
    
    loop_key = utils.default_prng_key(rng)
    training_logs = {"cost_loss": [], "potential_loss": [], "g_norm": [], "g_norm_potential" : []}
    it = 0
    g_norm, g_norm_potential = 0, 0
    scale = jnp.array(0.1)
    pbar = tqdm(loader, total=n_iters, colour='green', dynamic_ncols=True)
    for batch in pbar:
      # batch = jtu.tree_map(jnp.asarray, batch)
    
      src, tgt = batch["src_lin"], batch["tgt_lin"]
      # src_cond = batch.get("src_condition")
      it_key = jax.random.fold_in(loop_key, it)

      if it > self.pretrain_steps:
          _sample = self.buffer.sample()
          x_sample, t_sample = _sample["x"], _sample["t"]
          reg_weight = self.reg_weight
          # if it % 4 == 0:
          #     tt = self.time_sampler(it_key, src.shape[0])
          #     t_sample = t_sample * (1 - tt) + jnp.ones_like(t_sample) * tt
          #     x_sample = self.flow.compute_xt(it_key, tt, x_sample, tgt)
      else:
          bs = src.shape[0]
          t_sample = self.time_sampler(it_key, bs)
          x_sample = self.flow.compute_xt(it_key, t_sample, src, tgt)
          reg_weight = 0.0
          # self.state, loss, loss_potential, tx_seq, self.target_state, g_norm, g_norm_potential = self.train_step_with_potential(self.state, it_key, src, tgt, self.target_state)
      
      if it % 2 == 0:
        self.state, loss, loss_potential, tx_seq, self.target_state, g_norm, g_norm_potential, scale = self.train_step_cost(
          self.state, it_key, src, tgt, t_sample, x_sample, self.target_state, reg_weight, scale
        )
      else:
        self.state, loss, self.target_state = self.train_step_fast(
          self.state, it_key, src, tgt, t_sample, x_sample, self.target_state, reg_weight, scale
        )
        tx_seq = None

      if it % 2 == 0:
        training_logs["potential_loss"].append(loss_potential)
        training_logs["cost_loss"].append(loss)
        training_logs["g_norm"].append(g_norm)
        training_logs["g_norm_potential"].append(g_norm_potential)

        traj_idx = np.random.randint(tx_seq.x.shape[1])
        x_seq = tx_seq.x[:, traj_idx].reshape(-1, tx_seq.x.shape[-1])
        t_seq = tx_seq.t[:, traj_idx].reshape(-1)
        self.buffer.append(x=x_seq, t=t_seq)

      if it % 100 == 0 and it > 0:
        pbar.set_postfix({"pot_loss": loss_potential,
                          "cost_loss": loss,
                          "g_norm": g_norm, 
                          "g_norm_potential": g_norm_potential})

      if it % eval_every == 0 and it > 0 and callback is not None:
        callback(it, training_logs, self.transport)
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

  
  def get_inference(self):

    dt = 1.0 / 30
    t_0 = 0.0
    n = 30
  
    @jax.jit
    def inference(state, x_0, loop_key):

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
      
      def move(carry, _):
        t_, x_, cost, key_ = carry
        u = dsdx_fn(state.params, t_ * jnp.ones([x_0.shape[0],1]), x_, x_0)
        # At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
        U_t = self.flow.compute_potential(t_, x_)
        sigma = self.flow.compute_sigma_t(t_)
        key_, key_s = jax.random.split(key_)
        x_ = x_ - dt * u + sigma * jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
        t_ = t_ + dt
        # noise = jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
        # x_pred = x_ - dt * u + sigma * noise
        # # Corrector step (Heun's method)
        # u_corr = dsdx_fn(state.params, (t_ + dt) * jnp.ones([x.shape[0],1]), x_pred, x_0)
        # x_ = x_ - dt * 0.5 * (u + u_corr) + sigma * noise  # Same noise for both steps
        # t_ = t_ + dt

        cost += 0.5 * (u * u).sum(-1).mean() * dt + U_t.mean() * dt 
        return (t_, x_, cost, key_), x_
          
      (_, _, cost, _), result = jax.lax.scan(move, (t_0, x_0, 0.0, loop_key), None, length=n)

      return cost, result
    
    return inference

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    
    loop_key = jax.random.PRNGKey(0)
    t_0 = 0.0
    n = 30
    dt = 1.0 / 30
  
    cost, result = self.inference(self.state, x, loop_key)
    x_seq = [TimedX(t=t_0, x=x)]

    for i in range(n):
      t_ = x_seq[-1].t + dt
      x_seq.append(TimedX(t=t_, x=result[i]))
      
    return cost, x_seq
  

  def transport2(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    
    dt = 1.0 / 30
    # t_0 = 0.0
    n = 30
    loop_key = jax.random.PRNGKey(0)
    
    def solve_ode(state, x):

      def vector_field(t, y, args):
        dsdx_fn, key_s = args
        u = dsdx_fn(state.params, jnp.array(t)[None], y, y)
        return -u
        # sigma = self.flow.compute_sigma_t(t)
        # it = (t.squeeze() * 1000).astype(int)
        # it_key = jax.random.fold_in(key_s, it)
        # return -u + sigma * jax.random.normal(it_key, shape=x.shape)

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
      ode_term = diffrax.ODETerm(vector_field)

      def diffusion(t, y, args):
        s = self.flow.compute_sigma_t(t)
        diagonal = jnp.array([s, s])
        return lx.DiagonalLinearOperator(diagonal)
      
      brownian_motion = diffrax.VirtualBrownianTree(0.0, 1.0, tol=1e-5, shape=x.shape, key=loop_key)
      terms = diffrax.MultiTerm(ode_term, diffrax.ControlTerm(diffusion, brownian_motion))
      saveat = diffrax.SaveAt(ts=jnp.linspace(0, 1, n))
      
      result = diffrax.diffeqsolve(
          terms,
          t0=0,
          t1=1,
          y0=x,
          args=(dsdx_fn, loop_key),
          solver=diffrax.Tsit5(),
          dt0=dt,
          saveat=saveat,
          # stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
          **kwargs,
      )
      return result.ts, result.ys
      
    result_ts, result_ys = jax.jit(jax.vmap(solve_ode, in_axes=(None, 0)))(self.state, x)
    x_seq = []

    for i in range(result_ts.shape[1]):
      x_seq.append(TimedX(t=result_ts[:, i], x=result_ys[:, i]))
      
    return 0.0, x_seq
