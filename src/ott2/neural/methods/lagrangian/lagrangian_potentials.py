from abc import abstractmethod
import jax.numpy as jnp
import jax
from jax import nn
import numpy as np
from ott2.datasets import create_lagrangian_ds

import functools
from dataclasses import dataclass
from flax.struct import PyTreeNode

class LagrangianPotentialBase(PyTreeNode):
    D: int = 2
    scale: int = 1
    
    M_bounds = (0.001, 0.01)
    temp_bounds = (1e-1, 1e-2)

    M: float = 0.01
    temp: float = 1e-2

    x_axes_bounds = (-1.5, 1.5)
    y_axes_bounds = (-1.5, 1.5)
    
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    # Used for introducing smootheness of potential during training
    def get_annealed_params(self, t):
        assert 0 <= t and t <= 1
        if 1.-t < 1e-3:
            t = 1.
        elif t < 1e-3:
            t = 0.
        else:
            t = nn.sigmoid(10.*(t-0.5))

        M_start, M_end = self.M_bounds
        temp_start, temp_end = self.temp_bounds
        new_M = M_start + (M_end - M_start) * t
        new_temp = temp_start + (temp_end - temp_start) * t
        return self.replace(M=new_M, temp=new_temp)
    
    def get_boundaries(self):
        return jnp.linspace(self.x_axes_bounds[0] * self.scale, self.x_axes_bounds[1] * self.scale, num=200), \
            jnp.linspace(self.y_axes_bounds[0] * self.scale, self.y_axes_bounds[1] * self.scale, num=200)
            
class BoxPotential(LagrangianPotentialBase):
    xmin: float = -0.5
    xmax: float = 0.5
    ymin: float = -0.5
    ymax: float = 0.5
    temp_bounds = (1e-1, 2e-3)
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='box', key=None)

    def get_boundaries(self):
        return jnp.linspace(self.x_axes_bounds[0] * self.scale, self.x_axes_bounds[1] * self.scale), \
            jnp.linspace(self.y_axes_bounds[0] * self.scale, self.y_axes_bounds[1] * self.scale)
    
    def get_samples(self, size):
        box_sampler = self.sampler_func(batch_size=size)
        sampler = next(iter(box_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data
    
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
              nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
              nn.sigmoid((x[1] - self.ymax) / self.temp))
        U = Ux * Uy
        return -self.M * U


class Styblinski_tan(LagrangianPotentialBase):
    def __call__(self, x, xmin, xmax):
        a = -5
        b = 5
        x = a + ((x - xmin)*(b - a)) / (xmax - xmin)
        return 0.5*np.sum(jnp.power(x, 4) - 16*jnp.power(x, 2) + 5*x, axis=0)

class SlitPotential(LagrangianPotentialBase):
    xmin: float = -0.1
    xmax: float = 0.1
    ymin: float = -0.25
    ymax: float = 0.25
    M_bounds = (0., 1.)
    temp_bounds = (1e-1, 2e-3)

    x_axes_bounds = (-1.5, 1.5)
    y_axes_bounds = (-2., 2.)
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='slit')

    def get_samples(self, size, key):
        vneck_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(vneck_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data
    
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax) / self.temp)) - 1.
        U = Ux * Uy
        return U

class BabyMazePotential(LagrangianPotentialBase):
    xmin1: float = -0.5
    xmax1: float = -0.3
    ymin1: float = -1.99
    ymax1: float = -0.15
    xmin2: float = 0.3
    xmax2: float = 0.5
    ymin2: float = 0.15
    ymax2: float = 1.99
    M_bounds = (0., 10.)
    # temp_bounds = (1e-1, 2e-3)

    x_axes_bounds = (-1.5, 1.5)
    y_axes_bounds = (-1.5, 1.5)
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='babymaze')

    def get_samples(self, size, key):
        maze_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(maze_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux1 = (nn.sigmoid((x[0] - self.xmin1) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax1) / self.temp))
        Ux2 = (nn.sigmoid((x[0] - self.xmin2) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax2) / self.temp))

        Uy1 = (nn.sigmoid((x[1] - self.ymin1) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax1) / self.temp)) - 1.

        Uy2 = (nn.sigmoid((x[1] - self.ymin2) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax2) / self.temp)) - 1.
        U = Ux1 * Uy1 + Ux2 * Uy2
        return self.M*U


class WellPotential(LagrangianPotentialBase):
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.sum(x**2)
        return self.M*U


class HillPotential(LagrangianPotentialBase):
    M_bounds = (0., 0.05)
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        U = -jnp.exp(-jnp.sum(x**2))
        return self.M*U


class GSB_GMM_Potential(LagrangianPotentialBase):
    centers = jnp.array([[6,6], [6,-6], [-6,-6]])
    radius = 1.5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    x_axes_bounds = (-19., 19.)
    y_axes_bounds = (-19., 19.)

    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='gmm')

    def get_samples(self, size, key):
        vneck_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(vneck_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.
        for i in range(self.centers.shape[0]):
            dist = jnp.linalg.norm(x - self.centers[i])
            V -= nn.softplus(100 * (self.radius - dist)) * (self.radius > dist).astype(jnp.int32)

        return V

class VNeck_Potential(LagrangianPotentialBase):
    c_sq = 0.36
    coef = 5
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    x_axes_bounds = (-3., 3.)
    y_axes_bounds = (-5., 5.)
    
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='vneck')
    
    def get_samples(self, size, key):
        box_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(box_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data
    
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        xs_sq = x*x
        d = self.coef * xs_sq[0] - xs_sq[1]

        return - self.M * nn.sigmoid((-self.c_sq - d) / self.temp)
    

class VNeck_bench(LagrangianPotentialBase):

    x_axes_bounds = (-8., 8.)
    y_axes_bounds = (-2., 2.)

    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='vneck')
    
    def get_samples(self, size, key):
        box_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(box_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data


    def obstacle_cfg_vneck(self):
        c_sq = 0.36
        coef = 5
        return c_sq, coef


    def __call__(self, xt):
    
        assert xt.shape[-1] == 2

        c_sq, coef = self.obstacle_cfg_vneck()

        xt_sq = xt * xt
        d = coef * xt_sq[0] - xt_sq[1]

        cond = jnp.astype(jax.nn.softplus((-c_sq - d)) > 0.53, jnp.int32)
        cond_1 = jnp.astype(jax.nn.softplus((-c_sq - d)) < 5, jnp.int32)

        return -jax.nn.softplus((-c_sq - d)) * cond * cond_1 - 5 * (1 - cond_1)
    

class STunnel_bench(LagrangianPotentialBase):

    x_axes_bounds = (-13, 13)
    y_axes_bounds = (-13, 13)


    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='stunnel')
    
    def get_samples(self, size, key):
        box_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(box_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data


    def obstacle_cfg(self):
        a, b, c = 20, 1, 90
        centers = [[5, 6], [-5, -6]]
        return a, b, c, centers


    def __call__(self, xt):


        a, b, c, centers = self.obstacle_cfg()

        D = xt.shape[0]
        assert D == 2

        _xt = xt.reshape(D)
        x, y = _xt[0], _xt[1]

        d = a * (x - centers[0][0]) ** 2 + b * (y - centers[0][1]) ** 2
        # c1 = 1500 * (d < c)
        c1 = nn.softplus(c - d)

        d = a * (x - centers[1][0]) ** 2 + b * (y - centers[1][1]) ** 2
        # c2 = 1500 * (d < c)
        c2 = nn.softplus(c - d)

        cost = (c1 + c2)
        return -cost



class STunnel_Potential(LagrangianPotentialBase):
    a, b, c = 20, 1, 90
    centers = [[5,6], [-5,-6]]
    M_bounds = (0., 0.1)
    temp_bounds = (1., 0.1)

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.0
        d = self.a*(x[0]-self.centers[0][0])**2 + \
            self.b*(x[1]-self.centers[0][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        d = self.a*(x[0]-self.centers[1][0])**2 + \
            self.b*(x[1]-self.centers[1][1])**2
        V -= self.M * nn.sigmoid((self.c - d) / self.temp)

        return V
    

class Sphere_Potential(PyTreeNode):
    dim: int = 3
    r: float = 0.98
    sigma: float = 0.3
    x_axes_bounds = (-2*r, 2*r)
    y_axes_bounds = (-2*r, 2*r)

    def sampler_func(self):
        return functools.partial(create_sphere_ds, dim=self.dim, sigma=self.sigma)

    def get_samples(self, size, key):
        box_sampler = self.sampler_func()(batch_size=size, key=key)
        sampler = next(box_sampler)
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data

    def __call__(self, x):
        x_norm = jnp.linalg.norm(x, axis=-1)
        v = jax.lax.cond(
            x_norm >= self.r,
            lambda:  0.,
            lambda: -1.
        )
        return v

    def get_boundaries(self):
        x = np.linspace(-self.r, self.r, num=200)
        return [x] * self.dim
    


def create_sphere_ds(dim: int, 
                    sigma: float, 
                    batch_size: int, 
                    key: jax.random.PRNGKey):
   
    # Create base vectors in pure JAX
    src_base = jnp.zeros((batch_size, dim)).at[:, 2].set(1.0)
    trg_base = jnp.zeros((batch_size, dim)).at[:, 2].set(-1.0)
    
    while True:
        key, src_key, trg_key = jax.random.split(key, 3)
        
        # Generate and normalize source samples
        src_sample = jax.random.normal(src_key, (batch_size, dim)) * sigma + src_base
        src_sample = src_sample / jnp.linalg.norm(src_sample, axis=-1, keepdims=True)
        
        # Generate and normalize target samples
        trg_sample = jax.random.normal(trg_key, (batch_size, dim)) * sigma + trg_base
        trg_sample = trg_sample / jnp.linalg.norm(trg_sample, axis=-1, keepdims=True)

        ref_sample = jax.random.normal(src_key, (batch_size, dim)) * sigma + trg_base
        ref_sample = ref_sample / jnp.linalg.norm(ref_sample, axis=-1, keepdims=True)

        yield {
            "src_lin": src_sample,
            "tgt_lin": trg_sample,
            "ref": ref_sample
        }
        

class Smooth_Sphere_Potential(PyTreeNode):
    dim: int = 3
    r: float = 0.98
    sigma: float = 0.3
    x_axes_bounds = (-2*r, 2*r)
    y_axes_bounds = (-2*r, 2*r)
    temp: float = 1e-2

    def sampler_func(self):
        return functools.partial(create_sphere_ds, dim=self.dim, sigma=self.sigma)

    def get_samples(self, size, key):
        box_sampler = self.sampler_func()(batch_size=size, key=key)
        sampler = next(box_sampler)
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data

    def __call__(self, x):
        d = jnp.linalg.norm(x, axis=-1)
        v = - nn.sigmoid((1.0 - d) / self.temp)
        return v

    def get_boundaries(self):
        x = np.linspace(-self.r, self.r, num=200)
        return [x] * self.dim



import functools

def _random_centres(key, base, K, mean_spread):
    """Generate K random means on the sphere around `base`."""
    noise = jax.random.normal(key, (K, base.size)) * mean_spread
    centres = base + noise
    centres = centres / jnp.linalg.norm(centres, axis=1, keepdims=True)
    return centres                               # (K, dim)

def create_mog_sphere_ds(*, dim: int, sigma: float,
                         batch_size: int, K: int,
                         mean_spread: float, key):

    base_north = jnp.array([0., 0., 1.] + [0.]*(dim-3))
    base_south = jnp.array([0., 0., -1.] + [0.]*(dim-3))

    key_1 = jax.random.PRNGKey(1)
    k_cent_n, k_cent_s = jax.random.split(key_1, 2)

    # create component means --------------------------------------------
    centres_n = _random_centres(k_cent_n, base_north, K, mean_spread)
    centres_s = _random_centres(k_cent_s, base_south, K, mean_spread)

    while True:
        # split keys ---------------------------------------------------------
        key, k_cent_n, k_cent_s, k_sel_n, k_sel_s, k_noise = \
            jax.random.split(key, 6)

        # choose which component each sample comes from ---------------------
        comps_n = jax.random.randint(k_sel_n, (batch_size,), 0, K)
        comps_s = jax.random.randint(k_sel_s, (batch_size,), 0, K)

        means_n = centres_n[comps_n]                      # (B, dim)
        means_s = centres_s[comps_s]                      # (B, dim)

        # sample, add noise, project back to sphere -------------------------
        k_noise_n, k_noise_s = jax.random.split(k_noise)
        src = jax.random.normal(k_noise_n, (batch_size, dim)) * sigma + means_n
        tgt = jax.random.normal(k_noise_s, (batch_size, dim)) * sigma + means_s

        src = src / jnp.linalg.norm(src, axis=-1, keepdims=True)
        tgt = tgt / jnp.linalg.norm(tgt, axis=-1, keepdims=True)
        ref = (src * 1.0).at[:, 2].set(-src[:, 2])

        yield {"src_lin": src, "tgt_lin": tgt, "ref": ref,
               "src_comp": comps_n, "tgt_comp": comps_s}

class Smooth_Sphere_Potential_2(PyTreeNode):
    dim: int = 3
    r: float = 0.98
    sigma: float = 0.1
    K: int = 4
    mean_spread: float = 0.4
    temp: float = 1e-3
    x_axes_bounds = (-2*r, 2*r)
    y_axes_bounds = (-2*r, 2*r)

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    def sampler_func(self):
        return functools.partial(create_mog_sphere_ds,
                                 dim=self.dim,
                                 sigma=self.sigma,
                                 K=self.K,
                                 mean_spread=self.mean_spread)

    def get_samples(self, size, key):
        gen = self.sampler_func()(batch_size=size, key=key)
        batch = next(gen)
        return batch["src_lin"], batch["tgt_lin"]

    # ---------------------------------------------------------------------
    # Radial soft barrier towards the unit sphere
    # ---------------------------------------------------------------------
    def __call__(self, x):
        d = jnp.linalg.norm(x, axis=-1)
        return -jax.nn.sigmoid((1.0 - d) / self.temp)

    def get_boundaries(self):
        x = np.linspace(-self.r, self.r, num=200)
        return [x] * self.dim


