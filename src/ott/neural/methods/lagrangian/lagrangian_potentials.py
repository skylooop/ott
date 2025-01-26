from abc import abstractmethod
import jax.numpy as jnp
import jax
import numpy as np
from ott.datasets import create_lagrangian_ds

import functools
from flax.struct import PyTreeNode
from jaxtyping import Float, ArrayLike
import matplotlib.pyplot as plt

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
            t = jax.nn.sigmoid(10.*(t-0.5))

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
        #assert x.ndim == 1 and x.shape[0] == self.DD
        Ux = (jax.nn.sigmoid((x[0] - self.xmin) / self.temp) - \
              jax.nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (jax.nn.sigmoid((x[1] - self.ymin) / self.temp) - \
              jax.nn.sigmoid((x[1] - self.ymax) / self.temp))
        U = Ux * Uy
        return -self.M * U
    
def my_softplus(x, beta=1, threshold=20):
    # mirroring the pytorch implementation https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = jax.lax.select(x * beta < threshold, x, jax.numpy.ones_like(x))
    return jax.lax.select(x * beta < threshold, 1/beta * jax.numpy.log(1 + jax.numpy.exp(beta * x_safe)), x)

class DrunkernSpider(LagrangianPotentialBase):
    x_axes_bounds: tuple = (-14, 14)
    y_axes_bounds: tuple = (-14, 14)
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='drunken_spider', mean_left=-8.5, mean_right=8.5, key=None)
    
    def obstacle_cfg_drunken_spider(self):
        xys = [[-7, 0.5], [-7, -7.5]]
        widths = [14, 14]
        heights = [7, 7]
        return xys, widths, heights
    
    def get_samples(self, size, key):
        drunken_spider_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(drunken_spider_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data
    
    def __call__(self, xt):
        x, y = xt[0], xt[1]

        def cost_fn(xy, width, height):

            xbound = xy[0], xy[0] + width
            ybound = xy[1], xy[1] + height

            a = -5 * (x - xbound[0]) * (x - xbound[1])
            b = -5 * (y - ybound[0]) * (y - ybound[1])

            cost = my_softplus(a, beta=20, threshold=1) * my_softplus(b, beta=20, threshold=1)
            assert cost.shape == xt.shape[:-1]
            return cost

        return 10 * sum(
            cost_fn(xy, width, height)
            for xy, width, height in zip(*self.obstacle_cfg_drunken_spider())
        )

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
        Ux = (jax.nn.sigmoid((x[0] - self.xmin) / self.temp) - \
                jax.nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (jax.nn.sigmoid((x[1] - self.ymin) / self.temp) - \
                jax.nn.sigmoid((x[1] - self.ymax) / self.temp)) - 1.
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

    x_axes_bounds = (-2.5, 2.5)
    y_axes_bounds = (-2.5, 2.5)
    sampler_func = functools.partial(create_lagrangian_ds, geometry_str='babymaze')

    def get_samples(self, size, key):
        maze_sampler = self.sampler_func(batch_size=size, key=key)
        sampler = next(iter(maze_sampler))
        source_data = sampler['src_lin']
        target_data = sampler['tgt_lin']
        return source_data, target_data

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        Ux1 = (jax.nn.sigmoid((x[0] - self.xmin1) / self.temp) - \
                jax.nn.sigmoid((x[0] - self.xmax1) / self.temp))
        Ux2 = (jax.nn.sigmoid((x[0] - self.xmin2) / self.temp) - \
                jax.nn.sigmoid((x[0] - self.xmax2) / self.temp))

        Uy1 = (jax.nn.sigmoid((x[1] - self.ymin1) / self.temp) - \
                jax.nn.sigmoid((x[1] - self.ymax1) / self.temp)) - 1.

        Uy2 = (jax.nn.sigmoid((x[1] - self.ymin2) / self.temp) - \
                jax.nn.sigmoid((x[1] - self.ymax2) / self.temp)) - 1.
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

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D

        V = 0.
        for i in range(self.centers.shape[0]):
            dist = jnp.linalg.norm(x - self.centers[i])
            V -= self.M * jax.nn.sigmoid((self.radius - dist) / self.temp)
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

        return - self.M * jax.nn.sigmoid((-self.c_sq - d) / self.temp)


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
        V -= self.M * jax.nn.sigmoid((self.c - d) / self.temp)

        d = self.a*(x[0]-self.centers[1][0])**2 + \
            self.b*(x[1]-self.centers[1][1])**2
        V -= self.M * jax.nn.sigmoid((self.c - d) / self.temp)

        return V
    
## Utils

def plot_potential(potential, fig=None, ax=None, invert_sign=False):
    if ax is None:
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()

    x, y = potential.get_boundaries()
    x, y = jnp.meshgrid(x, y)
    mesh = jnp.stack([x.ravel(), y.ravel()], -1).reshape(-1, 2)
    if invert_sign:
        z = jax.vmap(potential)(mesh).reshape([x.shape[0], x.shape[0]])
    else:
        z = -jax.vmap(potential)(mesh).reshape([x.shape[0], x.shape[0]])

    contour = ax.contourf(x, y, z, x.shape[0], cmap='Blues')
    plt.colorbar(ax=ax, mappable=contour)
        
    ax.set_xlim(*potential.x_axes_bounds)
    ax.set_ylim(*potential.y_axes_bounds)
    ax.grid(False)
    return fig, ax
    
def draw_trajs(trajs: Float[ArrayLike, 'timestep point dim=2'], ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    colors = sns.color_palette("pastel", trajs.shape[1])
    
    for point in range(trajs.shape[1]):
        for t in range(1, trajs.shape[0]):
            ax.plot([trajs[t-1, point, 0], trajs[t, point, 0]],
                    [trajs[t-1, point, 1], trajs[t, point, 1]],
                    color=colors[point], linestyle="-", linewidth=1, marker='o', alpha=0.6, markersize=1)
    return ax