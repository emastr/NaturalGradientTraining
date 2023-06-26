from matplotlib import pyplot as plt
from natgrad.domains import Hyperrectangle, Domain
from jax import numpy as jnp
from jax import vmap


def default_axis(func):
    def new_func(*args, **kwargs):
        return func(*args, ax=kwargs.pop("ax", plt.gca()), **kwargs)
    return new_func
    

@default_axis
def plot_func(func, domain, ax: plt.Axes, N=200, **kwargs):
    """Plot a 2d domain.

    Args:
        domain (tuple): A tuple of two floats (a, b) representing the domain of a function.
    """
    bounding_box: Hyperrectangle = domain.bounding_box()
    lb, rb = bounding_box._l_bounds, bounding_box._r_bounds
    X, Y = jnp.meshgrid(jnp.linspace(lb[0], rb[0], N), jnp.linspace(lb[1], rb[1], N))
    xyflat = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    Z = func(xyflat).reshape((N, N))
    mask = vmap(domain.mask, (0))(xyflat).reshape((N, N))
    Z = jnp.where(mask, Z, jnp.nan)
    return ax.pcolormesh(X, Y, Z, **kwargs)
    
    
    
    
    