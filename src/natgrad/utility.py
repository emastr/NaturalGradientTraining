"""
Contains diverse utility functions.

The functions in this module serve different purposes. Their role in 
the natgrad library is not yet clear to me so they are grouped here
and await further placement & improvement.

Currently there are:

1) Grid Linesearch Methods.
2) Some pre-implemented trafos for inner products without good structure

"""
import jax.numpy as jnp
from jax import vmap, jit
from jaxtyping import Array, Float, PyTree, jaxtyped
from jax.flatten_util import ravel_pytree
from typeguard import typechecked as typechecker
from typing import Callable

from natgrad.inner import model_del_i_factory

def grid_line_search_factory(loss, steps):
    
    def loss_at_step(step, params, tangent_params):
        updated_params = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]
        return loss(updated_params)
        
    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None)))    

    @jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, tangent_params)], step_size
    
    return grid_line_search_update

#------------some PDE trafos--------------------------#

model_del_0 = model_del_i_factory(argnum=0)
model_del_1 = model_del_i_factory(argnum=1)

def model_heat_eq_factory(diffusivity=1.):
    @jaxtyped
    @typechecker
    def model_heat_eq(
                u_theta: Callable[[Float[Array, "2"]], Float[Array, ""]],
                g: Callable[[Float[Array, "2"]], PyTree]
        ) -> Callable[[Float[Array, "2"]], PyTree]:
        """
        Heat Eq for function of signature (2,) ---> Pytree

        Intended to use when defining inner products on model space.

        """
        
        dg_1 = model_del_0(u_theta, g)
        ddg_2 = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def return_heat_eq(x: Float[Array, "2"]) -> PyTree:
            flat_dg_1, unravel = ravel_pytree(dg_1(x))
            flat_ddg_2, unravel = ravel_pytree(ddg_2(x))
            return unravel(flat_dg_1 - diffusivity * flat_ddg_2)#0.25
        
        return return_heat_eq

    return model_heat_eq

def model_nonlinear(
            u_theta: Callable[[Float[Array, "d"]], Float[Array, ""]],
            g: Callable[[Float[Array, "d"]], PyTree]
       ) -> Callable[[Float[Array, "d"]], PyTree]:
    """
    Trafo for the u_theta dependent inner product coming from
    a(u_theta; v, w) = \int 3 * u_theta^2 v w dx 

    """
    def g_unravel(x):
        g_flat, unravel = ravel_pytree(g(x))
        nonlinear_flat = jnp.sqrt(3.) * u_theta(x) * g_flat
        assert jnp.shape(g_flat) == jnp.shape(nonlinear_flat)
        return unravel(nonlinear_flat)

    return g_unravel    

def model_wave_eq_factory(prop_speed=1.):
    @jaxtyped
    @typechecker
    def model_wave_eq(
                u_theta: Callable[[Float[Array, "2"]], Float[Array, ""]],
                g: Callable[[Float[Array, "2"]], PyTree]
           ) -> Callable[[Float[Array, "2"]], PyTree]:
        """
        Wave Eq for function of signature (2,) ---> Pytree

        Intended to use when defining inner products on model space.

        """
        
        ddg_1 = model_del_0(u_theta, (model_del_0(u_theta, g)))
        ddg_2 = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def return_wave_eq(x: Float[Array, "2"]) -> PyTree:
            flat_ddg_1, unravel = ravel_pytree(ddg_1(x))
            flat_ddg_2, unravel = ravel_pytree(ddg_2(x))
            return unravel(flat_ddg_1 - prop_speed * flat_ddg_2)#4.
        
        return return_wave_eq
    
    return model_wave_eq