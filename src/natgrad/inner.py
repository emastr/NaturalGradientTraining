"""
Contains implementation of inner products via specifying the transformation.

"""
import jax.flatten_util
import jax.numpy as jnp
from jax import hessian, jacfwd

from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker
from typing import Callable

@jaxtyped
@typechecker
def model_identity(
            u_theta: Callable[[Float[Array, "d"]], Float[Array, ""]],
            g: Callable[[Float[Array, "d"]], PyTree]
       ) -> Callable[[Float[Array, "d"]], PyTree]:
    """
    Identity for function of signature (d,) ---> Pytree. Intended to
    use when defining inner products on model space.
    
    Parameters
    ----------
    u_theta: Callable
        for fixed params theta: x -> u(theta, x). The function 
        model_laplace does not depend on this argument!
    g: Callable
        Typically: x -> del_theta u(theta, x)
    
    """
    return g

@jaxtyped
@typechecker
def model_laplace(
            u_theta: Callable[[Float[Array, "d"]], Float[Array, ""]],
            g: Callable[[Float[Array, "d"]], PyTree]
       ) -> Callable[[Float[Array, "d"]], PyTree]:
    """
    Computes the laplacian componentwise of a function that maps
    into parameter space. Typically the only usage for this method is
    to be passed to the gramian as the trafo argument.

    Parameters
    ----------
    u_theta: Callable
        for fixed params theta: x -> u(theta, x). The function 
        model_laplace does not depend on this argument!
    g: Callable
        Typically: x -> del_theta u(theta, x)
    
    """
    def g_ravel(x: Float[Array, "d"]) -> Float[Array, "d_param"]:
        return jax.flatten_util.ravel_pytree(g(x))[0]

    def g_laplace(x: Float[Array, "d"]) -> PyTree:
        unravel = jax.flatten_util.ravel_pytree(g(x))[1]
        return unravel(jnp.trace(hessian(g_ravel)(x), axis1=1, axis2=2))

    return g_laplace

@jaxtyped
@typechecker
def model_del_i_factory(
            argnum: int = 0,
       ) -> Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ]:
    """
        Partial derivative for a function of signature (d,) ---> PyTree
        Intended to use when defining inner products on model space.
        
    """
    @jaxtyped
    @typechecker
    def model_del_i(
                u_theta: PyTree,
                g: Callable[[Float[Array, "d"]], PyTree],
           ) -> Callable[[Float[Array, "d"]], PyTree]:
        
        @typechecker
        def g_splitvar(*args) -> PyTree:
            x_ = jnp.array(args)
            return g(x_)

        d_splitvar_di = jacfwd(g_splitvar, argnum)

        @typechecker
        def dg_di(x: Float[Array, "d"]) -> PyTree:
            return d_splitvar_di(*x)

        return dg_di

    return model_del_i