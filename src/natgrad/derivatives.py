import jax.numpy as jnp
from jax import hessian, jacfwd, grad
import jax.flatten_util

from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker
from natgrad.typemacros import MlpParams


# func must map (d,) ---> ()
# nicer would be an api like the grad trafo of jax...
@jaxtyped
@typechecker
def laplace(
            func: Callable[[Float[Array, "d"]], Float[Array, ""]]
       ) -> Callable[[Float[Array, "d"]], Float[Array, ""]]:
    """
    Computes Laplacian via trace of hessian
    
    """
    hesse = hessian(func)
    return lambda x: jnp.sum(jnp.diag(hesse(x)))

@jaxtyped
@typechecker
def del_i(
            g: Callable[[Float[Array, "d"]], Float[Array, ""]],
            argnum: int = 0,
       ) -> Callable[[Float[Array, "d"]], Float[Array, ""]]:
    """
    Partial derivative for a function of signature (d,) ---> ().
    Intended to use when defining PINN loss functions.
    
    """
    @typechecker
    def g_splitvar(*args) -> PyTree:
        x_ = jnp.array(args)
        return g(x_)

    d_splitvar_di = grad(g_splitvar, argnum)

    @typechecker
    def dg_di(x: Float[Array, "d"]) -> Float[Array, ""]:
        return d_splitvar_di(*x)

    return dg_di

