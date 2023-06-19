import jax.numpy as jnp
from jax import grad
import jax.flatten_util

from typing import Callable
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker

# model must map: [(*, *), ...], (d,) ---> ()
# returns: [(*, *), ...], [(*, *), ...] ---> (function: (d,) ---> ())
@jaxtyped
@typechecker
def pushforward_factory(
            model: Callable[
                [PyTree, Float[Array, "d"]], 
                Float[Array, ""]
                ]
       ) -> Callable[
                [PyTree, PyTree], 
                Callable[[Float[Array, "d"]], Float[Array, ""] ]
                ]:
    
    # maps: [(*, *), ...], (d,) ---> [(*, *), ...] 
    @jaxtyped
    @typechecker
    def del_theta_model(
                params: PyTree,
                x: Float[Array, "d"],
           ) -> PyTree:
        return grad(model)(params, x)

    # maps: [(*, *), ...], [(*, *), ...], (d,) ---> ()
    @jaxtyped
    @typechecker
    def pushforward_eval(
                params: PyTree, 
                tangent_params: PyTree,
                x: Float[Array, "d"],
           ) -> Float[Array, ""]:

        del_m_x = del_theta_model(params, x)
        
        return jnp.dot(
            jax.flatten_util.ravel_pytree(del_m_x)[0],
            jax.flatten_util.ravel_pytree(tangent_params)[0],
            )

    # maps: [(*, *), ...], [(*, *), ...] ---> (function: (d,) ---> ())
    @jaxtyped
    @typechecker
    def pushforward(
                params: PyTree, 
                tangent_params: PyTree,
           ) -> Callable[
                    [Float[Array, "d"]],
                    Float[Array, ""],
                    ]:
        return lambda x: pushforward_eval(params, tangent_params, x)
    
    return pushforward