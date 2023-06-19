import jax.numpy as jnp
from jax import grad, vmap
import jax.flatten_util
from jax.numpy.linalg import lstsq

from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker

# maps: (model, trafo) ---> (params, x ---> (param_dim, param_dim))
@jaxtyped
@typechecker
def pre_gram_factory(
            model: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            trafo: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
       ) -> Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, "Pdim Pdim"]
            ]:

    # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
    @jaxtyped
    @typechecker
    def del_theta_model(
                params: PyTree,
                x: Float[Array, "d"],
           ) -> PyTree:
        return grad(model)(params, x)
    
    @jaxtyped
    @typechecker
    def pre_gram(
                params: PyTree, 
                x: Float[Array, "d"]
           ) -> Float[Array, "Pdim Pdim"]:
        
        @jaxtyped
        @typechecker
        def g(y: Float[Array, "d"]) -> PyTree:
            #return trafo(lambda z: del_theta_model(params, z))(y)
            return trafo(
                lambda z: model(params, z),
                lambda z: del_theta_model(params, z),
            )(y)
        
        flat = jax.flatten_util.ravel_pytree(g(x))[0]
        flat_col = jnp.reshape(flat, (len(flat), 1))
        flat_row = jnp.reshape(flat, (1, len(flat)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram

@jaxtyped
@typechecker
def gram_factory(
            model: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            trafo: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree]
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
            integrator: Callable,
       ) -> Callable[
                [PyTree],
                Float[Array, "Pdim Pdim"]
            ]:

    pre_gram = pre_gram_factory(model, trafo)
    v_pre_gram = vmap(pre_gram, (None, 0))
    
    @jaxtyped
    @typechecker
    def gram(params: PyTree) -> Float[Array, "Pdim Pdim"]:
        gram_matrix = integrator(lambda x: v_pre_gram(params, x))
        return gram_matrix
    
    return gram

@jaxtyped
@typechecker
def nat_grad_factory(
            gram: Callable[
                [PyTree],
                Float[Array, "Pdim Pdim"]
            ]
       ) -> Callable[
                [PyTree, PyTree],
                PyTree
            ]:

    # maps: [(*,*), ..., (*,*)], [(*,*), ..., (*,*)] ---> [(*,*), ..., (*,*)]
    def natural_gradient(
                params: PyTree, 
                tangent_params: PyTree
           ) -> PyTree:

        gram_matrix = gram(params)
        flat_tangent, retriev_pytree  = jax.flatten_util.ravel_pytree(tangent_params)
        
        # solve gram dot flat_tangent.
        flat_nat_grad = lstsq(gram_matrix, flat_tangent)[0]
        return retriev_pytree(flat_nat_grad)

    return natural_gradient