import jax.numpy as jnp
from jax import grad, vmap
import jax.flatten_util
from jax.numpy.linalg import lstsq

from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker
from natgrad.types import InputType, MatrixType, ModelType, TrafoType, GramFuncType, GramEvalType, ParamType
from natgrad.types import FloatArrayN1, FloatArrayN2, FloatArrayN3, FloatArrayNd, PointType
from natgrad.linalg import sherman_morrison
                     
VecType = Float[Array, "1 N"]
# maps: (model, trafo) ---> (params, x ---> (param_dim, param_dim))
@jaxtyped
@typechecker
def pre_gram_factory(model: ModelType, trafo: TrafoType) -> GramFuncType:
    # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
    @jaxtyped
    @typechecker
    def del_theta_model(params: PyTree, x: InputType,) -> PyTree:
        return grad(model)(params, x)
    
    @jaxtyped
    @typechecker
    def pre_gram(params: PyTree, x: InputType) -> MatrixType:
        
        @jaxtyped
        @typechecker
        def g(y: InputType) -> PyTree:
            #return trafo(lambda z: del_theta_model(params, z))(y)
            return trafo(lambda z: model(params, z), lambda z: del_theta_model(params, z),)(y)
        
        flat = jax.flatten_util.ravel_pytree(g(x))[0]
        flat_col = jnp.reshape(flat, (len(flat), 1))
        flat_row = jnp.reshape(flat, (1, len(flat)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram

    
    
    
@jaxtyped
@typechecker
def gram_factory(model: ModelType, trafo: TrafoType, integrator: Callable,) -> GramEvalType:

    pre_gram = pre_gram_factory(model, trafo)
    v_pre_gram = vmap(pre_gram, (None, 0))
    
    @jaxtyped
    @typechecker
    def gram(params: PyTree) -> MatrixType:
        gram_matrix = integrator(lambda x: v_pre_gram(params, x))
        return gram_matrix
    
    return gram

@jaxtyped
@typechecker
def nat_grad_factory(gram: GramEvalType) -> ParamType:
    # maps: [(*,*), ..., (*,*)], [(*,*), ..., (*,*)] ---> [(*,*), ..., (*,*)]
    def natural_gradient(params: PyTree, tangent_params: PyTree) -> PyTree:
        
        gram_matrix = gram(params)
        flat_tangent, retriev_pytree  = jax.flatten_util.ravel_pytree(tangent_params)
        
        # solve gram dot flat_tangent.
        flat_nat_grad = lstsq(gram_matrix, flat_tangent)[0]
        return retriev_pytree(flat_nat_grad)

    return natural_gradient


@jaxtyped
@typechecker
def nat_grad_factory_generic(gram: GramEvalType, solver, eps=0, **kwargs) -> ParamType:
    # maps: [(*,*), ..., (*,*)], [(*,*), ..., (*,*)] ---> [(*,*), ..., (*,*)]
    def natural_gradient(params: PyTree, tangent_params: PyTree) -> PyTree:
        
        gram_matrix = gram(params, eps)
        flat_tangent, to_pytree  = jax.flatten_util.ravel_pytree(tangent_params)
        
        # solve gram dot flat_tangent.
        flat_nat_grad = solver(gram_matrix, flat_tangent, **kwargs)
        return to_pytree(flat_nat_grad)

    return natural_gradient



# class ShermanGram():
    
#     def __init__(self, init_matrix, decay=0.9):
#     # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
#         self.gram_inv = init_matrix
#         self.del_theta_model = lambda model, params, x: grad(model)(params, x)
#         self.decay = decay


#     def __call__(self, grad):
#         return self.gram_inv @ grad
    
    
#     def update_gram(self, model: ModelType, trafo: TrafoType, params: PyTree, x:InputType):
        
#         g = lambda y: trafo(lambda z: model(params, z), lambda z: self.del_theta_model(model, params, z),)(y)
#         flat = jax.flatten_util.ravel_pytree(g(x))[0]
#         flat_row = jnp.reshape(flat, (1, len(flat)))
#         flat_col = jnp.reshape(flat, (len(flat), 1))
#         self.gram_inv = sherman_morrison(self.gram_inv, self.decay * flat_row, flat_col)