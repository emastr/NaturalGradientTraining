# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

import natgrad.mlp as mlp
from natgrad.domains import Hyperrectangle
from natgrad.domains import SquareBoundary, Interval
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import laplace
from natgrad.inner import model_laplace, model_identity
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory

jax.config.update("jax_enable_x64", True)

# random seed
seed = 2

# domains
T = 1.
interior = Hyperrectangle(((-1, 1), (0, T)))
boundary_time = Interval(-1., 1.)
boundary_space_1 = Interval(0, T)
boundary_space_2 = Interval(0, T) 


# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integ_time = DeterministicIntegrator(boundary_time, 30)
boundary_integ_space_1 = DeterministicIntegrator(boundary_space_1, 30)
boundary_integ_space_2 = DeterministicIntegrator(boundary_space_2, 30)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 32, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(xt):
    return 0

# rhs
@jit
def f(xt):
    return 0.


def residual(params, x):
    grad_ = grad(model, 1)(params, x)
    

# natural gradient
nat_grad = nat_grad_factory(gram)

# trick to get the signature (params, v_x) -> v_residual
_residual = lambda params: laplace(lambda x: model(params, x))
residual = lambda params, x: (_residual(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

# loss
@jit
def interior_loss(params):
    return interior_integrator(lambda x: v_residual(params, x))

@jit
def boundary_loss(params):
    boundary_integrand = lambda x: model(params, x)**2
    return boundary_integrator(vmap(boundary_integrand, (0)))

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5    
   
# natural gradient descent with line search
for iteration in range(500):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)
    
    if iteration % 50 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    
        print(
            f'NG Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )
