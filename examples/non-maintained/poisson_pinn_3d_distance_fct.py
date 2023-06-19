"""
3d Pinn for Poisson equation on cube. Boundary values with distance function.

The solution is 
u(x,y,z) = sin(pi x) * sin(pi y) * sin(pi z)

on the domain [0,1]^3 with zero Dirichlet boundary values. The boundary values
are enforced using a smooth distance function.

"""

# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, hessian

import natgrad.mlp as mlp
from natgrad.domains import Cube
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivative.three_d import laplace
from natgrad.inner.any_d import model_laplace
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# domains
interior = Cube(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 15)
eval_integrator = DeterministicIntegrator(interior, 40)

# model -- BC are hard enforced
activation = lambda x : jnp.tanh(x)
layer_sizes = [3, 32, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
dist = lambda x: interior.distance_function(x)
model = lambda params, x: mlp.mlp(activation)(params, x) * dist(x)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)

# rhs
@jit
def f(xyz):
    return 3. * jnp.pi**2 * u_star(xyz)

# gramians
gram_laplace = gram_factory(
    model = model,
    trafo = model_laplace,
    integrator = interior_integrator
)

@jit
def gram(params):
    return gram_laplace(params)

# natural gradient
nat_grad = nat_grad_factory(gram)

# loss
@jit
def interior_loss(params):
    laplace_model = laplace(lambda x: model(params, x))
    integrand = lambda x: (laplace_model(x) + f(x))**2
    return interior_integrator(vmap(integrand, (0)))

@jit
def loss(params):
    return interior_loss(params)

# set up grid line search
grid = jnp.linspace(0, 3000, 3001)
steps = 0.985**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5    

# training loop
for iteration in range(151):
    grads = grad(loss)(params)
    params, actual_step = ls_update(params, grads)
    
    if iteration % 10 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    
        print(
            f'GD Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )
    
for epoch in range(1):
    
    # standard gradient descent with line search
    for iteration in range(0):
        grads = grad(loss)(params)
        params, actual_step = ls_update(params, grads)

        if iteration % 25 == 0:
            # errors
            l2_error = l2_norm(v_error, eval_integrator)
            h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
        
            print(
                f'GD Iteration: {iteration} with loss: {loss(params)} with error '
                f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
            )
    

    # natural gradient descent with line search
    for iteration in range(500):
        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)
        params, actual_step = ls_update(params, nat_grads)
        
        if iteration % 1 == 0:
            # errors
            l2_error = l2_norm(v_error, eval_integrator)
            h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
        
            print(
                f'NG Iteration: {iteration} with loss: {loss(params)} with error '
                f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
            )