"""
3d Pinn for Poisson equation on cube. Boundary values with penalty.

The solution is 
u(x,y,z) = sin(pi x) * sin(pi y) * sin(pi z)

on the domain [0,1]^3 with zero Dirichlet boundary values. 

"""

# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

import natgrad.mlp as mlp
from natgrad.domains import Cube, CubeBoundary
from natgrad.integrators import DeterministicIntegrator, EvolutionaryIntegrator
from natgrad.derivatives import laplace
from natgrad.inner import model_laplace, model_identity
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# domains
interior = Cube(1.)
boundary = CubeBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 15)
#interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(seed), N=3375)
boundary_integrator = DeterministicIntegrator(boundary, 15)
eval_integrator = DeterministicIntegrator(interior, 40)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [3, 32, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = lambda params, x: mlp.mlp(activation)(params, x)
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
gram_bdry = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = boundary_integrator
)

gram_laplace = gram_factory(
    model = model,
    trafo = model_laplace,
    integrator = interior_integrator
)

@jit
def gram(params):
    return gram_bdry(params) + gram_laplace(params)

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
    
    if iteration % 25 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    
        print(
            f'NG Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )