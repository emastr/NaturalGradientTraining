# Inverse problem, source recovery -- PDE constrained optimization
#
# The concrete example is:
#
# min E(u,f) = ||laplace(u) + f||^2 + ||u_d - u||^2 + ||u - g||^2_Gamma

import jax.numpy as jnp
from jax import grad, vmap, jit, random
import jax.flatten_util
from jax.numpy.linalg import lstsq
import jax
from matplotlib import pyplot as plt

import natgrad.mlp as mlp
from natgrad.domains import Hyperrectangle, Interval, RectangleBoundary, Square
from natgrad.integrators import EvolutionaryIntegrator
from natgrad.derivatives import laplace
from natgrad.inner import model_identity, model_laplace
from natgrad.gram import gram_factory
from natgrad.utility import grid_line_search_factory

from utility import IntervalBoundary, flatten_pytrees 
from utility import grid_line_search_factory, new_gram_factory

jax.config.update("jax_enable_x64", True)

alpha = 1.
beta  = 1.
gamma = 1.

# dimension
dim = 2

# domains
intervals = [(0., 1.) for i in range(0, dim)]
#interior = Hyperrectangle(intervals)
interior = Square(1.)
boundary = RectangleBoundary(intervals)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, random.PRNGKey(0), N=1500)
data_integrator = EvolutionaryIntegrator(interior, random.PRNGKey(0), N=40)
boundary_integrator = EvolutionaryIntegrator(boundary, random.PRNGKey(0), N=200)
eval_integrator = EvolutionaryIntegrator(interior, random.PRNGKey(0), N=5000)

# model_u
activation_u = lambda x : jnp.tanh(x)
layer_sizes_u = [dim, 32, 1]
params_u = mlp.init_params(layer_sizes_u, random.PRNGKey(0))
model_u = mlp.mlp(activation_u)
v_model_u = vmap(model_u, (None, 0))

# model_f, this is the control variable f
activation_f = lambda x : jnp.tanh(x)
layer_sizes_f = [dim, 8, 1]
params_f = mlp.init_params(layer_sizes_f, random.PRNGKey(0))
model_f = mlp.mlp(activation_f)
v_model_f = vmap(model_f, (None, 0))

@jit
def u_star(xy):
    x = xy[0]
    y = xy[1]
    return jnp.reshape(
        jnp.sin(jnp.pi * x) * jnp.exp(-jnp.pi * y),
        (),
    )
v_u_star = vmap(u_star, (0))


# trick to get the signature (params_u, params_v, v_x) -> v_residual
_laplace_model = lambda params_u: laplace(lambda x: model_u(params_u, x))
residual = lambda params_u, params_f, x: (_laplace_model(params_u)(x) + model_f(params_f, x))
v_residual =  jit(vmap(residual, (None, None, 0)))

# PDE loss: 
# 0.5 * || laplace(u_theta) + f_psi ||^2 
@jit
def loss_u_f(params_u, params_f):
    return 0.5 * interior_integrator(
        lambda x: (v_residual(params_u, params_f, x))**2.
    )

@jit
def loss_data(params_u):
    return 0.5 * data_integrator(
        lambda x: (v_model_u(params_u, x) - v_u_star(x))**2.
    )

@jit
def loss_boundary(params_u):
    return 0.5 * boundary_integrator(
        lambda x: (v_model_u(params_u, x) - v_u_star(x))**2.
    )

@jit
def loss(params_u, params_f):
    return loss_u_f(params_u, params_f) + loss_boundary(params_u) + loss_data(params_u)

# gramians
gram_A_1 = jit(gram_factory(
    model = model_u,
    trafo = model_identity,
    integrator = data_integrator
))

gram_A_2 = jit(gram_factory(
    model = model_u,
    trafo = model_laplace,
    integrator = interior_integrator
))

gram_A_3 = jit(gram_factory(
    model = model_u,
    trafo = model_identity,
    integrator = boundary_integrator
))

@jit
def gram_A(params_u):
    return gram_A_1(params_u) + gram_A_2(params_u) + gram_A_3(params_u)

gram_D = jit(gram_factory(
    model = model_f,
    trafo = model_identity,
    integrator = interior_integrator
))

gram_B = jit(new_gram_factory(
    model_1 = model_u,
    model_2 = model_f,
    trafo_1 = model_laplace,
    trafo_2 = model_identity,
    integrator = interior_integrator,
))

@jit
def gram(params_u, params_f):
    A = gram_A(params_u)
    B = gram_B(params_u, params_f)
    C = jnp.transpose(B)
    D = gram_D(params_f)
    col_1 = jnp.concatenate((A,C), axis=0)
    col_2 = jnp.concatenate((B,D), axis=0)
    return jnp.concatenate((col_1, col_2), axis=1)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

for iteration in range(400):
    grad_u = grad(loss, 0)(params_u, params_f)
    grad_f = grad(loss, 1)(params_u, params_f)
    
    gram_matrix = gram(params_u, params_f)

    flat_combined_grad, retrieve_pytrees = flatten_pytrees(grad_u, grad_f)
    long_flat_nat_grad = lstsq(gram_matrix, flat_combined_grad)[0]
    nat_grad_u, nat_grad_f = retrieve_pytrees(long_flat_nat_grad)
    
    params_u, params_f, actual_step = ls_update(params_u, params_f, nat_grad_u, nat_grad_f)
    
    if iteration % 50 == 0:
        print(
            f'ENGD Iteration: {iteration} with loss: '
            f'{loss(params_u, params_f)} and step: {actual_step}'
        )

#--------------------plots---------------------------------------------#
x = interior.deterministic_integration_points(160)

# plot stuff
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)

sc1 = ax1.scatter(x[:,0], x[:,1], c = v_model_u(params_u, x), s = 10)
fig.colorbar(sc1, ax=ax1)

sc2 = ax2.scatter(x[:,0], x[:,1], c = v_u_star(x), s = 10)
fig.colorbar(sc2, ax=ax2)

sc3 = ax3.scatter(x[:,0], x[:,1], c = v_model_f(params_f, x), s = 10)
fig.colorbar(sc3, ax=ax3)

sc4 = ax4.scatter(x[:,0], x[:,1], c = v_model_u(params_u, x) - v_u_star(x), s = 10)
#ax4.set_aspect(1.)
fig.colorbar(sc4, ax=ax4)

plt.show()