import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt
import numpy as np

from jax.numpy.linalg import lstsq
from jax.scipy.sparse.linalg import cg

import natgrad.mlp as mlp
from natgrad.domains import Interval
from natgrad.domains import PointBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import laplace
from natgrad.inner import model_laplace, model_identity
from natgrad.gram import gram_factory, nat_grad_factory, nat_grad_factory_generic
from natgrad.utility import grid_line_search_factory
from natgrad.plotting import poission_1d_plot

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# random seed and parameters
seed       = 2
a          = 0.
b          = 1.
omega      = 4.
eps        = 1e-4
iterations = 250
plot       = True

conj_grad=lambda A, b: cg(A, b, maxiter=100)[0]
least_sqs=lambda A, b: lstsq(A, b, rcond=1e-10)[0]

# domains
interior = Interval(a, b)
boundary = PointBoundary(((a, b)))

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 30)
eval_integrator = DeterministicIntegrator(interior, 300)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [1, 16,  1]
params_0 = mlp.init_params(layer_sizes, random.PRNGKey(seed))
params_1 = params_0.copy()
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    x = x[0]
    return jnp.sin(omega*jnp.pi*x)

# rhs
@jit
def f(x):
    x = x[0]
    return jnp.pi**2 *omega**2* jnp.sin(omega*jnp.pi*x) 

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
def gram(params, eps=0):
    gram_ = gram_laplace(params) + gram_bdry(params)
    #grad_, _ =  jax.flatten_util.ravel_pytree(grad(loss)(params))
    #grad_norm_ = jnp.linalg.norm(grad_)
    #return gram_ + eps * jnp.eye(gram_.shape[0]) * grad_norm_ ** 0.5
    return gram_ + eps * jnp.eye(gram_.shape[0]) 

# natural gradient for each optimizer
nat_grad_ls = nat_grad_factory_generic(gram,  least_sqs)
nat_grad_cg = nat_grad_factory_generic(gram,  conj_grad, eps=eps)

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
    return interior_loss(params) + boundary_loss(params)[0]

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors: 0 -> least sqs, 1 -> conj grad
error_0 = lambda x: model(params_0, x) - u_star(x)
error_1 = lambda x: model(params_1, x) - u_star(x)

v_error_0 = vmap(error_0, (0))
v_error_1 = vmap(error_1, (0))
v_error_abs_grad_0 = vmap(lambda x: jnp.dot(grad(error_0)(x), grad(error_0)(x))**0.5)
v_error_abs_grad_1 = vmap(lambda x: jnp.dot(grad(error_1)(x), grad(error_1)(x))**0.5)

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5    
   
# natural gradient descent with line search

# many containers
losses_0 = np.zeros(iterations)
losses_1 = np.zeros(iterations)
l2_errors_0 = np.zeros(iterations)
l2_errors_1 = np.zeros(iterations)
h1_errors_0 = np.zeros(iterations)
h1_errors_1 = np.zeros(iterations)

for iteration in range(iterations):
    #graidents
    grads_0 = grad(loss)(params_0)
    grads_1 = grad(loss)(params_1)

    # natural gradients
    nat_grads_ls = nat_grad_ls(params_0, grads_0)
    nat_grads_cg = nat_grad_cg(params_1, grads_1)

    # update
    params_ls = params_0
    params_cg = params_1
    params_0, actual_step_0 = ls_update(params_0, nat_grads_ls)
    params_1, actual_step_1 = ls_update(params_1, nat_grads_cg)

    # record losses and errors
    losses_0[iteration] = loss(params_0)
    losses_1[iteration] = loss(params_1)
    l2_errors_0[iteration] = l2_norm(v_error_0, eval_integrator)
    l2_errors_1[iteration] = l2_norm(v_error_1, eval_integrator)
    h1_errors_0[iteration] = l2_errors_0[iteration] + l2_norm(v_error_abs_grad_0, eval_integrator) 
    h1_errors_1[iteration] = l2_errors_1[iteration] + l2_norm(v_error_abs_grad_1, eval_integrator)

    if iteration % 50 == 0:
        gram_cg = gram(params_0, eps=eps)
        gram_ls = gram(params_1)

        cg_data = {"gram": gram_cg,
                   "params":params_cg, 
                   "update": actual_step_0,
                   "param_new": params_0,
                   "loss": losses_0[iteration], 
                   "natgrad": nat_grads_cg,
                   "error": l2_errors_0[iteration],
                }
        
        ls_data = {"gram": gram_ls,
                     "params":params_ls, 
                     "update": actual_step_1,
                     "param_new": params_1,
                     "loss": losses_1[iteration], 
                     "natgrad": nat_grads_ls,
                     "error": l2_errors_1[iteration],
                 }
        
        data = {"cg": cg_data, "ls": ls_data}

        jnp.save(f'/Users/mauriciodiaz.ortiz/Documents/Radboud_Phd/NaturalGradients/NaturalGradientTraining/data/poisson_1D/poisson_pinn_1d_{iteration}.npy', data)

        print(f'iteration: {iteration}')
        print('-'*20)
        print(
            f'least sqs | loss: {loss(params_0)}, L2: {l2_errors_0[iteration]},  H1: {h1_errors_0[iteration]}, step: {actual_step_0}'
        )

        print(
            f'conj grad | loss: {loss(params_1)}, L2: {l2_errors_1[iteration]}, H1: {h1_errors_1[iteration]}, step: {actual_step_1}'
        )
        print('-'*20)

if plot:
    poission_1d_plot(model, params_0, params_1, l2_errors_0, h1_errors_0, l2_errors_1, 
                     h1_errors_1, losses_0, losses_1, omega, a, b)

