import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

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
from jaxopt import LevenbergMarquardt
from jax.example_libraries import optimizers
from jax.lib import xla_bridge

#xla_bridge.set_default_device

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
matplotlib.rcParams.update({'font.size': 16})

# random seed and parameters
seed       = 2
a          = 0.
b          = 1.
omega      = 3.
eps        = 1e-4
iterations = 1001
repeats    = 10
plot       = True
save       = False
method     = 'ls'

key, subkey = random.split(random.PRNGKey(seed))

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [1, 16, 1]
params = mlp.init_params(layer_sizes, subkey)

model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# domains
interior = Interval(a, b)
boundary = PointBoundary(((a, b)))

# integrators
interior_integrator = DeterministicIntegrator(interior, 50)
boundary_integrator = DeterministicIntegrator(boundary, 50)
eval_integrator = DeterministicIntegrator(interior, 300)

# Functions
@jit
def u_star(x):
    x = x[0]
    return jnp.sin(omega*jnp.pi*x)

@jit
def f(x):
    x = x[0]
    return jnp.pi**2 *omega**2* jnp.sin(omega*jnp.pi*x) 

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5   

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


# Optimizer selection loop
if method == 'cg' or method == 'ls':
    gram_bdry = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = boundary_integrator)

    gram_laplace = gram_factory(
        model = model,
        trafo = model_laplace,
        integrator = interior_integrator)
    
    @jit
    def gram(params, eps=0):
        gram_ = gram_laplace(params) + gram_bdry(params)
        return gram_ + eps * jnp.eye(gram_.shape[0]) 
    
    _residual = lambda params: laplace(lambda x: model(params, x))
    residual = lambda params, x: (_residual(params)(x) + f(x))**2.
    v_residual =  jit(vmap(residual, (None, 0)))

    if method=='cg':
        optim = lambda A, b: cg(A, b, maxiter=50)[0]
    elif method == 'ls':
        optim = lambda A, b: lstsq(A, b, rcond=1e-10)[0]

    nat_grad = nat_grad_factory_generic(gram,  optim, eps=eps)

    _residual = lambda params: laplace(lambda x: model(params, x))
    residual = lambda params, x: (_residual(params)(x) + f(x))**2.
    v_residual =  jit(vmap(residual, (None, 0)))

    grid = jnp.linspace(0, 30, 31)
    steps = 0.5**grid
    ls_update = grid_line_search_factory(loss, steps)

elif method=='lm':
    params, unflatten = jax.flatten_util.ravel_pytree(params)
    res_bdry = lambda params, x: model(params, x)
    v_res_bdry = jit(vmap(res_bdry, (None, 0)))

    def residuals_lm(params):
        params = unflatten(params)
        boundary_res_pts = vmap(res_bdry, (None, 0))(params, boundary_integrator._x) / len(boundary_integrator._x)
        interior_res_pts = vmap(residual, (None, 0))(params, interior_integrator._x) / len(interior_integrator._x)
        return jnp.concatenate([boundary_res_pts, interior_res_pts])
    
    optim = LevenbergMarquardt(residuals_lm, maxiter=1, tol=1e-6, materialize_jac=True)
else:
    raise ValueError('method must be cg, ls, or lm')

error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))

# Training loop
losses = np.zeros(shape=(iterations, repeats))
l2_errors = np.zeros(shape=(iterations, repeats))

for i in tqdm(range(repeats)):
    key, subkey = random.split(random.PRNGKey(seed+i))
    params = mlp.init_params(layer_sizes, subkey)

    for ii in range(iterations):

        if method == 'ls' or method == "cg":
            grads = grad(loss)(params)
            params, actual_step = ls_update(params, grads)
            losses[ii, i] = loss(params)
            l2_errors[ii, i] = l2_norm(v_error, eval_integrator)
        else:
            if ii == 0:
                out = optim.run(params)
            else:
                out = optim.update(out.params, out.state)
            
            losses[ii, i] = loss(out.params)
            l2_errors[ii, i] = l2_norm(v_error, eval_integrator)

np.save(f'/Users/mauriciodiaz.ortiz/Documents/Radboud_Phd/NaturalGradients/NaturalGradientTraining/data/poisson_1D/1D_poisson_{method}_loss_baseline.npy', losses)
np.save(f'/Users/mauriciodiaz.ortiz/Documents/Radboud_Phd/NaturalGradients/NaturalGradientTraining/data/poisson_1D/1D_poisson_{method}_l2_baseline.npy', l2_errors)

