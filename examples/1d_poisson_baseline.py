import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt
import matplotlib
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
iterations = 50001
repeats    = 10
plot       = True
save       = False

key, subkey = random.split(random.PRNGKey(seed))

conj_grad=lambda A, b: cg(A, b, maxiter=50)[0]
least_sqs=lambda A, b: lstsq(A, b, rcond=1e-10)[0]

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


# domains
interior = Interval(a, b)
boundary = PointBoundary(((a, b)))

# integrators
interior_integrator = DeterministicIntegrator(interior, 50)
boundary_integrator = DeterministicIntegrator(boundary, 50)
eval_integrator = DeterministicIntegrator(interior, 300)

activation = lambda x : jnp.tanh(x)
layer_sizes = [1, 16, 1]
params_0 = mlp.init_params(layer_sizes, subkey)
params_1 = params_0.copy()

model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

exp_decay = optimizers.exponential_decay(0.01, 5000, 0.7)
opt_init, opt_update, get_params = optimizers.adam(exp_decay)
opt_state = opt_init(params_0)

# things to actually optimize
_residual = lambda params: laplace(lambda x: model(params, x))
residual = lambda params, x: (_residual(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

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

@jit
def step(istep, opt_state):
    param = get_params(opt_state) 
    g = grad(loss, argnums=0)(param)
    return opt_update(istep, g, opt_state)

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5

error_0 = lambda x: model(params_0, x) - u_star(x)
v_error_0 = vmap(error_0, (0))    

losses = np.zeros(shape=(iterations, repeats))
l2_errors = np.zeros(shape=(iterations, repeats))

for repeat in range(repeats):
    print(f'run {repeat}')
    print('-'*20)
    key, subkey = random.split(random.PRNGKey(seed + repeat))
    params_0 = mlp.init_params(layer_sizes, subkey)
    opt_state = opt_init(params_0)

    for it in range(iterations):
        opt_state = step(it, opt_state)
        params_0 = get_params(opt_state)
        losses[it][repeat] = loss(get_params(opt_state))
        l2_errors[it][repeat] = l2_norm(v_error_0, eval_integrator)

        if it%10000 == 0:
            print(f'iteration {it}, loss {losses[it][repeat]}, l2 error {l2_errors[it][repeat]}')

    print('-'*20)

if save:
    np.save('/Users/mauriciodiaz.ortiz/Documents/Radboud_Phd/NaturalGradients/NaturalGradientTraining/data/poisson_1D/1D_poisson_baseline_losses.npy', losses)
    np.save('/Users/mauriciodiaz.ortiz/Documents/Radboud_Phd/NaturalGradients/NaturalGradientTraining/data/poisson_1D/1D_poisson_baseline_l2_errors.npy', l2_errors)

its = np.arange(iterations)

plt.figure(figsize=(7, 6), dpi=100)
plt.plot(its, losses.mean(axis=1))
plt.fill_between(its, (losses.mean(axis=1)-losses.std(axis=1)), (losses.mean(axis=1)+losses.std(axis=1)), alpha=0.5)
plt.xlabel(r'$Iterations$')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.show()

plt.figure(figsize=(7, 6), dpi=100)
plt.plot(its, l2_errors.mean(axis=1))
plt.fill_between(its, l2_errors.mean(axis=1)-l2_errors.std(axis=1), l2_errors.mean(axis=1)+l2_errors.std(axis=1), alpha=0.5)
plt.xlabel(r'$Iterations$')
plt.ylabel(r'$L_2$ error')
#plt.yscale('log')
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.show()
