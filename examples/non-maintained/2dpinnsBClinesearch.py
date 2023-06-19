# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import jax.flatten_util

import natgrad.mlp as mlp
from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import laplace
from natgrad.inner.any_d import model_identity, model_laplace
from natgrad.gram import gram_factory, nat_grad_factory

from jax.scipy.optimize import minimize
from scipy.optimize import line_search

jax.config.update("jax_enable_x64", True)

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 80)
boundary_integrator = DeterministicIntegrator(boundary, 80)
eval_integrator = DeterministicIntegrator(interior, 600)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 32, 1]
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(xy):
    x = xy[0]
    y = xy[1]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

# rhs
@jit
def f(xy):
    return 2. * jnp.pi**2 * u_star(xy)

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
    return gram_laplace(params) + gram_bdry(params) #+ 0.0000001 * jnp.identity(len(jax.flatten_util.ravel_pytree(params)[0]))

# natural gradient
nat_grad = nat_grad_factory(gram)

# loss
@jit
def interior_loss(params):
    laplace_model = laplace(lambda x: model(params, x))
    integrand = lambda x: (laplace_model(x) + f(x))**2
    return interior_integrator(vmap(integrand, (0)))

@jit
def boundary_loss(params):
    boundary_integrand = lambda x: model(params, x)**2
    return boundary_integrator(vmap(boundary_integrand, (0)))

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

# line search
@jit
def grid_line_search(params, tangent_params):
    # grid of points [0.985**0, ..., 0.985**3000]
    grid = jnp.linspace(0, 3000, 3001)
    steps = 0.985**grid

    def loss_at_step(step):
        updated_params = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]
        return loss(updated_params)
    
    v_loss_at_step = vmap(loss_at_step)
    losses = v_loss_at_step(steps)
    step_size = steps[jnp.argmin(losses)]
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, tangent_params)], step_size

def optimize_line_search(params, tangent_params, initial_guess):
    def func(alpha):
        alpha_params = [(w - alpha * dw, b - alpha * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]
        return loss(alpha_params)
    
    return line_search(func, grad(func), initial_guess, jnp.array([0.001]))

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))

def L2Norm(func):
    return eval_integrator(lambda x: (func(x))**2)**0.5

def H1Norm(func):
    gradient_abs = lambda x: jnp.dot(grad(func)(x), grad(func)(x))**0.5
    return L2Norm(vmap(func, (0))) + L2Norm(vmap(gradient_abs, (0)))

# random seed
seed = 0
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))

for iteration in range(0):
    grads = grad(loss)(params)
    params, actual_step = grid_line_search(params, grads)
    print(
        f'Iteration: {iteration} with loss: {loss(params)} '
        f'with error L2: {L2Norm(v_error)} and error H1: '
        f'{H1Norm(error)} and step: {actual_step}'
    )

for epoch in range(1):
    for iteration in range(0):
        grads = grad(loss)(params)
        params, actual_step = grid_line_search(params, grads)

        print(
            f'GD Epoch: {epoch} Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {L2Norm(v_error)} and '
            f'error H1: {H1Norm(error)} and step: {actual_step}'
        )

    # natural gradient descent with line search
    for iteration in range(100):
        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)

        flat_natgrad, unravel = jax.flatten_util.ravel_pytree(nat_grads)
        # get some infor about our friend the natgrad
        
        
        normed_natgrad = unravel(1./jnp.amax(jnp.abs(flat_natgrad)) * flat_natgrad)

        params, actual_step = grid_line_search(params, normed_natgrad)
        #step = optimize_line_search(params, normed_natgrad, jnp.reshape(actual_step, (1,)))[0]
        #params = [(w - actual_step * dw, b - actual_step * db)
        #    for (w, b), (dw, db) in zip(params, normed_natgrad)]

        # errors
        print(
            f'ND Epoch: {epoch} Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {L2Norm(v_error)} and '
            f'error H1: {H1Norm(error)} and step: {actual_step}'
        )

    for iteration in range(100):
        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)

        params, actual_step = grid_line_search(params, nat_grads)
        #step = optimize_line_search(params, normed_natgrad, jnp.reshape(actual_step, (1,)))[0]
        #params = [(w - actual_step * dw, b - actual_step * db)
        #    for (w, b), (dw, db) in zip(params, normed_natgrad)]

        # errors
        print(
            f'NG Epoch: {epoch} Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {L2Norm(v_error)} and '
            f'error H1: {H1Norm(error)} and step: {actual_step}'
        )

from matplotlib import pyplot as plt
from natgrad.pushforward import pushforward_factory

push = pushforward_factory(model)
v_push = lambda params, tangent_params: vmap(push(params, tangent_params), (0))
v_u_star = vmap(u_star, (0))

grads = grad(loss)(params)
nat_grads = nat_grad(params, grads)
grads_pushed = v_push(params, grads)
natgrads_pushed = v_push(params, nat_grads)

x = interior.deterministic_integration_points(160)




# plot stuff
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
#plt.suptitle("Different Gradients for the Poisson Equation")

#new_fgrad = (1./jnp.amax(jnp.abs(v_model(params, x) - v_u_star(x)))) * (v_model(params, x) - v_u_star(x)).at[0].set(1.)
new_fgrad = (1./jnp.amax(jnp.abs(v_model(params, x) - v_u_star(x)))) * (v_model(params, x) - v_u_star(x)).at[1].set(-1.)
ax1.scatter(x[:,0], x[:,1], c = new_fgrad, s = 10)
ax1.set_aspect(1.)

#new_natgrad = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x).at[0].set(0.)
new_natgrad = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x)
#ax2.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x), s = 10)
ax2.scatter(x[:,0], x[:,1], c = new_natgrad, s = 10)
ax2.set_aspect(1.)

new_grad = (1./jnp.amax(jnp.abs(grads_pushed(x)))) * grads_pushed(x).at[0].set(0.)
#new_grad = (1./jnp.amax(jnp.abs(grads_pushed(x)))) * grads_pushed(x).at[1].set(-1.)
sc = ax3.scatter(x[:,0], x[:,1], c = new_grad, s = 10)
ax3.set_aspect(1.)

#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sc, cax=cbar_ax)

plt.savefig(
    'out/2dpinnsBClinesearch/pushes.png', 
    bbox_inches="tight",
    dpi=400,
    )