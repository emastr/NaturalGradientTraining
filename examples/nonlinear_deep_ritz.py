"""
-u'' + u^3 = pi^2 cos(pi x) + cos^3(pi x)    on (-1, 1)

with pure Neumann BC and solution u_star(x) = cos(pi x)

Implemented with Deep Ritz Approach.

"""


import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from natgrad.domains import Interval
import natgrad.mlp as mlp
from natgrad.integrators import TrapezoidalIntegrator
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory, model_nonlinear
from natgrad.inner import model_del_i_factory

jax.config.update("jax_enable_x64", True)

seed = 0

# integration
interval = Interval(-1., 1.)
integrator = TrapezoidalIntegrator(interval, 18000)
eval_integrator = TrapezoidalIntegrator(interval, 80000)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [1, 16, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation) 
v_model = vmap(model, (None, 0))

# solution and right-hand side
u_star = lambda x: jnp.reshape(jnp.cos(jnp.pi * x), ())
f = lambda x: (jnp.pi**2) * u_star(x) + u_star(x)**3

# gramians
model_derivative = model_del_i_factory()
gram_grad = gram_factory(
    model = model,
    trafo = model_derivative,
    integrator = integrator,
)

gram_nonlinear = gram_factory(
    model = model,
    trafo = model_nonlinear,
    integrator = integrator,
)

@jit
def gram(params):
    return gram_grad(params) + gram_nonlinear(params)

nat_grad = nat_grad_factory(gram)

# loss function
grad_model = vmap(grad(lambda params, x: model(params, x), 1), (None, 0))

@jit
def loss_gradient(params):
    grad_squared = lambda x: 0.5 * jnp.reshape(grad_model(params, x)**2, (len(x)))
    return integrator(grad_squared)

@jit
def loss_lower_order_term(params):
    model_to_four = lambda x: 0.25 * v_model(params, x)**4
    return integrator(model_to_four)

@jit
def loss_rhs(params):
    rhs = lambda x: vmap(f, (0))(x) * v_model(params, x)
    return integrator(rhs)

@jit 
def loss(params):
    return loss_gradient(params) + loss_lower_order_term(params) - loss_rhs(params)

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

for iteration in range(1000):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)
    
    if iteration % 10 == 0:
        # errors
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    
        print(
            f'NG Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
        )