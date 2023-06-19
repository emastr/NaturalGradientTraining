# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import jax.flatten_util
from matplotlib import pyplot as plt

import natgrad.mlp as mlp
from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator, EvolutionaryIntegrator
from natgrad.derivatives import del_i
from natgrad.utility import model_wave_eq_factory, grid_line_search_factory
from natgrad.inner import model_identity, model_del_i_factory
from natgrad.gram import gram_factory, nat_grad_factory

jax.config.update("jax_enable_x64", True)

seed = 0

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, 40)
interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(seed), N=1500)
initial_integrator = DeterministicIntegrator(initial, 40)
rboundary_integrator = DeterministicIntegrator(rboundary, 40)
lboundary_integrator = DeterministicIntegrator(lboundary, 40)
eval_integrator = DeterministicIntegrator(interior, 100)

#model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 128, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# initial condition
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)
v_u_0 = vmap(u_0, (0))

# solution
def u_star(tx):
    t = tx[0]
    x = tx[1]
    A = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * x)
    B = 0.5 * jnp.sin(4 * jnp.pi * x) * jnp.cos(8 * jnp.pi * t)
    return A + B

# assembling gramians
gram_l_boundary = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = lboundary_integrator
)

gram_r_boundary = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = rboundary_integrator
)

gram_initial = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = initial_integrator
)

model_del_0 = model_del_i_factory()
gram_initial_derivative = gram_factory(
    model = model,
    trafo = lambda u_theta, g: model_del_0(u_theta, g),
    integrator = initial_integrator
)

model_wave_eq = model_wave_eq_factory(prop_speed=4.)
gram_wave = gram_factory(
    model = model,
    trafo = model_wave_eq,
    integrator = interior_integrator
)

# the full inner product
@jit
def gram(params):
    return (
    gram_l_boundary(params) + 
    gram_r_boundary(params) + 
    gram_initial(params) + 
    gram_initial_derivative(params) + 
    gram_wave(params)
    )

nat_grad = nat_grad_factory(gram)

# differential operators
ddt = lambda g: del_i(del_i(g, 0), 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def wave_operator(u):
    return lambda tx: ddt(u)(tx) - 4. * ddx(u)(tx)

# trick to get the signature (params, v_x) -> v_residual
_residual = lambda params: wave_operator(lambda x: model(params, x))
residual = lambda params, x: (_residual(params)(x))**2
v_residual =  jit(vmap(residual, (None, 0)))

# loss terms
@jit
def loss_interior(params):
    return interior_integrator(lambda x: v_residual(params, x))

@jit
def loss_boundary(params):
    return (
        lboundary_integrator(lambda tx: v_model(params, tx)**2) 
            + rboundary_integrator(lambda tx: v_model(params, tx)**2))

@jit
def loss_initial(params):
    return initial_integrator(
        lambda tx: (v_u_0(tx) - v_model(params, tx))**2)

@jit
def loss_initial_derivative(params):
    dt_model = del_i(lambda tx: model(params, tx), 0)
    v_dt_model = vmap(dt_model, (0))
    return initial_integrator(lambda tx: v_dt_model(tx)**2)

@jit
def loss(params):
    return (
        loss_interior(params) + 
        loss_boundary(params) + 
        loss_initial(params) +
        loss_initial_derivative(params)
    )    

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

tx = interior.deterministic_integration_points(250)

# training loop
for iteration in range(50000):
    grads = grad(loss)(params)
    #grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, grads)
    
    if iteration % 50 == 0:
        interior_integrator.update(lambda tx: v_residual(params, tx))

    if iteration % 100 == 0:
        x_1 = interior_integrator._x
        plt.scatter(x_1[:,0], x_1[:,1], s = 10)
        plt.savefig('out/waveeq/points_evo_wave.png')
        plt.clf()

        sc = plt.scatter(tx[:,0], tx[:,1], c = v_model(params, tx), s = 20)
        plt.colorbar(sc)
        plt.savefig('out/waveeq/model_preds.png')
        plt.clf()
    
    l2_error = l2_norm(v_error, eval_integrator)
    #h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    if iteration % 50 == 0:
        print(
            f'Seed: {seed} NGD Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {l2_error} and error H1: '
            #f'{h1_error} and step: {actual_step}'
        )
