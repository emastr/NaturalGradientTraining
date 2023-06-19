# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt

import natgrad.mlp as mlp
from natgrad.pushforward import pushforward_factory
from natgrad.domains import Square, SquareBoundary
from natgrad.integrators import DeterministicIntegrator, EvolutionaryIntegrator
from natgrad.derivatives import del_i
from natgrad.inner import model_identity, model_del_i_factory
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory

# we need double precision for the least squares solve
jax.config.update("jax_enable_x64", True)

# random seed
seed = 5

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(seed), N=500)
initial_integrator = DeterministicIntegrator(initial, 40)
rboundary_integrator = DeterministicIntegrator(rboundary, 40)
lboundary_integrator = DeterministicIntegrator(lboundary, 40)
eval_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(seed), N=5000)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 64, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))

# initial condition
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)
v_u_0 = vmap(u_0, (0))

# solution
def u_star(tx):
    t = tx[0]
    x = tx[1]
    return jnp.exp(-jnp.pi**2 * t * 0.25) * jnp.sin(jnp.pi * x)

# assembling the trafo that determines the energy inner product
model_del_0 = model_del_i_factory(0)
model_del_1 = model_del_i_factory(1)

def model_heat_eq(u_theta, g):
        
        # I don't like this syntax. By default arguments u_theta 
        # should not be needed. Right now this gives me trouble
        # with the typing system. I'll try to fix that.
        dg_1 = model_del_0(u_theta, g)
        ddg_2 = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def return_heat_eq(x):
            flat_dg_1, unravel = ravel_pytree(dg_1(x))
            flat_ddg_2, unravel = ravel_pytree(ddg_2(x))
            return unravel(flat_dg_1 - 0.25 * flat_ddg_2)
        
        return return_heat_eq

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

gram_heat = gram_factory(
    model = model,
    trafo = model_heat_eq,
    integrator = interior_integrator
)

# the full inner product
@jit
def gram(params):
    return (
    gram_l_boundary(params) + 
    gram_r_boundary(params) + 
    gram_initial(params) + 
    gram_heat(params)
    )

# maps: params, tangent_params ---> tangent_params
nat_grad = nat_grad_factory(gram)

# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def heat_operator(u):
    return lambda tx: dt(u)(tx) - 0.25 * ddx(u)(tx)

# trick to get the signature (params, v_x) -> v_residual
_residual = lambda params: heat_operator(lambda x: model(params, x))
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
def loss(params):
    return loss_interior(params) + loss_boundary(params) + loss_initial(params)    

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

# training loop
for iteration in range(1000):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)
    
    if iteration % 10 == 0:
        interior_integrator.update(lambda tx: v_residual(params, tx))
    
    l2_error = l2_norm(v_error, eval_integrator)
    h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
    if iteration % 10 == 0:
        print(
            f'Seed: {seed} NGD Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {l2_error} and error H1: '
            f'{h1_error} and step: {actual_step}'
        )
    
    
#------------------------grafical output--------------------------#

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), sharex=False, sharey=False)
#ax1.scatter(x_1[:,0], x_1[:,1], s = 10)
#ax1.set_aspect(1.)

x_1 = interior_integrator._x
plt.scatter(x_1[:,0], x_1[:,1], s = 10)
plt.savefig('out/heateq/points_evo_heat.png')
plt.clf()

v_u_star = vmap(u_star, (0))
push = pushforward_factory(model)
v_push = lambda params, tangent_params: vmap(push(params, tangent_params), (0))
tx = interior.deterministic_integration_points(250)

grads = grad(loss)(params)
nat_grads = nat_grad(params, grads)
grads_pushed = v_push(params, grads)
natgrads_pushed = v_push(params, nat_grads)

sc = plt.scatter(tx[:,0], tx[:,1], c = (1./jnp.amax(jnp.abs(natgrads_pushed(tx)))) * natgrads_pushed(tx), s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/natgrad_pushed.png')
plt.clf()

sc = plt.scatter(tx[:,0], tx[:,1], c = (1./jnp.amax(jnp.abs(v_model(params, tx) - v_u_star(tx)))) * (v_model(params, tx) - v_u_star(tx)), s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/FspaceGrad.png')
plt.clf()

a = (1./jnp.amax(jnp.abs(natgrads_pushed(tx)))) * natgrads_pushed(tx)
b = (1./jnp.amax(jnp.abs(v_model(params, tx) - v_u_star(tx)))) * (v_model(params, tx) - v_u_star(tx))
sc = plt.scatter(tx[:,0], tx[:,1], c = a-b, s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/FspaceGradMinusNat.png')
plt.clf()

sc = plt.scatter(tx[:,0], tx[:,1], c = v_u_star(tx), s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/solution.png')
plt.clf()

sc = plt.scatter(tx[:,0], tx[:,1], c = v_model(params, tx), s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/model_preds.png')
plt.clf()

sc = plt.scatter(tx[:,0], tx[:,1], c = grads_pushed(tx), s = 20)
plt.colorbar(sc)
plt.savefig('out/heateq/grad_pushed.png')