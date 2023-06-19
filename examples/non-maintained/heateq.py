# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt

import natgrad.mlp as mlp
from natgrad.pushforward import pushforward_factory
from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import del_i, model_heat_eq
from natgrad.inner.any_d import model_identity
from natgrad.gram import gram_factory, nat_grad_factory

from typeguard import typechecked

jax.config.update("jax_enable_x64", True)

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, 40)
initial_integrator = DeterministicIntegrator(initial, 40)
rboundary_integrator = DeterministicIntegrator(rboundary, 40)
lboundary_integrator = DeterministicIntegrator(lboundary, 40)
eval_integrator = DeterministicIntegrator(interior, 400)

def weight_initial(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)

def weight_boundary(tx):
    t = tx[0]
    x = tx[1]
    return x * (x - 1.) * t * (t - 2.)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 64, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(5))

model_ = mlp.mlp(activation)
model = lambda params, tx: model_(params, tx)# * weight_boundary(tx)# + weight_initial(tx)
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

# linesearch
@jit
def grid_line_search(params, grads):
    grid = jnp.linspace(0, 3000, 3001)
    steps = 0.985**grid

    #grid = jnp.linspace(0, 30, 31)
    #steps = 0.5**grid

    def loss_at_step(step):
        updated_params = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, grads)]
        return loss(updated_params)
    
    v_loss_at_step = vmap(loss_at_step)

    losses = v_loss_at_step(steps)
    step_size = steps[jnp.argmin(losses)]
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)], step_size


# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def heat_operator(u):
    return lambda tx: (dt(u)(tx) - 0.25 * ddx(u)(tx))**2

# loss terms
@jit
def loss_interior(params):
    heat_model = heat_operator(lambda tx: model(params, tx))
    return interior_integrator(vmap(heat_model, (0)))
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

# gradient update
@jit
def update(params, tangent_params, step_size):
    
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]

error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))

def L2Norm(func):
    return eval_integrator(lambda x: (func(x))**2)**0.5

losses_grads = []
losses_natgrads = []
# training loop
for epoch in range(1):
    step = 0.01
    for iteration in range(0):
        grads = grad(loss)(params)
        params, actual_step = grid_line_search(params, grads)
        #params = update(params, grads, step)
        
        error = L2Norm(v_error)
        print(f'Epoch: {epoch} GD: Iteration: {iteration} with loss: {loss(params)} with step: {actual_step} with error: {error}')
        losses_grads.append(loss(params))
        losses_natgrads.append(None)
        
    # natural gradient descent with line search
    for iteration in range(3000):
        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)
        
        #params = line_search(loss, params, nat_grads)
        params, actual_step = grid_line_search(params, nat_grads)

        error = L2Norm(v_error)
        print(f'Epoch {epoch} NG: Iteration: {iteration} with loss: {loss(params)} with step: {actual_step} with error: {error}')
        losses_grads.append(None)
        losses_natgrads.append(loss(params))
        

plt.plot(losses_grads)
plt.plot(losses_natgrads)
plt.savefig('out/heateq/losses.png')
plt.clf()

#------------------------grafical output--------------------------#
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