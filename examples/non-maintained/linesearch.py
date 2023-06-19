# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import jax.flatten_util

import natgrad.mlp as mlp
from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import del_i, model_heat_eq
from natgrad.inner.any_d import model_identity
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.utility import grid_line_search_factory
import jaxopt

jax.config.update("jax_enable_x64", True)

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
initial_integrator = DeterministicIntegrator(initial, 30)
rboundary_integrator = DeterministicIntegrator(rboundary, 30)
lboundary_integrator = DeterministicIntegrator(lboundary, 30)
eval_integrator = DeterministicIntegrator(interior, 300)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 16, 1]

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
def gram_reg(params):
    return (
    gram_l_boundary(params) + 
    gram_r_boundary(params) + 
    gram_initial(params) + 
    gram_heat(params) + 
    0.0000001 * jnp.identity(len(jax.flatten_util.ravel_pytree(params)[0]))
    )

@jit
def gram(params):
    return (
    gram_l_boundary(params) + 
    gram_r_boundary(params) + 
    gram_initial(params) + 
    gram_heat(params)
    )

# maps: params, tangent_params ---> tangent_params
nat_grad_reg = nat_grad_factory(gram_reg)
nat_grad = nat_grad_factory(gram)

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

# linesearch
# set up grid line search
grid = jnp.linspace(0, 3000, 3001)
steps = 0.985**grid
ls_update = grid_line_search_factory(loss, steps)    

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))

def L2Norm(func):
    return eval_integrator(lambda x: (func(x))**2)**0.5

def H1Norm(func):
    gradient_abs = lambda x: jnp.dot(grad(func)(x), grad(func)(x))**0.5
    return L2Norm(vmap(func, (0))) + L2Norm(vmap(gradient_abs, (0)))

def func_factory(params, tangent_params):
        def func(alpha):
            alpha_params = [(w - alpha * dw, b - alpha * db)
                for (w, b), (dw, db) in zip(params, tangent_params)]
            return loss(alpha_params)
        return func
    

# training
for seed in range(0, 10):
    params = mlp.init_params(layer_sizes, random.PRNGKey(seed))

    errorsL2GD = []
    errorsH1GD = []
    errorsL2NGD = []
    errorsH1NGD = []

    # training loop
    for iteration in range(0):
        grads = grad(loss)(params)
        params, actual_step = ls_update(params, grads)

        print(
            f'seed: {seed} GD: Iteration: {iteration} with loss: '
            f'{loss(params)} with step: {actual_step} with L2 error: '
            f'{L2Norm(v_error)} and H1 error: {H1Norm(error)}'
        )

        errorL2 = L2Norm(v_error)
        errorsL2GD.append(errorL2)
        errorsH1GD.append(H1Norm(error))
        
        if iteration < 999:
            errorsL2NGD.append(None)
            errorsH1NGD.append(None)
        else:
            errorsL2NGD.append(errorL2)
            errorsH1NGD.append(H1Norm(error))

    for epoch in range(1):
        for iteration in range(300):
            grads = grad(loss)(params)
            nat_grads = nat_grad(params, grads)

            flat_natgrad, unravel = jax.flatten_util.ravel_pytree(nat_grads)
            normed_natgrad = unravel(1./jnp.amax(jnp.abs(flat_natgrad)) * flat_natgrad)
            _, actual_step = ls_update(params, normed_natgrad)
            #print(f'actual step: {actual_step}')

            func = func_factory(params, normed_natgrad)
            BFGS = jaxopt.BFGS(
                fun = func,
                value_and_grad=False,
            )
            state = BFGS.init_state(actual_step)
            for iter in range(5):
                actual_step, state = BFGS.update(actual_step, state)
                #print(f'BFGS: iter {iter} actual step: {actual_step}')
                #print(f'iter: {iter} state 0: {state[0]} state 1: {state[1]} l2 error: {L2Norm(v_error)} state 2: {state[2]} state 3: {state[3]}')
            
            params = [(w - actual_step * dw, b - actual_step * db)
                for (w, b), (dw, db) in zip(params, normed_natgrad)]

            errorL2 = L2Norm(v_error)
            errorsL2GD.append(errorL2)
            errorsH1GD.append(H1Norm(error))
            
            if iteration == 199 or iteration == 399 or iteration == 599 or iteration == 799 or iteration == 999 or iteration == 1199 or iteration == 1399 or iteration == 1599:
                errorsL2NGD.append(errorL2)
                errorsH1NGD.append(H1Norm(error))
            else:
                errorsL2NGD.append(None)
                errorsH1NGD.append(None)

            print(
                f'seed: {seed} Epoch: {epoch} NGD: Iteration: '
                f'{iteration} with loss: {loss(params)} with '
                f'step: {actual_step} with L2 error: {errorL2} '
                f'and H1 error {H1Norm(error)}'
            )

        from natgrad.pushforward import pushforward_factory
        from matplotlib import pyplot as plt

        push = pushforward_factory(model)
        v_push = lambda params, tangent_params: vmap(push(params, tangent_params), (0))
        v_u_star = vmap(u_star, (0))

        grads = grad(loss)(params)
        nat_grads = nat_grad(params, grads)
        grads_pushed = v_push(params, grads)
        natgrads_pushed = v_push(params, nat_grads)

        x = interior.deterministic_integration_points(160)

        sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x), s = 10)
        plt.colorbar(sc)
        plt.savefig('out/linesearch/natgrad_pushed.png')
        plt.clf()

        sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(v_model(params, x) - v_u_star(x)))) * (v_model(params, x) - v_u_star(x)), s = 10)
        plt.colorbar(sc)
        plt.savefig('out/linesearch/FspaceGrad.png')
        plt.clf()

        a = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x)
        b = (1./jnp.amax(jnp.abs(v_model(params, x) - v_u_star(x)))) * (v_model(params, x) - v_u_star(x))
        sc = plt.scatter(x[:,0], x[:,1], c = a-b, s = 10)
        plt.colorbar(sc)
        plt.savefig('out/linesearch/FspaceGradMinusNat64.png')
        plt.clf()

        sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(grads_pushed(x)))) * grads_pushed(x), s = 10)
        plt.colorbar(sc)
        plt.savefig('out/linesearch/grad_pushed.png')
        exit()

            
        for iteration in range(1000):
            grads = grad(loss)(params)
            nat_grads = nat_grad(params, grads)
            flat_natgrad, unravel = jax.flatten_util.ravel_pytree(nat_grads)
            normed_natgrad = unravel(1./jnp.amax(jnp.abs(flat_natgrad)) * flat_natgrad)

            params, actual_step = ls_update(params, nat_grads)

            errorL2 = L2Norm(v_error)
            errorsL2GD.append(None)
            errorsH1GD.append(None)
            errorsL2NGD.append(errorL2)
            errorsH1NGD.append(H1Norm(error))

            print(f'seed: {seed} Epoch {epoch} NG: Iteration: {iteration} with loss: {loss(params)} with step: {actual_step} with L2 error: {errorL2} and H1 error {H1Norm(error)}')
    exit()
    jnp.save(
        'data_generation/HeatEq/out/NEWTRAININGerrorsL2GD_seed_' + str(seed) + '.npy',
        jnp.array(errorsL2GD), allow_pickle=False,
        )

    jnp.save(
        'data_generation/HeatEq/out/NEWTRAININGerrorsH1GD_seed_' + str(seed) + '.npy',
        jnp.array(errorsH1GD), allow_pickle=False,
        )
    jnp.save(
        'data_generation/HeatEq/out/NEWTRAININGerrorsL2NGD_seed_' + str(seed) + '.npy',
        jnp.array(errorsL2NGD), allow_pickle=False,
        )

    jnp.save(
        'data_generation/HeatEq/out/NEWTRAININGerrorsH1NGD_seed_' + str(seed) + '.npy',
        jnp.array(errorsH1NGD), allow_pickle=False,
        )
