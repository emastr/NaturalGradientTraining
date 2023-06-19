"""
1d Pinn with exactly imposed boundary values.

"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt

from natgrad.domains import Interval
import natgrad.mlp as mlp
from natgrad.integrators import TrapezoidalIntegrator
from natgrad.pushforward import pushforward_factory
from natgrad.gram import gram_factory, nat_grad_factory
from natgrad.inner.any_d import model_laplace as model_second_derivative

from jaxtyping import Array, Float, PyTree
from typeguard import typechecked as typechecker


jax.config.update("jax_enable_x64", True)

# integration
interval = Interval()
integrator = TrapezoidalIntegrator(interval, 250)
eval_integrator = TrapezoidalIntegrator(interval, 2500)

# model
activation = lambda x : jnp.tanh(x)
#activation = lambda x : jnp.sin(x)**2
#activation = lambda x : jnp.maximum(0., x)**3
layer_sizes = [1, 32, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(0))

# maps: [(*, *), ..., (*, *)], (1,) ---> ()
model = mlp.mlp(activation) 

# maps: [(*, *), ..., (*, *)], (1,) ---> ()
@typechecker
def truncated_model(params: PyTree, x: Float[Array, "1"]) -> Float[Array, ""]:
    return jnp.reshape(model(params, x) * (x - jnp.ones_like(x)) * x, ())

# maps: [(*, *), ..., (*, *)], (n,1) ---> (n,)
v_model = vmap(model, (None, 0))

# maps: [(*, *), ..., (*, *)], (n,1) ---> (n,)
v_truncated_model = vmap(truncated_model, (None, 0))

# right-hand side
# maps: (n, 1) ---> (n,)
def f(x):
    return jnp.reshape(3.14159265**2 * jnp.sin(3.14159265 * x), (len(x)))

# solution, maps like rhs
# maps: (n, 1) ---> (n,)
u_star = lambda x: (3.14159265**2)**(-1.) * f(x)

"""
x = jnp.array([1.])
xx = jnp.array([[1.], [1.], [0.]])
print(truncated_model(params, x))
print(jnp.shape(truncated_model(params, x)))
print()
print(grad(lambda x: truncated_model(params, x))(x))
print(jnp.shape(grad(lambda x: truncated_model(params, x))(x)))
"""


# maps [(*,*), ..., (*,*)] ---> ()
def loss(params):
    # maps: (n, 1) ---> (n,1)
    laplace_model_ = vmap(grad(
        lambda x: jnp.reshape(grad(lambda x: truncated_model(params, x))(x),())
        ), (0))
    
    # maps: (n, 1) ---> (n,)
    laplace_model = lambda x: jnp.reshape(laplace_model_(x), (len(x)))

    # maps: (n, 1) ---> (n,)
    integrand = lambda x: (laplace_model(x) + f(x))**2

    #print(jnp.shape(integrator(integrand)))
    #exit()

    return jnp.reshape(integrator(integrand), ())

# L2Norm
# func must map like u_star, i.e, f
def L2Norm(func):
    return eval_integrator(
        lambda x: (func(x))**2
        )**0.5


# maps: [(*, *), ...], [(*, *), ...] ---> (function: (d,) ---> ())
push = pushforward_factory(model)

# maps: [(*,*), ...], [(*,*), ...] ---> (function: (n,d) ---> (n,))
v_push = lambda params, tangent_params: vmap(push(params, tangent_params), (0))


# maps: params ---> (Pdim, Pdim)
gram = gram_factory(
    model = truncated_model,
    trafo = model_second_derivative,
    integrator = integrator
)

# maps: params, tangent_params ---> tangent_params
nat_grad = nat_grad_factory(gram)

# gradient update
@jit
def update(params, step_size):
    grads = grad(loss)(params)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


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
            for (w, b), (dw, db) in zip(params, grads)]



error_func = lambda x: v_truncated_model(params, x) - u_star(x)

# training loop
step = .055
for iteration in range(50):
    params = update(params, step)
    if iteration > 4000:
        step = 0.01
    error = L2Norm(error_func)
    print(f'Iteration: {iteration} with loss: {loss(params)} with error: {error}')
    
# natural gradient descent with line search
for iteration in range(100):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    
    #params = line_search(loss, params, nat_grads)
    params = grid_line_search(params, nat_grads)

    error = L2Norm(error_func)
    print(f'Iteration: {iteration} with loss: {loss(params)} with error: {error}')
    



#----------------grafical output -----------------#
x = interval.deterministic_integration_points(N=50)

# test pushforward
param_grad = grad(loss)(params)
tangent_function = v_push(params, param_grad)
kla = (1./jnp.amax(jnp.abs(tangent_function(x)))) * tangent_function(x)
plt.plot(x, kla, color='green', label='Push')

# test natural sobolev gradient
param_sob_nat_grad = nat_grad(params, param_grad)
klakla_ = v_push(params, param_sob_nat_grad)
kkk_ = (1./jnp.amax(jnp.abs(klakla_(x)))) * klakla_(x)
plt.plot(x, kkk_, color='blue', label='SobNatGrad')


# F space gradient
norm = jnp.amax(jnp.abs(v_truncated_model(params, x) - (3.14159265**2)**(-1.) * f(x)))
plt.plot(x, 1./norm * (v_truncated_model(params, x) - (3.14159265**2)**(-1.) *  f(x)), color='red', label='Error')
plt.legend()

plt.savefig('out/pinn.png')

plt.clf()
plt.plot(x, v_truncated_model(params, x))
plt.savefig('out/pinn_preds.png')
plt.clf()
plt.plot(x, v_truncated_model(params, x) - (3.14159265**2)**(-1.) *  f(x))
plt.savefig('out/pinn_error.png')