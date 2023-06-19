import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt

import natgrad.mlp as mlp

from natgrad.gram import gram_factory, nat_grad_factory

from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.derivatives import laplace, model_laplace

from typing import Any
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

#-Delta u(x,y) = sin(x)sin(y) on the domain [-pi,pi]

jax.config.update("jax_enable_x64", True)

# integration 
square = Square(3.14159265)
#square = Square(1.)
square_boundary = SquareBoundary(3.14159265)
integrator = DeterministicIntegrator(square, 20)
eval_integrator = DeterministicIntegrator(square, 200)
bdry_integrator = DeterministicIntegrator(square_boundary, 50)
x = square.deterministic_integration_points(20)

# model
activation = lambda x : jnp.tanh(x)
#activation = lambda x : jnp.sin(x)**2
#activation = lambda x : jnp.maximum(0., x)**3
layer_sizes = [2, 32, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(0))
# maps: [(*, *), ..., (*, *)], (2,) ---> ()
model = mlp.mlp(activation) 


distance_func = square.distance_function
v_distance_func = vmap(distance_func, (0))

# maps: [(*, *), ..., (*, *)], (2,) ---> ()
@typechecker
def truncated_model(params: Any, x: Float[Array, "2"]) -> Float[Array, ""]:
    return model(params, x) * distance_func(x)

# maps: [(*, *), ..., (*, *)], (n,2) ---> (n,)
v_model = vmap(model, (None, 0))

# maps: [(*, *), ..., (*, *)], (n,2) ---> (n,)
v_truncated_model = vmap(truncated_model, (None, 0))

# right-hand side
# maps: (n, 2) ---> (n,)
def f_(xy):
    x = xy[:,0]
    y = xy[:,1]
    return 2. * jnp.sin(y) * jnp.sin(x)

# right-hand side
# maps: (2,) ---> ()
def f(xy):
    x = xy[0]
    y = xy[1]
    return 2. * jnp.sin(y) * jnp.sin(x)

# solution
# maps: (n, 2) ---> (n,)
def u_star(xy):
    x = xy[:,0]
    y = xy[:,1]
    return jnp.sin(y) * jnp.sin(x)

def u_star_(xy):
    x = xy[0]
    y = xy[1]
    return jnp.sin(y) * jnp.sin(x)

laplace_test = laplace(u_star_)
v_laplace_test = vmap(laplace_test, (0))
plt.scatter(x[:,0], x[:,1], c = -v_laplace_test(x) - f_(x), s = 20)
#plt.show()

# maps [(*,*), ..., (*,*)] ---> ()
@typechecker
def loss(params: Any) -> Float[Array, ""]:
    # maps (2,) ---> ()
    laplace_model = laplace(lambda x: truncated_model(params, x))

    # maps: (2,) ---> ()
    integrand = lambda x: (laplace_model(x) + f(x))**2

    # maps: (n, 2) ---> (n,)
    v_integrand_ = vmap(integrand, (0))
    @jaxtyped
    @typechecker
    def v_integrand(x: Float[Array, "n 2"]) -> Float[Array, "n"]:
        return v_integrand_(x)
    
    return jnp.reshape(integrator(v_integrand), ())

# L2Norm
# func must map: (n, 2) ---> (n,)
def L2Norm(func):
    return eval_integrator(
        lambda x: (func(x))**2
        )**0.5

# maps: params ---> (Pdim, Pdim)
gram_laplace = gram_factory(
    model = truncated_model,
    trafo = model_laplace,
    integrator = integrator,
)

# maps: params, tangent_params ---> tangent_params
nat_grad = nat_grad_factory(gram_laplace)



@jit
def grid_line_search(params, grads):
    grid = jnp.linspace(0, 400, 401)
    steps = 0.9**grid

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

# gradient update
@jit
def update(params, step_size):
    grads = grad(loss)(params)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

error_func = lambda x: v_truncated_model(params, x) - u_star(x)

# training loop
step = .0001
for iteration in range(100):
    params = update(params, step)
    if iteration > 4000:
        step = 0.01
    error = L2Norm(lambda x: v_truncated_model(params, x) - u_star(x))
    print(f'Iteration: {iteration} with loss: {loss(params)} with error: {error}')

# natural gradient descent with line search
for iteration in range(100):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    
    #params = line_search(loss, params, nat_grads)
    params, step_size = grid_line_search(params, nat_grads)

    error = L2Norm(error_func)
    print(f'Iteration: {iteration} with loss: {loss(params)} with error: {error} with step size: {step_size}')




plt.scatter(x[:,0], x[:,1], c = v_truncated_model(params, x), s = 20)
plt.savefig('out/2dpreds.png')
plt.clf()
plt.scatter(x[:,0], x[:,1], c = u_star(x), s = 20)
plt.savefig('out/2dgroundtruth.png')
plt.clf()
plt.scatter(x[:,0], x[:,1], c = v_truncated_model(params, x) - u_star(x), s = 20)
plt.savefig('out/2derror.png')
