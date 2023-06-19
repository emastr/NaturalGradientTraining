import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from matplotlib import pyplot as plt
import jax.flatten_util

import natgrad.mlp as mlp
from natgrad.pushforward import pushforward_factory

from natgrad.domains import Square
from natgrad.domains import SquareBoundary
from natgrad.integrators import DeterministicIntegrator
from natgrad.inner.any_d import model_identity
from natgrad.gram import gram_factory, nat_grad_factory

from jaxtyping import Array, Float, PyTree
from typeguard import typechecked as typechecker

jax.config.update("jax_enable_x64", True)

# integration 
square = Square(1.)
square_boundary = SquareBoundary(1.)
bdry_integrator = DeterministicIntegrator(square_boundary, 100)
interior_integrator = DeterministicIntegrator(square, 60)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [2, 16, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(0))
model = mlp.mlp(activation) 
v_model = vmap(model, (None, 0))

# maps: [(*, *), ...], [(*, *), ...] ---> (function: (d,) ---> ())
push = pushforward_factory(model)

# maps: [(*,*), ...], [(*,*), ...] ---> (function: (n,d) ---> (n,))
v_push = lambda params, tangent_params: vmap(push(params, tangent_params), (0))

# maps: params ---> (Pdim, Pdim)
gram_bdry = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = bdry_integrator
)

# maps: params ---> (Pdim, Pdim)
gram_interior = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = interior_integrator
)

# the full inner product
gram = lambda params: gram_bdry(params)# + 0.00000001 * gram_interior(params)

# maps: params, tangent_params ---> tangent_params
nat_grad = nat_grad_factory(gram)

# maps [(*,*), ..., (*,*)] ---> ()
@typechecker
def loss_bdry(params: PyTree) -> Float[Array, ""]:
    
    # maps: (2,) ---> ()
    bdry_integrand = lambda x: model(params, x)**2

    v_bdry_integrand = vmap(bdry_integrand, (0))
    
    return jnp.reshape(bdry_integrator(v_bdry_integrand), ())

# first order update
@jit
def update(params, tangent_params, step_size):
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]


for iteration in range(20):
    grads = grad(loss_bdry)(params)
    
    nat_grads = nat_grad(params, grads)
    gram_matrix_bdry = gram_bdry(params)
    flat_grad = jax.flatten_util.ravel_pytree(grads)[0]
    flat_nat_grad = jax.flatten_util.ravel_pytree(nat_grads)[0]
    r = jnp.linalg.matrix_rank(gram_matrix_bdry)

    params = update(params, grads, step_size = 0.01)
    print(
        f'Iteration: {iteration}; '
        f'Loss: {loss_bdry(params)} Residual: '
        f'{jnp.linalg.norm(jnp.dot(gram_matrix_bdry, flat_nat_grad) - flat_grad)}; '
        f'Rank of the Gram matrix: {r}'
    )
    


x = square.deterministic_integration_points(80)
sc = plt.scatter(x[:,0], x[:,1], c = v_model(params, x), s = 20)
plt.colorbar(sc)
plt.savefig('out/boundaryL2/model.png')
plt.clf()

grads = grad(loss_bdry)(params)
nat_grads = nat_grad(params, grads)
grads_pushed = v_push(params, grads)
natgrads_pushed = v_push(params, nat_grads)

# natgrad
sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x), s = 20)
plt.colorbar(sc)
plt.savefig('out/boundaryL2/natgrad_pushed.png')
plt.clf()

# Fspace grad
sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(v_model(params, x)))) * (v_model(params, x)), s = 20)
plt.colorbar(sc)
plt.savefig('out/boundaryL2/FspaceGrad.png')
plt.clf()

# Param grad
sc = plt.scatter(x[:,0], x[:,1], c = (1./jnp.amax(jnp.abs(grads_pushed(x)))) * grads_pushed(x), s = 20)
plt.colorbar(sc)
plt.savefig('out/boundaryL2/grad_pushed.png')
plt.clf()

# Difference natgrad Fspace grad
a = (1./jnp.amax(jnp.abs(natgrads_pushed(x)))) * natgrads_pushed(x)
b = (1./jnp.amax(jnp.abs(v_model(params, x)))) * (v_model(params, x))
sc = plt.scatter(x[:,0], x[:,1], c = a-b, s = 20)
plt.colorbar(sc)
plt.savefig('out/boundaryL2/FspaceGradMinusNat.png')
plt.clf()