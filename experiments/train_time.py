# export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
import jax
from jax.scipy.sparse.linalg import cg
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.numpy.linalg import lstsq
import matplotlib.pyplot as plt

import natgrad.mlp as mlp
from natgrad.domains import Square
from natgrad.domains import SquareBoundary, Polygon
from natgrad.integrators import RandomIntegrator, DeterministicIntegrator, EvolutionaryIntegrator
from natgrad.derivatives import laplace
from natgrad.inner import model_laplace, model_identity
from natgrad.gram import gram_factory, nat_grad_factory, nat_grad_factory_generic
from natgrad.utility import grid_line_search_factory
from natgrad.plotting import plot_2d_func, plot_2d_funcs
from natgrad.linalg import sherman_morrison
from jaxopt import LevenbergMarquardt as LMJax
from jaxopt.linear_solve import solve_cg
from natgrad.levenberg_marquardt import LevenbergMarquardt as LM
from natgrad.logger import EventTracker
import timeit

jax.config.update("jax_enable_x64", True)



def get_cg_solver(*args, **kwargs):
    def cg_solver(A, b, **kwargs2):
        kwargs2.update(kwargs)
        return solve_cg(A, b, *args, **kwargs)
    return cg_solver

presets = {"LM_full": {"materialize_jac": "full", "solver": get_cg_solver(maxiter=50), "damping_parameter": 1e-8},
           "LM_semi": {"materialize_jac": "semi", "solver": get_cg_solver(maxiter=50), "damping_parameter": 1e-8},
           "LM_none": {"materialize_jac": "none", "solver": get_cg_solver(maxiter=50), "damping_parameter": 1e-8}}

# random seed
seed = 2
freq = 1.

# domains
interior = Square(1.) #Polygon(jnp.array([[0,0], [1,0], [0,1]])) # S
boundary = interior.boundary()

# integrators
key = random.PRNGKey(seed)
interior_integrator = RandomIntegrator(interior, key, 2000)
boundary_integrator = RandomIntegrator(boundary, key, 2000)
eval_integrator = RandomIntegrator(interior, key, 200)

# model
activation = lambda x : jnp.tanh(x)
model = mlp.mlp(activation)
v_model = vmap(model, (None, 0))


# solution
@jit
def u_star(xy): return jnp.sin(freq * jnp.pi * xy[0]) * jnp.sin(freq * jnp.pi * xy[1])

# rhs
@jit
def f(xy):
    return 2. * (freq * jnp.pi)**2 * u_star(xy)


# loss
@jit
def interior_loss(params):
    return interior_integrator(lambda x: v_res_interior(params, x)**2)

@jit
def boundary_loss(params):
    return boundary_integrator(lambda x: v_res_bdry(params, x)**2)

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)


grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# trick to get the signature (params, v_x) -> v_residual
_res = lambda params: laplace(lambda x: model(params, x))
res_interior = lambda params, x: (_res(params)(x) + f(x))
v_res_interior =  jit(vmap(res_interior, (None, 0)))

res_bdry = lambda params, x: model(params, x)
v_res_bdry = jit(vmap(res_bdry, (None, 0)))

def l2_norm(f, integrator):
        return integrator(lambda x: (f(x))**2)**0.5    



            

Ns = [8, 12, 16, 24, 32, 48, 64, 81]
for N in Ns: #[Ns[i] for i in [6, 7, 8]]:    
    for preset in presets.keys():
        
        print(f"\n {preset} Net size = {N}")
        layer_sizes = [2, N, N, 1]
            
        params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
        _, unflatten = jax.flatten_util.ravel_pytree(params)
        flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]

        error = lambda x: model(params, x) - u_star(x)
        v_error = vmap(error, (0))
        v_error_abs_grad = vmap(lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5)

        def residuals(params):
            params = unflatten(params)
            boundary_res_pts = vmap(res_bdry, (None, 0))(params, boundary_integrator._x) / (boundary_integrator._N)**0.5
            interior_res_pts = vmap(res_interior, (None, 0))(params, interior_integrator._x) / (interior_integrator._N)**0.5
            return jnp.concatenate([boundary_res_pts, interior_res_pts])

        optim = LM(residuals, scale_invariant=False, **presets[preset])
        optim.__post_init__()

        def test():
            params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
            nat_grad, grad, loss_val = optim.update(flatten(params))
            params, actual_step = ls_update(params, unflatten(nat_grad))
        
        time = timeit.timeit(test, number=128//N) / float(128 // N)
        jnp.save(f"/home/emastr/phd/NaturalGradientTraining/data/timing_batch_2000_{preset}_{N}.npy", {"time": time})
        
        