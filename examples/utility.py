import jax.numpy as jnp
from jax import random
from jax import grad, vmap, jit, random
import jax.flatten_util
import jax
from jax.flatten_util import ravel_pytree

from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker

from natgrad.domains import Interval

class TimeDomain():
    """
    Implements I x Omega, where I is given through two floats
    and Omega is a domain class. If t_0 = t_1 then its the timeslice
    with value t_1.

    todos: alternative constructor with TimeInterval, 
    proper documentation, checks if arguments are within bounds etc. 
    
    """
    def __init__(self, t_0, t_1, domain):

        self._t_0 = t_0
        self._t_1 = t_1
        self._omega = domain

    def measure(self) -> float:
        #NOT GOOD for time slices!
        return 1.
        if self._t_0 == self._t_1:
            return self._omega.measure()
        else:
            return (self._t_1 - self._t_0) * self._omega.measure()
    
    @jaxtyped
    @typechecker
    def random_integration_points(
                self, 
                key: Any, 
                N: int = 50
           ) -> Float[Array, "N d"]:
        
        key_1, key_2 = random.split(key, num=2)

        t = random.uniform(
            key_1, 
            shape =(N, 1), 
            minval = jnp.broadcast_to(
                self._t_0, 
                shape=(N, 1),
                ), 
            maxval = jnp.broadcast_to(
                self._t_1, 
                shape=(N, 1),
                ),
        )
        x = self._omega.random_integration_points(key_2, N)
        return jnp.concatenate((t, x), axis=1)
        
# something like a tensor dataset...
class DataSet():
    def __init__(self, inputs, labels):
        """
        inputs is a tensor of shape (n, d) and corresponds to the n
        data points where we have observations. labels is of shape 
        (n, dim_data) where n is again the number of observations
        and dim_data depends on whether the observations are scalar
        or vectors.
        
        """
        if len(inputs) != len(labels):
            raise ValueError(
                f'[Constructor DataSet: inputs and labels not same '
                f'length]'
            )
        
        self._inputs = inputs
        self._labels = labels
        self._length = len(inputs)

    def sample(self, key, N=50):
        idx = random.randint(
            key,
            shape = (N,),
            minval = 0,
            maxval= self._length,
        )
        
        return self._inputs[idx], self._labels[idx]

class DataIntegrator():
    """
    Something is off here as now the dataset
    is both present in the DataSet class as a member variable
    and in the integrator...

    """
    def __init__(
            self, 
            key, 
            dataset, 
            N = 50, 
            loss = lambda x: x**2,
        ):

        self._N = N
        self._key = key
        self._dataset = dataset
        self._loss = loss
        self._x, self._y = self._dataset.sample(self._key, self._N) 
        
    def __call__(self, f):
        """Inteded to pass to gramian"""
        return jnp.mean(f(self._x), axis=0)
    
    def data_loss(self, f):
        """Intended to use in loss function"""
        return 0.5 * jnp.mean(self._loss(f(self._x) - self._y))
        

    def new_rand_points(self):
        self._key = random.split(self._key, num=1)[0]
        self._x, self._y = self._dataset.sample(self._key, self._N)


# block gramian handling
class IntervalBoundary():
    """
    An interval boundary class providing integration points.

    Parameters
    ----------
    a : float
        Left interval side.
    b : float
        Right interval side.

    """
    def __init__(self, a: float = 0., b: float = 1.):
        if b <= a:
            raise ValueError(
                "[Constructor of Interval]: a < b must hold"
            )
        
        self._a = a
        self._b = b
        
    @jaxtyped
    @typechecker
    def deterministic_integration_points(
                self, 
                N: int = 2
           ) -> Float[Array, "N 1"]:
        """
        N equally spaced collocation points in [a, b].
        
        """
        x = jnp.array([self._a, self._b])
        return jnp.reshape(x, (2, 1))
    
    def measure(self):
        return 1.

    @jaxtyped
    @typechecker
    def random_integration_points(
                self,
                key, 
                N: int = 2
           ) -> Float[Array, "N 1"]:
        """
        N equally spaced collocation points in [a, b].
        Exactly the same as deterministic integration.
        
        """
        idx = random.randint(
            key,
            shape = (N,),
            minval = 0,
            maxval= 2,
        )

        x = jnp.array([self._a, self._b])
        return jnp.reshape(x, (2, 1))[idx]


def flatten_pytrees(pytree_1, pytree_2):
    f_pytree_1, unravel_1 = ravel_pytree(pytree_1)
    f_pytree_2, unravel_2 = ravel_pytree(pytree_2)

    len_1 = len(f_pytree_1)
    len_2 = len(f_pytree_2)
    flat = jnp.concatenate([f_pytree_1, f_pytree_2], axis=0)

    def retrieve_pytrees(flat):
        flat_1 = flat[0:len_1]
        flat_2 = flat[len_1:len_1+len_2]
        return unravel_1(flat_1), unravel_2(flat_2)

    return flat, retrieve_pytrees

def grid_line_search_factory(loss, steps):
    
    def loss_at_step(
            step, 
            params_u, 
            params_v, 
            tangent_params_u,
            tangent_params_v,
        ):
        updated_params_u = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params_u, tangent_params_u)]
        updated_params_v = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params_v, tangent_params_v)]
        return loss(updated_params_u, updated_params_v)
        
    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None, None, None)))    

    @jit
    def grid_line_search_update(
            params_u, 
            params_v, 
            tangent_params_u,
            tangent_params_v,
        ):
        losses = v_loss_at_steps(
            steps, 
            params_u, 
            params_v, 
            tangent_params_u, 
            tangent_params_v
        )
        step_size = steps[jnp.argmin(losses)]
        
        new_params_u = [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params_u, tangent_params_u)]
        
        new_params_v = [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params_v, tangent_params_v)]
        
        return new_params_u, new_params_v, step_size
    return grid_line_search_update

# maps: (model, trafo) ---> (params, x ---> (param_dim, param_dim))
@jaxtyped
@typechecker
def new_pre_gram_factory(
            model_1: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            model_2: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            trafo_1: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
            trafo_2: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
       ) -> Callable[
                [PyTree, PyTree, Float[Array, "d"]],
                Float[Array, "Pdim Pdim"]
            ]:

    # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
    @jaxtyped
    @typechecker
    def del_theta_model_1(
                params: PyTree,
                x: Float[Array, "d"],
           ) -> PyTree:
        return grad(model_1)(params, x)
    
    # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
    @jaxtyped
    @typechecker
    def del_theta_model_2(
                params: PyTree,
                x: Float[Array, "d"],
           ) -> PyTree:
        return grad(model_2)(params, x)
    
    @jaxtyped
    @typechecker
    def pre_gram(
                params_1: PyTree, 
                params_2: PyTree, 
                x: Float[Array, "d"]
           ) -> Float[Array, "Pdim1 Pdim2"]:
        
        @jaxtyped
        @typechecker
        def g_1(y: Float[Array, "d"]) -> PyTree:
            return trafo_1(
                lambda z: model_1(params_1, z),
                lambda z: del_theta_model_1(params_1, z),
            )(y)
        
        @jaxtyped
        @typechecker
        def g_2(y: Float[Array, "d"]) -> PyTree:
            return trafo_2(
                lambda z: model_2(params_2, z),
                lambda z: del_theta_model_2(params_2, z),
            )(y)
        
        flat_1 = jax.flatten_util.ravel_pytree(g_1(x))[0]
        flat_col = jnp.reshape(flat_1, (len(flat_1), 1))
        
        flat_2 = jax.flatten_util.ravel_pytree(g_2(x))[0]
        flat_row = jnp.reshape(flat_2, (1, len(flat_2)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram



# maps: (model, trafo) ---> (params, x ---> (param_dim, param_dim))
@jaxtyped
@typechecker
def new_default_pre_gram_factory(
            model_1: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            trafo_1: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
            model_2: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ] = None,
            trafo_2: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree],
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ] = None,
       ) -> Callable[
                [PyTree, PyTree, Float[Array, "d"]],
                Float[Array, "Pdim Pdim"]
            ]:

    # maps: [(*, *), ..., (*, *)], (d,)  --->   [(*, *), ..., (*, *)]
    @jaxtyped
    @typechecker
    def del_theta_model_1(
                params: PyTree,
                x: Float[Array, "d"],
           ) -> PyTree:
        return grad(model_1)(params, x)
    
    if model_2 is not None:
        @jaxtyped
        @typechecker
        def del_theta_model_2(
                    params: PyTree,
                    x: Float[Array, "d"],
            ) -> PyTree:
            return grad(model_2)(params, x)
    
    @jaxtyped
    @typechecker
    def pre_gram(
                params_1: PyTree, 
                params_2: PyTree, 
                x: Float[Array, "d"]
           ) -> Float[Array, "Pdim1 Pdim2"]:
        
        @jaxtyped
        @typechecker
        def g_1(y: Float[Array, "d"]) -> PyTree:
            return trafo_1(
                lambda z: model_1(params_1, z),
                lambda z: del_theta_model_1(params_1, z),
            )(y)
        
        @jaxtyped
        @typechecker
        def g_2(y: Float[Array, "d"]) -> PyTree:
            return trafo_2(
                lambda z: model_2(params_2, z),
                lambda z: del_theta_model_2(params_2, z),
            )(y)
        
        flat_1 = jax.flatten_util.ravel_pytree(g_1(x))[0]
        flat_col = jnp.reshape(flat_1, (len(flat_1), 1))
        
        flat_2 = jax.flatten_util.ravel_pytree(g_2(x))[0]
        flat_row = jnp.reshape(flat_2, (1, len(flat_2)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram




@jaxtyped
@typechecker
def new_gram_factory(
            model_1: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            model_2: Callable[
                [PyTree, Float[Array, "d"]],
                Float[Array, ""]
            ],
            trafo_1: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree]
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
            trafo_2: Callable[
                [
                    Callable[[Float[Array, "d"]], Float[Array, ""]], 
                    Callable[[Float[Array, "d"]], PyTree]
                    ],
                Callable[[Float[Array, "d"]], PyTree]
            ],
            integrator: Callable,
       ) -> Callable[
                [PyTree, PyTree],
                Float[Array, "Pdim1 Pdim2"]
            ]:

    pre_gram = new_pre_gram_factory(model_1, model_2, trafo_1, trafo_2)
    v_pre_gram = vmap(pre_gram, (None, None, 0))
    
    @jaxtyped
    @typechecker
    def gram(params_1: PyTree, params_2: PyTree) -> Float[Array, "Pdim1 Pdim2"]:
        gram_matrix = integrator(lambda x: v_pre_gram(params_1, params_2, x))
        return gram_matrix
    
    return gram