from jaxtyping import Array, Float, PyTree, jaxtyped
import jax.numpy as jnp
from typeguard import typechecked

# Accepts floating-point 2D arrays with matching dimensions
def matrix_multiply(x: Float[Array, "dim1 dim2"],
                    y: Float[Array, "dim2 dim3"]
                  ) -> Float[Array, "dim1 dim3"]:
    ...

def accepts_pytree_of_ints(x: PyTree[int]):
    ...

def accepts_pytree_of_arrays(x: PyTree[Float[Array, "batch c1 c2"]]):
    ...

@jaxtyped
@typechecked
def u_star(x: Float[Array, "batch 1"]
         ) -> Float[Array, "batch"]:
    #return jnp.array([jnp.sum(jnp.sin(x)), jnp.sum(jnp.sin(x)), jnp.sum(jnp.sin(x))])
    return jnp.reshape(jnp.sin(2. * 3.14159265 * x), (len(x))) # we want (n,1) --> (n,)

x = jnp.array([[1.], [1.]])
print(jnp.shape(u_star(x)))