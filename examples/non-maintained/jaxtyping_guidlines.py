"""
Example file for our usage of jaxtyping.

I strongly suggest that we type our array operations carefully. From my
(limited) experience with array-based packages (everything from numpy 
over tensorflow to jax) I make a lot of mistakes with wrong shapes and 
broadcasting of arrays etc. Therefore its nice to automate checks.
Jaxtyping is a small packages with that purpose. 

This file serves to identify useful practices that we might wish to 
adopt. The main purpose is not pedantic strictness but just the 
guarantee of correctness of our code.

"""


import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker
from typing import Any, Union

import natgrad.mlp as mlp
from jax import random


# right-hand side
# maps: (n, 1) ---> (n,)

@jaxtyped
@typechecker
def f(x: Float[Array, "n 1"]) -> Float[Array, "n"]:
    """
    @typechecker raises an error when either the input is not an array
    with variable first dimension and second dimension of 1 or when
    the output is not one-dimensional with variable length. BUT 
    @typchecker does not check if both input and output are of
    length n. This is done by @jaxtyped.

    Adavantage of using this: we know at least that our function f
    shall only receive what we intend to and shall only return what we
    desire. 

    Potential source of error: Still intermediate comuptations within
    f are not checked.

    I do not see any downside with annotating this way.
    
    """
    return jnp.reshape(jnp.ones_like(x), (len(x)))

x = jnp.array([[1.], [1.]])
print('f(x)', f(x))
print('shape', jnp.shape(f(x)))

#--------------------PyTrees The relaxed version------------------#

# will check that the weights are a PyTree with leaf variables 
# that are arrays that are either 2-dim or 1-d. Will not check
# that the specific sizes will match in any way. That can be 
# achieved with the stricter setup below.
MlpParams = PyTree[Union[Float[Array, "m n"], Float[Array, "l"]]]

model_ = mlp.mlp(lambda x: jnp.maximum(0., x)) 

@typechecker
def model(theta: MlpParams, x: Float[Array, "1"]) -> Float[Array, ""]:
    return model_(theta, x)

layer_sizes = [1, 5, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(0))
x = jnp.array([1.])

model(params, x)


#-------------------------The strict version----------------------#

ShallowParams = PyTree[Union[
    Float[Array, "n 1"], 
    Float[Array, "n"], 
    Float[Array, "1 n"], 
    Float[Array, "1"]]
    ]

model_ = mlp.mlp(lambda x: jnp.maximum(0., x)) 

# maps: [(*, *), ..., (*, *)], (1,) ---> ()
@jaxtyped
@typechecker
def model(theta: ShallowParams, x: Float[Array, "1"]) -> Float[Array, ""]:
    return model_(theta, x)

layer_sizes = [1, 5, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(0))
x = jnp.array([1.])

model(params, x)
