"""
Contains implementation of a MLP, i.e., a fully connected model.

"""
import jax.numpy as jnp
from jax import random

# ------Copied from Jax documentation-----
def random_layer_params(m: int, n: int, key, scale: float = 1e-1):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
# ----------Copy ends----------------------

# ------Copied from Jax documentation-----
def zero_random_layer_params(m: int, n: int, key, scale: float = 1e-1):
  w_key, b_key = random.split(key)
  return 0.01 * random.normal(w_key, (n, m)), 0.01 * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def zero_init_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [zero_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def mlp(activation): 
    def model(params, inpt):
        hidden = inpt
        for w, b in params[:-1]:
            outputs = jnp.dot(w, hidden) + b
            hidden = activation(outputs)
  
        final_w, final_b = params[-1]
        return jnp.reshape(jnp.dot(final_w, hidden) + final_b, ())
    return model

