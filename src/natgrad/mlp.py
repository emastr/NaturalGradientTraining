"""
Contains implementation of a MLP, i.e., a fully connected model.

"""
import jax.numpy as jnp
from jax import random, vmap

# ------Copied from Jax documentation-----
def random_layer_params(m: int, n: int, key, scale: float = 1e-1):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
# ----------Copy ends----------------------

def init_random_features(layers, key, feature_scale=1., **kwargs):
  params = init_params(layers, key=random.split(key, num=1)[0])
  params[0] = (params[0][0] * feature_scale, params[0][1] * feature_scale)
  return params
  
  
# ------Copied from Jax documentation-----
def zero_random_layer_params(m: int, n: int, key, scale: float = 1e-1):
  w_key, b_key = random.split(key)
  return 0.01 * random.normal(w_key, (n, m)), 0.01 * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def zero_init_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [zero_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

  

def affine(param, inpt):
  w, b = param
  return jnp.dot(w, inpt) + b

def mlp(activation): 
    def model(params, inpt):
        hidden = affine(params[0], inpt)
        for p in params[1:]:
          hidden = affine(p, activation(hidden))        
        return jnp.reshape(hidden, ())
    return model
  
  
def mlp_skip(activation): 
    """Just a MLP with skip connections. Intermediane layers must have the same size."""
    
    # Define model
    def model(params, inpt):
        hidden = affine(params[0], inpt)
        for p in params[1:-1]:
            hidden = hidden + activation(affine(p, hidden))
            
        return jnp.reshape(affine(params[-1], hidden), ())
    return model


def trig_features(param, x):
    return jnp.cos(affine(param, x))
    
def random_features_mlp(param, x):
    features = trig_features(param[0], x)
    return mlp(jnp.tanh)(param[1:], features)
