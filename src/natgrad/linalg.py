import jax.numpy as jnp

def sherman_morrison(Ainv, u, v):
    Ainv_u = Ainv @ u
    v_Ainv = v @ Ainv
    return Ainv - jnp.outer(Ainv_u, v_Ainv) / (1 + v @ Ainv_u)