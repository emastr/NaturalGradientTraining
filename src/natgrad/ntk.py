import jax.numpy as jnp
from jax import grad, vmap
import jax.flatten_util

def tangent_operator_factory(model, integrator, u_star):

    # maps: PyTree [(*, *), ..., (*, *)], (1,) ---> [(*, *), ..., (*, *)]
    del_theta_model = grad(
        lambda params, x: jnp.reshape(model(params, x), ())
        ) 

    # maps: [(*, *), ..., (*, *)], (1,), (1,) ---> ()
    def kernel(params, x, y):
        del_m_x = del_theta_model(params, x)
        del_m_y = del_theta_model(params, y)
        ntk = jnp.dot(
                jax.flatten_util.ravel_pytree(del_m_x)[0],
                jax.flatten_util.ravel_pytree(del_m_y)[0],
                )
        return ntk
    
    # maps: [(*, *), ..., (*, *)], (n,1), (1,) ---> (n,)
    # maps: (theta, (x_1, x_2, ...x_n), y) ---> (kernel(theta, x_1, y), kernel(theta, x_2, y), ..., kernel(theta, x_n, y))         
    v_kernel = vmap(kernel, (None, 0, None), 0)
    
    # maps: [(*, *), ..., (*, *)], (n,1) ---> (n,)
    v_model = vmap(model, (None, 0))

    def tangent_operator_evaluation(params, y):
        """
        The tangent operator evaluated at a point y \in [0,1]

        [(*, *), ..., (*, *)], (1,)  ---------> ()
        (theta, y)   ------> int_0^1 (u(theta,x) - u^*(x)) * kernel(theta,x,y) dx

        """
        
        return integrator(
            lambda x: (v_model(params, x) - u_star(x)) * v_kernel(params, x, y)
            )
    
    # maps: [(*, *), ..., (*, *)], (n, 1) ---> (n,)
    tangent_operator = vmap(tangent_operator_evaluation, (None, 0), 0)

    return tangent_operator
