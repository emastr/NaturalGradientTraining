"""
This module contains the implementation of integration routines.

"""
import jax.numpy as jnp
from jax import random

class EvolutionaryIntegrator():
    """
    First Implementation of an evolutionary integrator following
    the proposed algorithm of Daw et al in "Mitigating Propagation
    Failure in PINNs using Evolutionary Sampling". 
    
    """
    def __init__(self, domain, key, N=50, K=None):
        self._domain = domain
        self._N = N
        self._key = key
        self._x = self._domain.random_integration_points(self._key, N)
        self._key = random.split(self._key, num=1)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        

    def __call__(self, f):
        """
        Integration happens here, f must map (n,...) to (n,...)
        """

        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))
            
            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean

    def update(self, residual):
        # compute fitness from residual
        fitness = jnp.abs(residual(self._x))
        
        # set the threshold
        threshold = jnp.mean(fitness)
        
        # remove non-fit collocation points
        mask = jnp.where(fitness > threshold, False, True)
        x_fit = jnp.delete(self._x, mask, axis=0)
        
        # add new uniformly drawn collocation points to fill up
        N_fit = len(self._x) - len(x_fit)
        x_add = self._domain.random_integration_points(self._key, N_fit)
        self._x = jnp.concatenate([x_fit, x_add], axis=0)
        
        # advance random number generator
        self._key = random.split(self._key[0], num=1)

class RandomIntegrator():
    """
    Monte Carlo Integration
    """
    def __init__(self, domain, key, N=50):
        self._domain = domain
        self._N = N
        self._x = self._domain.random_integration_points(key, self._N)
        self._key = random.split(key, num=1)[0]
        

    def __call__(self, f):
        """
        Integration happens here, f must map (n,...) to (n,...)
        """
        return self._domain.measure() * jnp.mean(f(self._x), axis=0)

    def update(self):
        self._x = self._domain.random_integration_points(self._key, self._N)
        self._key = random.split(self._key, num=1)[0]

class TrapezoidalIntegrator():
    """
    Integration over intervals using trapezoidal rule.

    """
    def __init__(self, domain, N=50, K=None):
        self._domain = domain
        self._x = domain.deterministic_integration_points(N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        
    def __call__(self, f):
        """
        Integration happens here. 

        f must map (n,1) tensors to (n,...) tensors.
        
        """
        
        sum = 0
        if self._K is not None:
            sums = []
            for x in self.x_split:
                sums.append(jnp.sum(f(x), axis=0))
            
            sum = jnp.sum(jnp.array(sums), axis=0)

        else:
            y = f(self._x)
            sum = jnp.sum(y, axis=0)

        x_first = jnp.expand_dims(self._x[0], axis=0)
        x_last  = jnp.expand_dims(self._x[-1], axis=0)
        dx = self._x[1,0] - self._x[0,0]

        # get rid of first axis of f(...) by evaluating at [0]
        return dx * (sum - 0.5 * (f(x_first)[0] + f(x_last)[0]))
            
class DeterministicIntegrator():
    """
    Integration over domains.

    """
    def __init__(self, domain, N=50, K=None):
        self._domain = domain
        self._x = domain.deterministic_integration_points(N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        
        
    def __call__(self, f):
        """
        Integration happens here, f must map (n,...) to (n,...)
        """

        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))
            
            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean

    # Test that assembly of Gramian also works!
    def old_call(self, f):
        y = f(self._x)

        if len(jnp.shape(y)) == 3:
            #integrate matrix-valued functions
            return self._domain.measure() * jnp.mean(y, axis=0)
        
        else:
            return self._domain.measure() * jnp.mean(y)