# todos:
# 1.1) provide constructor for Boundary from interior via classmethods.
# 2) Provide a way to go from a domain to its time dependent version.
# 2.1) Make CubeBoundary okay, e.g., provide random integration points etc.
# 3) Missing details: Deterministic integration points for general 3d rectangle
# 4) longer term: provide parametric domains.

"""
Contains the implementation of the computational domains.

"""

import jax.numpy as jnp
from jax import random
import math

from typing import Any
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

class Hyperrectangle():
    """
    A product of intervals in R^d.
    
    The hyperrectangle is specified as a product of intervals.
    For example 
    
    intervals = ((0., 1.), (0., 1.), (0., 1.)) 
    
    is the unit cube in R^3. The assumption is that intervals
    is convertable to an array of shape (d, 2).

    Note that no method for deterministic integration points is 
    provided in this class. The Hyperrectangle is potentially a high
    dimensional object. Deterministic integration points should be
    implemented in child classes.

    Parameters
    ----------
    intervals
        An iterable of intervals, see example above.

    """
    def __init__(self, intervals):
        
        self._intervals = jnp.array(intervals)

        l_bounds = None
        r_bounds = None

        if jnp.shape(self._intervals) == (2,):
            l_bounds = self._intervals[0]
            r_bounds = self._intervals[1]

        else:
            l_bounds = self._intervals[:,0]
            r_bounds = self._intervals[:,1]
        
        self._l_bounds = jnp.reshape(
            jnp.asarray(l_bounds, dtype=float),
            newshape = (-1),
            )

        self._r_bounds = jnp.reshape(
            jnp.asarray(r_bounds, dtype=float),
            newshape = (-1),
            )
        
        if len(self._l_bounds) != len(self._r_bounds):
            raise ValueError(f'[In constructor of Hyperrectangle]: intervals '
                    f'is not convertable to an array of shape (d, 2).')

        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(f'[In constructor of Hyperrectangle]: The '
                    f'lower bounds must be smaller than the upper bounds.')

        self._dimension = len(self._l_bounds)


    def measure(self) -> float:
        return jnp.product(self._r_bounds - self._l_bounds)

    @jaxtyped
    @typechecker
    def random_integration_points(
                self, 
                key: Any, 
                N: int = 50
           ) -> Float[Array, "N d"]:
        """
        N uniformly drawn collocation points in the hyperrectangle.
        
        Parameters
        ----------
        key
            A random key from jax.random.PRNGKey(<int>).
        N=50: int
            Number of random points.

        """
        return random.uniform(
            key, 
            shape =(N, self._dimension), 
            minval = jnp.broadcast_to(
                self._l_bounds, 
                shape=(N, self._dimension),
                ), 
            maxval = jnp.broadcast_to(
                self._r_bounds, 
                shape=(N, self._dimension),
                ),
        )

    @jaxtyped
    @typechecker
    def distance_function(
                self, 
                x: Float[Array, "d"]
           ) -> Float[Array, ""]:
        """
        A smooth approximation of the distance fct to the boundary.

        Note that when using this function in implementations for
        loss functions one should explicitly vectorize it using
        for instance vmap(distance_function, (0)) to let it act on
        arrays of shape (n, d) and return (n,).
        
        Parameters
        ----------
        x: Float[Array, "d"]
            A single spatial point x of shape (d,) where d is the
            dimension of the Hyperrectangle.
        """
        
        return jnp.product((x - self._l_bounds) * (x - self._r_bounds))

class Cube(Hyperrectangle):
    """
    A cube of the form [0, a]^3.

    Parameters
    ----------
    a: float
        The side length.
    
    """
    def __init__(self, a):
        if a <= 0.:
            raise ValueError(
                "[Constructor Cube:] Side-length must be positive."
            )
        self._a = a
        super().__init__(((0., a), (0., a), (0., a)))

    @jaxtyped
    @typechecker
    def deterministic_integration_points(
                self, 
                N: int,
           ) -> Float[Array, "N 3"]:
        """
        Grid based integration points.

        Parameters
        ----------
        N: int
            N is the number of integration points in [0,1] meaning
            that in [0,1]^3 there are N^3 integration points.
        
        """
        squareList = []
        a = self._a
        M = max(math.ceil(a) * N, 2)
        for i in range(1, M - 1):
            x = a/(M - 1) * i
            for j in range(1, M - 1):
                y = a/(M - 1) * j
                for k in range(1, M - 1):
                    z = a/(M - 1) * k
                    squareList.append([x, y, z])
    
        if not squareList:
            raise Exception("Too few points to resolve the square.")
        
        return jnp.asarray(squareList)

class Square(Hyperrectangle):
    """
    A square of the form [0, a]^2.

    Parameters
    ----------
    a: float
        The side length.
    
    """
    def __init__(self, a):
        if a <= 0.:
            raise ValueError(
                "[Constructor Square:] Side-length must be positive."
            )
        self._a = a
        super().__init__(((0., a), (0., a)))

    @jaxtyped
    @typechecker
    def deterministic_integration_points(
                self, 
                N: int,
           ) -> Float[Array, "N 2"]:
        """
        Grid based integration points.

        Parameters
        ----------
        N: int
            N is the number of integration points in [0,1] meaning
            that in [0,1]^2 there are N^2 integration points.
        
        """
        squareList = []
        a = self._a
        M = max(math.ceil(a) * N, 2)
        for i in range(1, M-1):
            x = a/(M - 1) * i
            for j in range(1,M-1):
                y = a/(M - 1) * j
                squareList.append([x,y])
    
        if not squareList:
            raise Exception("Too few points to resolve the square.")
        
        # of shape (n,2)
        return jnp.asarray(squareList)

class Interval(Hyperrectangle):
    """
    An interval class providing integration points.

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
        
        super().__init__((a, b))

        self._a = a
        self._b = b
        
    @jaxtyped
    @typechecker
    def deterministic_integration_points(
                self, 
                N: int = 50
           ) -> Float[Array, "N 1"]:
        """
        N equally spaced collocation points in [a, b].
        
        """
        x = jnp.linspace(
            start=self._a,
            stop=self._b, 
            num=N
            )
        return jnp.reshape(x, (N, 1))

class PointBoundary():
    """
    Additional class to accomodate point boundaries. (For 1D problems)
    """
    def __init__(self, intervals):
        self._intervals = jnp.array(intervals).reshape(1, 2)
        self._l_bounds = self._intervals[:, 0]
        self._r_bounds = self._intervals[:, 1]
        
        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(
                f'[In constructor of PointBoundary]: The '
                f'l_bounds must be smaller than the r_bounds.'
                )

        self._dimension = len(self._l_bounds)
        
    def measure(self) -> float:
        return self._r_bounds - self._l_bounds

    @typechecker
    def random_integration_points():
        pass
    
    @typechecker    
    def deterministic_integration_points(
                self, 
                N: int
           ) -> Float[Array, "2 1"]:
        
        # end points of the interval
        a = self._l_bounds[0]
        b = self._r_bounds[0]
        
        return jnp.asarray([a, b]).reshape(2, 1) 
    
class RectangleBoundary():
    """
    One side of a rectangle as a domain.
    
    The numbering is the following:
    
    ----2----
    |       |
    3       1
    |       |
    ----0----

    Parameters
    ----------
    intervals: Array like
        anything that can be converted into an array of shape (2, 2).
        For example, intervals = ((0., 1.), (0, 1.)) will be [0,1]^2.

    side_number: int or slice_object
        Default means the full boundary is returned. Indices or
        slices between 0 and 3 can be used to retrieve other boundaries.

    """
    def __init__(self, intervals, side_number=slice(0, 4)):
        
        if isinstance(side_number, int):
            self._side_number = slice(side_number, side_number + 1)
        
        elif isinstance(side_number, slice):
            self._side_number = side_number
        
        else:
            raise TypeError(
                f'[Constructor Rectangle Boundary:] side_number '
                f'must be integer or slice object.'
            )
        self._intervals = jnp.array(intervals)
        
        if jnp.shape(self._intervals) != (2, 2):
            raise ValueError(
                f'[Constructor Rectangle Boundary Side:] Shape of '
                f'intervals must be (2, 2).'
            )
        
        self._l_bounds = self._intervals[:, 0]
        self._r_bounds = self._intervals[:, 1]
        
        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(
                f'[In constructor of RectangeleBoundarySide]: The '
                f'l_bounds must be smaller than the r_bounds.'
                )

        self._dimension = len(self._l_bounds)

    def measure(self) -> float:
        
        sides_list = [0, 1, 2, 3][self._side_number]
        
        sum = 0
        for side in sides_list:
            if side == 0 or side == 2:
                sum += self._r_bounds[0] - self._l_bounds[0]
            else:
                sum += self._r_bounds[1] - self._l_bounds[1]
        
        return sum

    @typechecker
    def random_integration_points(
                self,
                key,
                N: int = 50
           ) -> Float[Array, "N 2"]:
        
        keys = random.split(key, num=4)
        
        # rectangle = [a_0, b_0] x [a_1, b_1]
        a_0 = self._l_bounds[0]
        b_0 = self._r_bounds[0]
        a_1 = self._l_bounds[1]
        b_1 = self._r_bounds[1]

        # number of points in [a_0, b_0] and [a_1, b_1]
        M_0 = jnp.maximum(math.ceil((b_0 - a_0) * N), 1)
        M_1 = jnp.maximum(math.ceil((b_1 - a_1) * N), 1)

        # points in the four sides
        points_0 = random.uniform(keys[0], (M_0, 1), minval=a_0, maxval=b_0)
        points_1 = random.uniform(keys[1], (M_1, 1), minval=a_1, maxval=b_1)
        points_2 = random.uniform(keys[2], (M_0, 1), minval=a_0, maxval=b_0)
        points_3 = random.uniform(keys[3], (M_1, 1), minval=a_1, maxval=b_1)

        # padding
        a_0_s = a_0 * jnp.ones(shape = (M_0, 1))
        b_0_s = b_0 * jnp.ones(shape = (M_1, 1))
        a_1_s = a_1 * jnp.ones(shape = (M_0, 1))
        b_1_s = b_1 * jnp.ones(shape = (M_0, 1))

        side_0 = jnp.concatenate([points_0, a_1_s], axis = 1)
        side_1 = jnp.concatenate([b_0_s, points_1], axis = 1)
        side_2 = jnp.concatenate([points_2, b_1_s], axis = 1)
        side_3 = jnp.concatenate([a_0_s, points_3], axis = 1)

        sides = [side_0, side_1, side_2, side_3]

        # of shape (n, 2)
        if self._side_number == None:
            return jnp.reshape(jnp.array(sides), (-1,2))
        else:
            return jnp.reshape(
                jnp.array(sides[self._side_number]), 
                (-1,2),
            )
    
    @typechecker    
    def deterministic_integration_points(
                self, 
                N: int
           ) -> Float[Array, "N 2"]:
        
        # rectangle = [a_0, b_0] x [a_1, b_1]
        a_0 = self._l_bounds[0]
        b_0 = self._r_bounds[0]
        a_1 = self._l_bounds[1]
        b_1 = self._r_bounds[1]
        
        # number of points in [a_0, b_0] and [a_1, b_1]
        M_0 = jnp.maximum(math.ceil((b_0 - a_0) * N), 1)
        M_1 = jnp.maximum(math.ceil((b_1 - a_1) * N), 1)

        # use M_0-1 to not have double corners
        interval_x = jnp.reshape(
            jnp.linspace(a_0, b_0, M_0)[0:M_0 - 1], 
            (M_0 - 1, 1)
        )

        # use M_0-1 to not have double corners
        interval_x_back = jnp.reshape(
            jnp.linspace(b_0, a_0, M_0)[0:M_0 - 1], 
            (M_0 - 1, 1)
        )

        # use M_1-1 to not have double corners
        interval_y = jnp.reshape(
            jnp.linspace(a_1, b_1, M_1)[0:M_1 - 1], 
            (M_1 - 1, 1)
        )

        # use M_1-1 to not have double corners
        interval_y_back = jnp.reshape(
            jnp.linspace(b_1, a_1, M_1)[0:M_1 - 1], 
            (M_1 - 1, 1)
        )

        # padding
        a_0_s = a_0 * jnp.ones(shape = (M_0 - 1, 1))
        b_0_s = b_0 * jnp.ones(shape = (M_1 - 1, 1))
        a_1_s = a_1 * jnp.ones(shape = (M_0 - 1, 1))
        b_1_s = b_1 * jnp.ones(shape = (M_0 - 1, 1))

        side_0 = jnp.concatenate([interval_x, a_1_s], axis = 1)
        side_1 = jnp.concatenate([b_0_s, interval_y], axis = 1)
        side_2 = jnp.concatenate([interval_x_back, b_1_s], axis = 1)
        side_3 = jnp.concatenate([a_0_s, interval_y_back], axis = 1)

        sides = [side_0, side_1, side_2, side_3]

        # of shape (n, 2)
        if self._side_number == None:
            return jnp.reshape(jnp.array(sides), (-1,2))
        else:
            return jnp.reshape(
                jnp.array(sides[self._side_number]), 
                (-1,2),
            )

class SquareBoundary(RectangleBoundary):
    """
    Boundary of the Square [0, a]^2.

    """
    def __init__(self, a, side_number=slice(0, 4)):
        if a <= 0.:
            raise ValueError("A side-length must be positive.")
        self._a = a

        super().__init__(((0., a), (0., a)), side_number=side_number)

# just a preliminary implementation
class CubeBoundary():
    """
    Constructor that sets side-length, i.e., the cube is [0, a]^3
    
    """
    def __init__(self, a):
        print(
            f"[CubeBoundary Constructor] This is a preliminary"
            f" implementation!"
        )

        if a <= 0.:
            raise Exception("A side-length must be positive.")

        self._a = a

    def measure(self):
        return self._a * self._a #?

    # N is number of points in [0,1]
    def deterministic_integration_points(self, N):
        square = Square(self._a)
        x = square.deterministic_integration_points(N)
        zeros = jnp.zeros((len(x)))
        ones =  self._a * jnp.ones((len(x)))
        
        bdry_0 = jnp.transpose(jnp.array((zeros,  x[:,0], x[:,1])))
        bdry_1 = jnp.transpose(jnp.array((x[:,0], zeros,  x[:,1])))
        bdry_2 = jnp.transpose(jnp.array((x[:,0], x[:,1], zeros)))
        
        bdry_3 = jnp.transpose(jnp.array((ones,   x[:,0], x[:,1])))
        bdry_4 = jnp.transpose(jnp.array((x[:,0], ones,   x[:,1])))
        bdry_5 = jnp.transpose(jnp.array((x[:,0], x[:,1], ones)))
        
        return jnp.concatenate(
            (bdry_0, bdry_1, bdry_2, bdry_3, bdry_4, bdry_5), 
            axis=0,
            )
        
    def distance_function(self, xyz):
        pass