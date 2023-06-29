# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Levenberg-Marquardt algorithm in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

from typing import Tuple
from typing_extensions import Literal
import jax
import jax.numpy as jnp

from jax import jit
from jaxopt._src import base
from jaxopt._src.linear_solve import solve_cg
from jaxopt._src.linear_solve import solve_cholesky
from jaxopt._src.linear_solve import solve_inv
from jax.numpy.linalg import lstsq
from jaxopt._src.tree_util import tree_l2_norm, tree_inf_norm, tree_sub, tree_add, tree_mul


@dataclass(eq=False)
class LevenbergMarquardt():
  """Levenberg-Marquardt nonlinear least-squares solver.

    Given the residual function `func` (x): R^n -> R^m, `least_squares` finds a
    local minimum of the cost function F(x):

    ```
    argmin_x F(x) = 0.5 * sum(f_i(x)**2), i = 0, ..., m - 1
    f(x) = func(x, *args)
    ```

    If stop_criterion is 'madsen-nielsen', the convergence is achieved once the
    coeff update satisfies ``||dcoeffs||_2 <= xtol * (||coeffs||_2 + xtol) `` or
    the gradient satisfies ``||grad(f)||_inf <= gtol``.

  Attributes:
    residual_fun: a smooth function of the form ``residual_fun(x, *args,
      **kwargs)``.
    maxiter: maximum increase_factormber of iterations.
    damping_parameter: The parameter which adds a correction to the equation
      derived for updating the coefficients using Gauss-Newton method. Please
      see section 3.2. of K. Madsen et al. in the book "Methods for nonlinear
      least squares problems" for more information.
    stop_criterion: The criterion to use for the convergence of the while loop.
      e.g., for 'madsen-nielsen' the criteria is to satisfy the two equations
      for delta_params and gradient that is mentioned above. If 'grad-l2' is
      selected, the convergence is achieved if l2 of gradient is smaller or
      equal to tol.
    tol: tolerance.
    xtol: float, optional The convergence tolerance for the second norm of the
      coefficient update.
    gtol: float, optional The convergence tolerance for the inf norm of the
      residual gradient.
    solver: str, optional The solver to use when finding delta_params, the
      update to the params in each iteration. This is done through solving a
      system of linear equation Ax=b. 'cholesky' (Cholesky factorization), 'inv'
      (explicit multiplication with matrix inverse). The user can provide custom
      solvers, for example using jaxopt.linear_solve.solve_cg which are more
      scalable for runtime but take longer compilations. 'cholesky' is
      faster than 'inv' since it uses the symmetry feature of A.
    geodesic: bool, if we would like to include the geodesic acceleration when
      solving for the delta_params in every iteration.
    contribution_ratio_threshold: float, the threshold for acceleration/velocity
      ratio. We update the parameters in the algorithm only if the ratio is
      smaller than this threshold value.
    implicit_diff: bool, whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    verbose: bool, whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    jit: whether to JIT-compile the bisection loop (default: "auto").
    unroll: whether to unroll the bisection loop (default: "auto").

  Reference: This algorithm is for finding the best fit parameters based on the
    algorithm 6.18 provided by K. Madsen & H. B. Nielsen in the book
    "Introduction to Optimization and Data Fitting".
  """
  residual_fun: Callable
  maxiter: int = 30
  damping_parameter: float = 1e-6
  scale_invariant: bool = True
  solver: Union[Literal['cholesky', 'inv'], Callable] = solve_cg
  materialize_jac: int = 'semi'
  has_aux: bool = False


  def __post_init__(self):
    if self.has_aux:
      self._fun_with_aux = self.residual_fun
      self._fun = lambda *a, **kw: self._fun_with_aux(*a, **kw)[0]
    else:
      self._fun = self.residual_fun
      self._fun_with_aux = lambda *a, **kw: (self.residual_fun(*a, **kw), None)
    # For geodesic acceleration, we define Hessian of the residual function.
    if self.materialize_jac == 'full' or self.materialize_jac == 'semi':
      self._jac_fun = jax.jacfwd(self._fun, argnums=(0))  

  def update(self, params, *args, **kwargs) -> Tuple:
    """Performs one iteration of the least-squares solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.

    Returns:
      (params, state)
    """

    # Current value of the loss function F=0.5*||f||^2.
    
    residual, aux = self._fun_with_aux(params, *args, **kwargs)
    # For geodesic acceleration, we calculate  jtrpp=JT * r",
    # where J is jacobian and r" is second order directional derivative.
    if self.materialize_jac == 'full':
      jac = self._jac_fun(params, *args, **kwargs)
      jt = jac.T
      jtj = jt @ jac
      gradient = jt @ residual
      damping_factor  = self.damping_parameter * jnp.max(jnp.diag(jtj)) if self.scale_invariant else self.damping_parameter
      jtj_corr = jtj + damping_factor * jnp.identity(params.size) 
      nat_grad = lstsq(jtj_corr, gradient)[0]
    
    elif self.materialize_jac == 'semi':
      jac = self._jac_fun(params, *args, **kwargs)
      jt = jac.T
      jtj = None
      gradient = jt @ residual
      matvec = lambda v: self._semi_jtj_op(v, jt, *args, **kwargs)
      damping_factor = self.damping_parameter * jnp.max(jnp.linalg.norm(jt, axis=0)**2) if self.scale_invariant else self.damping_parameter
      nat_grad = self.solver(matvec, gradient, init=gradient, ridge=damping_factor)
      
    elif self.materialize_jac == 'none':
      jt = None
      jtj = None
      gradient = self._jt_op(params, residual, *args, **kwargs)
      matvec = jit(lambda v: self._jtj_op(params, v, *args, **kwargs))
      jtj_diag = self._jtj_diag_op(params, *args, **kwargs)
      damping_factor = self.damping_parameter * jnp.max(jtj_diag) if self.scale_invariant else self.damping_parameter
      nat_grad = self.solver(matvec, gradient, init=gradient, ridge=damping_factor)
      
    else:
      raise ValueError('materialize_jac should be one of "full", "semi", or "none".')

    # Checking if the dparams satisfy the "sufficiently small" criteria.
    return nat_grad, gradient, 0.5*(residual @ residual)


  def _jt_op(self, params, residual, *args, **kwargs):
    """Product of J^T and residual -- J: jacobian of fun at params."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    _, vjpfun = jax.vjp(fun_with_args, params)
    jt_op_val, = vjpfun(residual)
    return jt_op_val

  def _jtj_op(self, params, vec, *args, **kwargs):
    """Product of J^T.J with vec using vjp & jvp, where J is jacobian of fun at params."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    _, vjpfun = jax.vjp(fun_with_args, params)
    _, jvp_val = jax.jvp(fun_with_args, (params,), (vec,))
    jtj_op_val, = vjpfun(jvp_val)
    return jtj_op_val
  
  def _semi_jtj_op(self, vec, jt, *args, **kwargs):
    """Product of J^T.J with vec using vjp & jvp, where J is jacobian of fun at params."""
    vec_ = jt.T @ vec
    return jt @ vec_

  def _jtj_diag_op(self, params, *args, **kwargs):
    """Diagonal elements of J^T.J, where J is jacobian of fun at params."""
    diag_op = lambda v: v.T @ self._jtj_op(params, v, *args, **kwargs)
    return jax.vmap(diag_op)(jnp.eye(len(params))).T

  def _d2fvv_op(self, primals, tangents1, tangents2, *args, **kwargs):
    """Product with d2f.v1v2."""
    fun_with_args = lambda p: self._fun(p, *args, **kwargs)
    g = lambda pr: jax.jvp(fun_with_args, (pr,), (tangents1,))[1]
    return jax.jvp(g, (primals,), (tangents2,))[1]

