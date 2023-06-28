from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker

# Useful type aliases
InputType = Float[Array, "d"]
MatrixType = Float[Array, "Pdim Pdim"]
ParamType = PyTree

# Type aliases for functions
ModelType = Callable[[PyTree, InputType], Float[Array, ""]]
ParamModelType = Callable[[InputType], Float[Array, ""]]

# Type aliases for functionals and operators
TrafoType = Callable[[ParamModelType, Callable[[InputType], PyTree],], Callable[[InputType], PyTree]]
GramFuncType = Callable[[PyTree, InputType], MatrixType]
GramEvalType = Callable[[PyTree], MatrixType]
ParamType = Callable[[PyTree, PyTree], PyTree]

# Nd arrays
FloatArrayN1 = Float[Array, "N 1"]
FloatArrayN2 = Float[Array, "N 2"]
FloatArrayN3 = Float[Array, "N 3"]
FloatArrayNd = Float[Array, "N d"]
PointType = Float[Array, "d"]