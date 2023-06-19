"""
Macros for types. Specifically for network parameters.

"""

from jaxtyping import Array, Float, PyTree
from typing import Union

# Type of a (possibly) deep fully connected network
# compatible layer sizes are not checked
MlpParams = PyTree[Union[Float[Array, "m n"], Float[Array, "l"]]]

# Exact Type of Parameters of a Shallow Network with variable width
# compatible layer sizes are checked.
ShallowParams = PyTree[Union[
    Float[Array, "n 1"], 
    Float[Array, "n"], 
    Float[Array, "1 n"], 
    Float[Array, "1"]]
    ]
