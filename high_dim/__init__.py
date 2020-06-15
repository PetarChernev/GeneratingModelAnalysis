"""
This package defines the high dimensional representations of the models.
It defines an abstract interface class, which exposes the represent(data: pd.Series) -> np.array function.
Concrete implementations consist of different rolling computations, which produce a timeseries of high dimensional
representation of a rolling window of the data.
"""

from .interface import HighDim
from .distribution import Distribution
from .spectrum import Spectrum