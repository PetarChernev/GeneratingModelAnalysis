import pandas as pd
import numpy as np

from .high_dim import HighDim


class ReturnsDistribution(HighDim):
    """
    Represents a price timeseries as the distribution of returns in the specified window.
    """
    def __init__(self, bins, window):
        """
        :param bins: (np.array or int) the bins, with which to build the distribution. This is passed as the "bins"
            argument of pandas.cut
        :param window: (int)
        """
        dimensions = bins if isinstance(bins, int) else len(bins)
        HighDim.__init__(self, dimensions=dimensions, window=window)
        self.bins = bins
        
    def represent(self, data):
        returns = pd.Series(data / data.shift()).dropna() - 1
        day = returns.index[self.window - 1]
        distribution = pd.cut(returns[:day], self.bins).value_counts(sort=False)
        distribution.index = pd.IntervalIndex(distribution.index)
        result = [distribution.values.copy()]
        for i in range(self.window, len(returns)):
            for interval in distribution.index:
                if returns.iloc[i - self.window] in interval:
                    distribution.loc[interval] -= 1
                if returns.iloc[i] in interval:
                    distribution.loc[interval] += 1
            result.append(distribution.values.copy())
        return np.array(result)