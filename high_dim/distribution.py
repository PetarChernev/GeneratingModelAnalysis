import pandas as pd
import numpy as np

from .interface import HighDim


class Distribution(HighDim):
    """
    Represents a price timeseries as the distribution of returns in the specified window.
    """
    END_BIN_PERCENTAGE = 2

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
        day = data.index[self.window - 1]
        distribution = pd.cut(data[:day], self.bins).value_counts(sort=False)
        distribution.index = pd.IntervalIndex(distribution.index)
        result = [distribution.values.copy()]
        for i in range(self.window, len(data)):
            for interval in distribution.index:
                if data.iloc[i - self.window] in interval:
                    distribution.loc[interval] -= 1
                if data.iloc[i] in interval:
                    distribution.loc[interval] += 1
            result.append(distribution.values.copy())
        return np.array(result)

    @staticmethod
    def get_bins(data, n_bins, boundary_percentile=1):
        """
        Calculates an array of bins of length n_bins, such that boundary_percentile percent of the returns of the passed
        data falls in the first and the last bin respectively.
        :param data: (pd.Series)
        :param n_bins: (int)
        :param boundary_percentile: (int)
        :return: (np.array)
        """
        ends = np.percentile(data, [boundary_percentile, 100 - boundary_percentile])
        return np.concatenate(([1.01 * min(data)], np.linspace(*ends, n_bins), [1.01 * max(data)]))
