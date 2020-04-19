import pandas as pd
import numpy as np


class ReturnsDistribution:
    def __init__(self, bins, window):
        self.bins = bins
        self.window = window
        
    def __call__(self, prices):
        returns = pd.Series(np.log((prices / prices.shift()).dropna()))
        day = returns.index[self.window - 1]
        distribution = pd.cut(returns[:day], self.bins).value_counts(sort=False)
        distribution.index = pd.IntervalIndex(distribution.index)
        result = [distribution.copy()]
        for i in range(self.window, len(returns)):
            for interval in distribution.index:
                if returns.iloc[i - self.window] in interval:
                    distribution.loc[interval] -= 1
                if returns.iloc[i] in interval:
                    distribution.loc[interval] += 1
            result.append(distribution.copy())
        return pd.DataFrame(result, index=returns.index[self.window - 1:])
