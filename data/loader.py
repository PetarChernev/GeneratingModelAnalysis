import pandas as pd
import os
import random

class DataLoader:
    def __init__(self):
        self._data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'daily_data.csv')).set_index('TIMESTAMP')
        self._data.index = pd.to_datetime(self._data.index)

    def get_daily_data(self, isin=None):
        """
        Gets the timeseries of the daily data for the specified ISIN. If not ISIN is passed, data for a random ISIN
        is returned
        :param isin: (Optional[str])
        :return: (pd.Series)
        """
        if isin is None:
            isin = self._data.columns[random.randint(0, len(self._data.columns))]
        return self._data[isin].dropna()