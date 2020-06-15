import abc


class HighDim(abc.ABC):
    """
    An interface for the rolling high dimensional representation calculators which take in data and produce
    high dimensional representation of the model, which generated the data.
    """
    def __init__(self, dimensions: int, window: int):
        """
        :param dimensions: (int) the dimension of the representation
        :param window; (int) the window of the rolling calculation
        """
        self.dim = dimensions
        self.window = window

    @abc.abstractmethod
    def represent(self, data):
        """
        Takes in a timeseries and does a rolling window computation. For each window, it generates a high dimensional
        vector. Returns the timeseries of these vectors with length len(data) - window
        :param data: (pd.Series)
        :return: (np.ndarray[len(data) - self.window, self.dimensions])
        """
        pass
