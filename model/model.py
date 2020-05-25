import abc
import pandas as pd


class Model(abc.ABC):
    """
    An abstract class that represents a generating model.
    """
    def __init__(self):
        self.series = []

    @abc.abstractmethod
    def generate(self, n=None):
        """
        Returns a generator that generates n data points. If n is not passed, generates data infinitely
        :param n: (Optional[int]) the length of data to generate.
        :return: (Generator[float, None, None])
        """
        pass

    def get_series(self, n):
        """
        Generates n data points and returns them as a list
        :param n: (int)
        :return: (List[float])
        """
        series = [p for p in self.generate(n)]
        return pd.Series(series)


class JumpModels(Model):
    """
    A composition of multiple Models. The jumps attribute holds the jump moments, at which the generating model
    switches to the next one in the models attribute.
    """
    def __init__(self, models, jumps):
        Model.__init__(self)
        self.models = models
        self.jumps = jumps
        self.t = 0

    def generate(self, n=None):
        jumps = [0] + self.jumps
        for j in range(1, len(jumps)):
            # calculate how much time is spent in one model between jumps
            model_stay_time = jumps[j] - jumps[j - 1]
            # generate from said model for this amount of time
            for p in self.models[j-1].generate(model_stay_time):
                self.t += 1
                if self.t >= n:
                    return
                yield p
        # generate from the last model
        for p in self.models[-1].generate(n - jumps[-1]):
            yield p
