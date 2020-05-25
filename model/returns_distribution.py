from .model import Model


class ReturnsDistributionModel(Model):
    """
    A model which draws the next price based on a predefined distribution of the returns.
    """
    def __init__(self, distribution):
        Model.__init__(self)
        self.distribution = distribution

    def generate(self):
        raise NotImplementedError()