from .model import Model
from math import sin, pi


class SinModel(Model):
    """
    A simple model that generates a sinusoidal timeseries.
    """
    def __init__(self, period, phi_0=0):
        Model.__init__(self)
        self.phi = phi_0
        self.velocity = 2 * pi / period

    def generate(self, n=None):
        for _ in range(n):
            angle = sin(self.phi)
            self.phi += self.velocity
            yield angle + 2
