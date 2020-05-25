import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from high_dim import HighDim


class ModelRepresentation:
    """
    Uses a composition of a high dimensional representation and dimensionality reduction to transform a one
    dimensional timeseries into a timeseries of the estimation of the models, which generated said timeseries,
    in model state space.
    """
    def __init__(self, high_dim: HighDim, dimensionality_reduction):
        self.high_dim = high_dim
        self.dimensionality_reduction = dimensionality_reduction
        self.models = None

    def represent(self, timeseries):
        if hasattr(self.dimensionality_reduction, 'fit_transform'):
            assert callable(self.dimensionality_reduction.fit_transform)
            self.models = self.dimensionality_reduction.fit_transform(self.high_dim.represent(timeseries))
        else:
            rolling = self.high_dim.represent(timeseries)
            self.models = self.dimensionality_reduction.fit(rolling).transform(rolling)
        return self.models

    def plot(self):
        print(self.dimensionality_reduction.components_)

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*zip(*self.models), cmap=plt.cm.get_cmap('viridis'), c=np.linspace(0, 1, self.models.shape[0]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title('PCA of distribution evolution, starts from yellow')
        plt.show()