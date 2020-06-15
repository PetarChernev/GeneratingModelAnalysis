import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from latex import matrix

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
        high_dim = self.high_dim.represent(timeseries)
        # dimensionality reduction algorithms expect the have the features as columns
        # we assume that the number of features is smaller than the number of data points
        # if we have the features as rows, transpose
        if high_dim.shape[0] < high_dim.shape[1]:
            high_dim = high_dim.T
        if hasattr(self.dimensionality_reduction, 'fit_transform'):
            assert callable(self.dimensionality_reduction.fit_transform)
            self.models = self.dimensionality_reduction.fit_transform(high_dim)
        else:
            self.models = self.dimensionality_reduction.fit(high_dim).transform(high_dim)
        return self.models

    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*zip(*self.models), cmap=plt.cm.get_cmap('viridis'), c=np.linspace(0, 1, self.models.shape[0]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title('PCA of distribution evolution, starts from yellow')
        plt.tight_layout()
        plt.show()