from .interface import DimensionalityReduction
from fastdtw import fastdtw
from sklearn.manifold import smacof
from scipy.spatial.distance import pdist, squareform


class DynamicTimeWarpAndMDS(DimensionalityReduction):
    def __init__(self, n_components):
        super().__init__(n_components)
        #self.transformation = Reduction(n_components=n_components, affinity="precomputed")

    @staticmethod
    def metric(a, b):
        return fastdtw(a, b)[0]

    def fit_transform(self, data):
        dist_matrix = squareform(pdist(data, metric=self.metric))
        return smacof(dist_matrix)
