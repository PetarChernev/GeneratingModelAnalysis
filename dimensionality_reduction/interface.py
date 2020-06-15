import abc


class DimensionalityReduction(abc.ABC):
    def __init__(self, n_components):
        self.n_components = n_components

    @abc.abstractmethod
    def fit_transform(self, data):
        pass

    @staticmethod
    def not_fit_error():
        return Exception('Trying to use transform(data) method on a DimensionalityReduction object, which has not been '
                         'fit. Call fit(data) method first.')