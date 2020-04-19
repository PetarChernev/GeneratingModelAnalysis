class ModelRepresentation:
    def __init__(self, rolling_func, dimensionality_reduction):
        self.rolling_func = rolling_func
        self.dimensionality_reduction = dimensionality_reduction

    def represent(self, timeseries):
        return self.dimensionality_reduction.fit_transform(self.rolling_func(timeseries))