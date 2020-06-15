from high_dim import Spectrum
from dimensionality_reduction.dtw_and_mds import DynamicTimeWarpAndMDS
from model_repr import ModelRepresentation
from data.loader import DataLoader
from data.preprocess import gaussian_smooth, get_returns

loader = DataLoader()
data = loader.get_daily_data('CH0012032048')
data = get_returns(gaussian_smooth(data, window=50, std=0.9))


high_dim = Spectrum(window=50)

representation = ModelRepresentation(high_dim=high_dim,
                                     dimensionality_reduction=DynamicTimeWarpAndMDS(n_components=3))

representation.represent(data)

high_dim.plot()
representation.plot()