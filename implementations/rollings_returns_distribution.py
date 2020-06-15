from high_dim import Distribution
from sklearn.decomposition import PCA
from model_repr import ModelRepresentation
from data.loader import DataLoader
from data.preprocess import smooth, get_returns

loader = DataLoader()
data = loader.get_daily_data('CH0012032048')
data = get_returns(smooth(data, window=20))

bins = Distribution.get_bins(data, 10, 1)

high_dim = Distribution(bins=bins, window=100)
representation = ModelRepresentation(high_dim=high_dim,
                                     dimensionality_reduction=PCA(n_components=3))

representation.represent(data)

representation.plot()