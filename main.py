import numpy as np
import pandas as pd

from high_dim import ReturnsDistribution
from sklearn.decomposition import PCA
from model_repr import ModelRepresentation
from model import JumpModels, SinModel

model = JumpModels([SinModel(20), SinModel(100)], [200])

bins = np.concatenate(([-10], np.linspace(-0.015, 0.015, 6), [10]))
high_dim = ReturnsDistribution(bins=10, window=100)
representation = ModelRepresentation(high_dim=high_dim,
                                     dimensionality_reduction=PCA(n_components=3))

data = pd.read_csv('data/daily_data.csv')['CH0012032048']
# data = model.get_series(400)
representation.represent(data)

representation.plot()