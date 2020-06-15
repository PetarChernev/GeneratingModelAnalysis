from scipy.signal import stft
from high_dim.interface import HighDim
import matplotlib.pyplot as plt
import numpy as np


class Spectrum(HighDim):
    def __init__(self, window=256, step=10):
        super().__init__(dimensions=window, window=window)
        self.step = step
        self.f = None
        self.t = None
        self.z = None

    def represent(self, data):
        """
        Calculates the spectrogram of the passed data and normalizes it by the frequencies to produces the analog of
        an energy spectrum per frequency. This is done because we are more interested in the higher frequencies to
        detect moving between support and resistance levels, while the lowest frequencies represent more long term
        trends, which can be detected in other ways.
        :param data: (pd.Series)
        :return: (np.array[self.window // 2 + 1, len(data)])
        """
        # get the spectrogram
        self.f, self.t, self.z = stft(data, nperseg=self.window, noverlap=self.window - self.step, window='boxcar')
        # multiply frequencies vector piecewise with each column of the spectrogram matrix
        self.z = np.multiply(self.f, self.z.T).T
        return abs(self.z)

    def plot(self):
        amp = np.abs(np.max(self.z))
        plt.pcolormesh(self.t, self.f, np.abs(self.z), vmin=0, vmax=amp)
        plt.show()