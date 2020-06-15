def smooth(data, window):
    return data.rolling(window).mean()


def gaussian_smooth(data, window, std=0.5):
    return data.rolling(window, win_type='gaussian').mean(std=std)


def get_returns(data):
    return (data / data.shift()).dropna() - 1