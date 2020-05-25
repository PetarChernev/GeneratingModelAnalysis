import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


COLORS = ('g', 'b', 'r')
BUY_HOLD_SELL_MULTIPLIERS = (1, 0, -1)


def get_rolling_distribution(series, bins, window=100):
    day = series.index[window-1]
    distribution = pd.cut(series[:day], bins).value_counts(sort=False)
    distribution.index = pd.IntervalIndex(distribution.index)
    result = [distribution.copy()]
    for i in range(window, len(series)):
        for interval in distribution.index:
            if series.iloc[i - window] in interval:
                distribution.loc[interval] -= 1
            if series.iloc[i] in interval:
                distribution.loc[interval] += 1
        result.append(distribution.copy())
    return pd.DataFrame(result, index=series.index[window-1:])


def test_regime_profit(regimes, prices, buy_hold_sell):
    assert len(regimes) == len(prices)
    positions = [BUY_HOLD_SELL_MULTIPLIERS[buy_hold_sell.index(r)] for r in regimes]
    returns = (prices / prices.shift()).fillna(1)
    profits = positions * (returns - 1) + 1
    return profits.product() - 1


def plot(data, regimes, buy_hold_sell):
    plt.close()
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = ListedColormap([COLORS[i] for i in np.argsort(buy_hold_sell)])
    ax.scatter(data.index,
               data,
               c=regimes,
               cmap=cmap,
               s=3)
    plt.show()


def run_trader(data):
    bins = np.concatenate(([-10], np.linspace(-0.015, 0.015, 6), [10]))
    window = 50
    rolling_average = 20

    data = data.dropna().rolling(rolling_average).mean()
    returns = pd.Series(np.log((data / data.shift()).dropna()))
    rolling_distribution = get_rolling_distribution(returns, bins, window=window)
    assert all(rolling_distribution.sum(axis=1) == window)

    print(f'All data length: {len(rolling_distribution)}')
    split = int(len(rolling_distribution) * 0.8)
    validation = rolling_distribution[split:]
    rolling_distribution = rolling_distribution[:split]
    print(f'Train length: {len(rolling_distribution)}')
    print(f'Validation length: {len(validation)}')

    pca = PCA(n_components=2)
    states = pca.fit(rolling_distribution).transform(rolling_distribution)

    kmeans = KMeans(3).fit(states)
    regimes = kmeans.predict(states)

    transform = pca.components_
    clusters = kmeans.cluster_centers_
    side_bins_count = int(len(transform[0]) / 2)
    returns_profitability_transform = np.arange(-side_bins_count, side_bins_count + 1)
    state_space_basis_profitability = transform.dot(returns_profitability_transform)
    clusters_profitability = clusters.dot(state_space_basis_profitability)
    buy_hold_sell = list(np.flip(np.argsort(clusters_profitability)))

    # plot(data[rolling_distribution.index], regimes, buy_hold_sell)

    states_val = pca.transform(validation)
    regimes_val = kmeans.predict(states_val)

    validation_prices = data[validation.index]

    # plot(validation_prices, regimes_val, buy_hold_sell)

    return test_regime_profit(regimes_val, validation_prices, buy_hold_sell),\
           validation_prices.iloc[-1]/validation_prices.iloc[0] - 1


if __name__ == '__main__':
    df = pd.read_csv('../../daily_data.csv').set_index('TIMESTAMP')
    df.index = pd.to_datetime(df.index)
    outperforms = []
    for isin in df.columns:
        profits, buy_and_hold_profits = run_trader(df[isin])
        print(f"{isin}: trader: {profits} | baseline: {buy_and_hold_profits} | outperform: {profits - buy_and_hold_profits}")
        outperforms.append(profits - buy_and_hold_profits)
    print(f'average outperform: {sum(outperforms) / len(outperforms)}')