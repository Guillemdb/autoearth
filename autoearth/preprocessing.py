from typing import List

import numpy as np
import pandas as pd
from pygam import LinearGAM
import ruptures
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import pacf, acf


def get_breakpoints(
    df: pd.DataFrame, model: str = "rbf", min_size: int = 5, jump: int = 1, pen: int = 2
) -> List[int]:
    """
    Calculate the breakpoints of a time series or a group of time series using binary segmentation.

    For more info http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/binseg.html.

    :param df: DataFrame containing the target time series as columns.
    :param model: segment model, [“l1”, “l2”, “rbf”,…]. Not used if 'custom_cost' is not None.
    :param min_size: minimum segment length. Defaults to 5 samples.
    :param jump: subsample (one every jump points). Defaults to 1 sample.
    :param pen:  penalty value (>0).
    :return: list containing the indexes where breakpoints happen.
    """
    signal = (df.values - df.values.mean(axis=0)) / df.values.std(axis=0)
    algo = ruptures.Binseg(model=model, min_size=min_size, jump=jump).fit(signal)
    result = algo.predict(pen=pen)
    return result


def cooks_distance(df, column, target):
    """
    Calculate cooks distance of the target column of the DataFrame with \
    respect to the target column.

    :param df: DataFrame containing at least two columns for the least squares regression.
    :param column: Column name of the values to be used as `x` for the cooks distance.
    :param target: Column name of the target variable for the regression.
    :return: Series containing the cooks distances for every value in column.
    """
    m = ols('{} ~ {}'.format(column, target), df).fit()
    infl = m.get_influence()
    sm_fr = infl.summary_frame()
    cooksd = sm_fr["cooks_d"]
    cooksd.name = column
    return cooksd.copy()


def get_outliers_df(df, target, threshold: float = 5.):
    """
    Return a DataFrame of booleans with same shape as df containing True \
    in all the values where the Cooks distance with respect to a target variable \
    is greater than a threshold.
    """
    data = []
    for col in df.columns:
        if col != target:
            cooks_d = cooks_distance(df, column=col, target=target)
            mask = cooks_d >= threshold * cooks_d.mean()
            data.append(mask.copy())
        else:
            x = pd.Series(name=col, data=np.zeros(len(df.index), dtype=bool), index=df.index)
            data.append(x.copy())
    return pd.concat(data, axis=1)


def filter_outliers(df, target="ACINETO", cooks_max: float = 5):
    def fill_missing(df):
        """
        Calculate the mean of a sequence of length \
        7 ignoring the central value.
        """
        vals = df[:3][1:].mean()
        return vals
    mask = get_outliers_df(df, target, threshold=cooks_max)
    fill = df.rolling(7, center=True, min_periods=1).apply(fill_missing)
    adjusted = df.where(~mask.values, fill).copy()
    return adjusted.dropna()


def remove_outliers(df, target="ACINETO", pen: int = 2):
    breakpoints = [0] + get_breakpoints(df, pen=pen) + [-1]
    filtered = []
    for i in range(len(breakpoints) - 1):
        mini_df = df.iloc[breakpoints[i]: breakpoints[i + 1]]
        if len(mini_df) > 0:
            filtered.append(filter_outliers(mini_df, target=target))
    return pd.concat(filtered, axis=0)


def remove_all_outliers(df, target="ACINETO", pen: int = 2):
    filtered = []
    for col in df.columns:
        if col != target:
            one_df = remove_outliers(df[[col, target]], target=target, pen=pen)
            filtered.append(one_df[[col]])
        else:
            filtered.append(df[[target]])
    return pd.concat(filtered, axis=1)


def get_gam_significances(X, y, threshold=0.05):
    gam = LinearGAM(n_splines=5).gridsearch(X, y, progress=False)
    sigs = np.array(gam.statistics_["p_values"]) > threshold
    return sigs


def get_signifficant_correlations(series: np.ndarray, nlags: int = 7,
                                  alpha: float = 0.05, partial: bool = True):
    corr_func = pacf if partial else acf
    vals, conf = corr_func(series, nlags=nlags, alpha=alpha)
    return vals, vals >= (conf[:, 1] - vals)


def apply_normality_test(error: np.ndarray):
    return stats.normaltest(error)


def probplot_r_squared(error: np.ndarray):
    _, (__, ___, r_norm) = stats.probplot(error, plot=None, fit=True)
    return r_norm ** 2


def hernandez_entropy_ratio(x: np.ndarray, n: int):
    max_entropy = (2 - (1 / n) ** (1 / n)) ** n
    abs_error = np.abs(x)
    norm = abs_error / abs_error.sum()
    entropy = np.prod(2 - norm ** norm)
    entropy_ratio = entropy / max_entropy
    return entropy_ratio
