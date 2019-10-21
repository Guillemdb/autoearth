from functools import partial

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from pyearth import Earth as EarthModel


def get_model_data(model):
    data = []
    i = 0
    for bf in model.basis_:
        data.append([str(bf), not bf.is_pruned()] + [
                      '%g' % model.coef_[c, i] if not bf.is_pruned() else
                      'None' for c in range(model.coef_.shape[0])])
        if not bf.is_pruned():
            i += 1
    return data


def get_sign(name):
    parts = name.split("-")
    try:
        val = float(parts[0][2:])
        return -1
    except Exception as e:
        try:
            val = float(parts[1][:-1])
            return 1
        except:
            return None


def get_basis_data(basis):
    name = str(basis)
    # col, threshold, sign, linear
    if name == "(Intercept)":
        return name, np.nan, np.nan, False
    if "h" not in name:  # linear term
        return name, np.nan, 0, True
    # hinge function
    sign = get_sign(name)
    sp = name.split("-")
    x_name = sp[0][2:] if sign == 1 else sp[1][:-1]
    threshold = sp[1][:-1] if sign == 1 else sp[0][2:]
    try:
        threshold = float(threshold)
    except:
        threshold, x_name = x_name, threshold
        threshold = float(threshold)
    return x_name, threshold, sign, False


def _name_to_data(name, columns):
    cols = {"x{}".format(i):c for i, c in enumerate(columns)}
    # col, threshold, sign, linear
    if name == "(Intercept)":
        return name, np.nan, np.nan, False
    if "h" not in name: # linear term
        return cols.get(name, name), np.nan, 0, True
    # hinge function
    sign = get_sign(name)
    sp = name.split("-")
    x_name = sp[0][2:] if sign == 1 else sp[1][:-1]
    threshold = sp[1][:-1] if sign == 1 else sp[0][2:]
    try:
        threshold = float(threshold)
    except:
        threshold, x_name = x_name, threshold
        threshold = float(threshold)
    return cols.get(x_name, x_name), threshold, sign, False


def get_signifficant_features(model):
    return [get_basis_data(bf)[0] for bf in model.basis_ if not bf.is_pruned()]


def model_summary(model, columns):
    data = get_model_data(model)
    col_names = ["feature", "threshold", "sign_thres", "is_linear_term", "slope"]
    return pd.DataFrame([_name_to_data(x[0], columns) + tuple((x[-1],))
                         for x in data if x[1]], columns=col_names)


def hinge_from_row(row):
    def hinge(x, thres, slope, sign):
        if sign == 1:
            return slope * max(0, x - thres)
        elif sign == -1:
            return slope * max(thres - x, 0)
        elif sign == 0:
            return slope * x
        return 0
    h_f = partial(hinge, thres=float(row["threshold"]), slope=float(row["slope"]),
                  sign=row["sign_thres"])
    return h_f


def earth_basis(df):
    funcs = []
    for ix, row in df.iterrows():
        if row.loc["feature"] == "(Intercept)":
            fun = lambda x: float(row["slope"])
        else:
            fun = hinge_from_row(row)
        funcs.append(fun)

    def _apply_basis(vector, funcs):
        return np.array([sum([fun(v) for fun in funcs]) for v in vector])

    return partial(_apply_basis, funcs=funcs)


def plot_feature(feat_df, df, n_vals: int = 100):
    feat_name = feat_df[["feature"]].values[-1]
    x = np.linspace(df[feat_name].min(), df[feat_name].max(), n_vals)
    hinge = earth_basis(feat_df)
    vals = np.array([hinge(i) for i in x])
    plt.plot(x, vals)
    plt.show()


def get_basis_vals(feat_df, df, n_vals: int = 100):
    feat_name = feat_df[["feature"]].values[-1]
    x = np.linspace(df[feat_name].min(), df[feat_name].max(), n_vals)
    hinge = earth_basis(feat_df)
    vals = np.array([hinge(i) for i in x])
    return x, vals


def get_feature_data(summary, df):
    feat_names = list(sorted(list(set(summary["feature"].values.tolist()) - set(["(Intercept)"]))))
    bdict = {}
    inter = summary["feature"] == "(Intercept)"
    for fn in feat_names:
        ix = summary["feature"] == fn
        cond = np.logical_or(ix, inter)
        feat_df = summary[cond]
        x, y = get_basis_vals(feat_df, df)
        bdict[fn] = {"x": x, "y": y}
    return bdict


def plot_thresholds(summary, df):
    feat_d = get_feature_data(summary, df)
    plot_d = {col: hv.Curve(feat_d[col]).opts(shared_axes=False, normalize=False,
                                              title=col, tools=["hover"])
              for col in feat_d.keys()}
    return hv.NdLayout(plot_d, kdims="Serie")


class Diagnostics:

    def __init__(self, env, features, *args, **kwargs):
        self.env = env
        self.solution = features
        self.data = env.X.loc[:, features.astype(bool)].copy()
        self.y = self.env.y
        self.model = EarthModel(*args, **kwargs)
        self.y_pred = None
        self.error = None
        self._fit()

    def _fit(self):
        self.model.fit(self.data, self.y)
        self.y_pred = self.model.predict(self.data)
        self.error = (self.y_pred.flatten() - self.env.y.flatten())

    def summary(self):
        return model_summary(self.model, self.data.columns).sort_values("feature")

    def plot_thresholds(self):
        return plot_thresholds(self.summary(), self.data)

    def plot_autocorrelations(self):
        from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
        _ = plot_pacf(self.error)
        _ = plot_acf(self.error)

    def plot_qq(self):
        fig, ax = plt.subplots()
        _, (slope, intercept, r_norm) = scipy.stats.probplot(self.error, plot=ax, fit=True)
        print("R squared {:.4f}".format(r_norm ** 2))

    def plot_pred(self):
        df = pd.DataFrame({"predicted": self.y_pred, "True value": self.y}, index=self.data.index)
        return df.hvplot().opts(title="Model prediction for {}".format(self.env.target))

    def score(self):
        mse, gvc, rsq, grsq = self.model.mse_, self.model.gcv_, self.model.rsq_, self.model.grsq_
        msg = "MSE: {:.4f}, GCV: {:.4f}, RSQ:{:.4f}, GRSQ: {:.4f}".format(mse, gvc, rsq, grsq)
        print(msg)
