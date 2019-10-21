from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
from pyearth import Earth as EarthModel
from pygam import LinearGAM

from statsmodels.tsa.tsatools import lagmat
from fragile.core.states import States
from fragile.core.utils import relativize
from fragile.optimize.env import Function
from autoearth.preprocessing import (apply_normality_test, get_gam_significances,
                                     get_signifficant_correlations,
                                     hernandez_entropy_ratio, probplot_r_squared,
                                     remove_all_outliers)
from autoearth.diagnostics import get_signifficant_features


def calculate_earth_error(X, y, *args, **kwargs):
    earth = EarthModel(*args, **kwargs)
    model = earth.fit(X, y)
    pred = model.predict(X)
    error = pred.flatten() - y.flatten()
    features = get_signifficant_features(model)
    return error, model, features


def evaluate_one_reward(point: np.ndarray, y, data: pd.DataFrame):
    point = np.array(point)
    cols = [c for (c, p) in zip(data.columns, point) if p]
    if len(cols) == 0:
        return 0, (0, 0, 0, 0), True
    selected_features = data[cols]

    error, model, earth_features = calculate_earth_error(X=selected_features, y=y)
    entropy = hernandez_entropy_ratio(error, len(selected_features))
    _, p_is_normal = apply_normality_test(error)
    rsq_qplot = probplot_r_squared(error)
    corrs, corrs_too_big = get_signifficant_correlations(error, nlags=6, alpha=0.05)

    corr_penalty = np.linalg.norm(corrs).sum()
    normal_score = np.log(p_is_normal)
    pac_ratio = corrs_too_big.sum() / float(len(corrs_too_big))
    entropy_score = entropy / np.linalg.norm(error)
    model_score = model.rsq_ * -np.log(model.mse_) * -np.log(model.gcv_)
    sigs = get_gam_significances(X=selected_features.values, y=y)
    # passes_all_tests = not corrs_too_big.any() and p_is_normal > 0.05 and rsq_qplot > 0.95
    # final_cols = [1 if c in earth_features else 0 for c in data.columns]
    scores = entropy_score, -corr_penalty, -pac_ratio, normal_score
    return model_score, scores, not sigs.all()


class Earth(Function):

    def __init__(self, data_path, columns: list, target="ACINETO", processes: int=None,  *args,
                 **kwargs):
        self.target = target
        self.X, self.y = None, None
        self.n_cols = None
        self.columns = columns
        self.load_data(data_path, target=target)
        self.earth = EarthModel(max_degree=1, allow_linear=True,max_terms=50,
                                minspan_alpha=0.000001, thresh=1e-13, penalty=2.)
        super(Earth, self).__init__(function=self._evaluate_model,
                                    shape=(len(self.X.columns),),
                                    low=0,
                                    high=len(self.X.columns) - 1,
                                    *args,
                                    **kwargs)
        self.entropy = 0
        self.rsq = 0
        self.pool = multiprocessing.Pool(processes=processes)
        from fragile.earth.models import EarthSampler
        self._sampler = EarthSampler(columns=self.X.columns, dim=self.shape[0])

    def __repr__(self):

        msg = "Model R2: {:.3f}, entropy: {:.3f}\n".format(self.rsq, self.entropy)
        return msg + super(Earth, self).__repr__()

    def step(self, model_states: States, env_states: States) -> States:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        new_points = model_states.actions
        rewards, ends = self.function(new_points)
        rewards = rewards.flatten()

        last_states = self._get_new_states(new_points, rewards, ends, model_states.n)
        return last_states

    def reset(self, batch_size: int = 1, **kwargs) -> States:
        """
        Resets the environment to the start of a new episode and returns an
        States instance describing the state of the Environment.
        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            States instance describing the state of the Environment. The first
            dimension of the data tensors (number of walkers) will be equal to
            batch_size.
        """
        ends = np.ones(batch_size, dtype=bool)
        new_points = self._sample_init_points(batch_size=batch_size)
        rewards, ends = self.function(new_points)
        print("generating init_solution")
        i = 0
        while ends.all() and i < 0:
            print("trial {}".format(i))
            rewards, ends = self.function(new_points)
            rewards[ends] = 0
            rewards = rewards.flatten()
            new_points = self._sampler.modify_actions(new_points)
            i += 1
        new_states = self._get_new_states(new_points, rewards, ends, batch_size=batch_size)
        return new_states

    @staticmethod
    def gam_significance_analysis(X, y, threshold=0.05):
        gam = LinearGAM(n_splines=5).gridsearch(X, y)
        return np.array(gam.statistics_["p_values"]) > threshold

    def get_gam_significances(self, point):
        cols = [c for (c, p) in zip(self.X.columns, point) if p]
        if len(cols) == 0:
            return None
        X = self.X[cols].values
        sigs = self.gam_significance_analysis(X, self.y)
        return sigs

    def gam_end_condition(self, point):
        sigs = self.get_gam_significances(point)
        return not sigs.all() if sigs is not None else True

    def load_data(self, data_path="data.xls", target="ACINETO"):
        def otras_col(df):
            df["otras"] = df["DDD_ABT_"] - df["CARBA"] - df["FQ"] - df["CEFARES"] - df["CEFAPSEU"]
            return df.drop("DDD_ABT_", axis=1)
        df = pd.read_excel(data_path)
        df = df[self.columns + [target]].copy().set_index("year_mon")
        df = otras_col(df)
        df = remove_all_outliers(df, target=target).dropna(axis=0)

        def get_lags(df, col, lags, max_lags: int = 7):
            data = pd.concat([df[[col]], lagmat(df[[col]], max_lags, use_pandas=True)],
                             axis=1).iloc[max_lags:].copy()
            cols = ["{}.L.{}".format(col, i) if i != 0 else str(col) for i in lags]

            return data.loc[:, cols].copy()

        lags = pd.concat([get_lags(df, c, (range(1, 7) if c != target else range(0, 3)))
                          for c in df.columns], axis=1)
        self.X = lags.drop(target, axis=1)
        self.y = lags[target].values
        self.n_cols = len(self.X.columns)

    def _evaluate_model_slow(self, points: np.ndarray) -> np.ndarray:
        func = partial(evaluate_one_reward, y=np.array(self.y), data=pd.DataFrame(self.X))
        vals = [func(p) for p in points.tolist()]
        return np.array(vals)

    def _evaluate_model(self, points):

        func = partial(evaluate_one_reward, y=np.array(self.y), data=pd.DataFrame(self.X))
        result = self.pool.map(func, points.tolist())#, chunksize=points.shape[0] //
        # multiprocessing.cpu_count())
        model_score, scores, ends = tuple(zip(*result))
        ends = np.zeros_like(np.array(ends))
        scores = np.array(scores)
        score = relativize(np.array(model_score)) ** 0.5
        for i in range(scores.shape[1]):
            score = score * relativize(scores[:, i])
        entropy = np.array(score)
        #ends = score < score.mean()
        self.entropy = max(self.entropy, entropy.max())
        return score, ends

    def _sample_init_points(self, batch_size: int):
        # new_points = (np.random.random(tuple([batch_size]) + self.shape) > 0.5).astype(int)
        new_points = np.zeros(tuple([batch_size]) + self.shape).astype(int)
        flip_values = self.random_state.randint(0, new_points.shape[1], size=len(new_points))
        for i, n in enumerate(flip_values):
            new_points[i, n] = 1
        return new_points


class ACINETO(Earth):

    def __init__(self, *args, **kwargs):
        col_names = ['year_mon', 'CARBA', 'FQ', 'CEFARES', 'DDD_ABT_',
                     'CEFAPSEU', 'ALCBEDDA', 'OccupRat', 'COLACINE']  # 'swabrat'
        kwargs["columns"] = kwargs.get("columns", col_names)
        kwargs["target"] = kwargs.get("target", "ACINETO")
        super(ACINETO, self).__init__(*args, **kwargs)

    def load_data(self, data_path="data.xls", target="ACINETO"):
        def otras_col(df):
            df["otras"] = df["DDD_ABT_"] - df["CARBA"] - df["FQ"] - df["CEFARES"] - df["CEFAPSEU"]
            return df.drop("DDD_ABT_", axis=1)
        df = pd.read_excel(data_path)
        df = df[self.columns + [target]].copy().set_index("year_mon")
        df = otras_col(df)
        df = remove_all_outliers(df, target=target).dropna(axis=0)

        def get_lags(df, col, lags, max_lags: int = 7):
            data = pd.concat([df[[col]], lagmat(df[[col]], max_lags, use_pandas=True)],
                             axis=1).iloc[max_lags:].copy()
            cols = ["{}.L.{}".format(col, i) if i != 0 else str(col) for i in lags]

            return data.loc[:, cols].copy()

        lags = pd.concat([get_lags(df, c, (range(1, 7) if c != target else [0, 1, 2, 6, 11, 12]))
                          for c in df.columns], axis=1)
        self.X = lags.drop(target, axis=1)
        self.y = lags[target].values
        self.n_cols = len(self.X.columns)


class KLEBCRE(Earth):

    def __init__(self, *args, **kwargs):
        col_names = ['year_mon', 'CARBA', 'FQ', 'CEFARES', 'DDD_ABT_',
                     'CEFAPSEU', 'ALCBEDDA', 'OccupRat', 'COLKLEB', 'swabrat']
        kwargs["columns"] = kwargs.get("columns", col_names)
        kwargs["target"] = kwargs.get("target", "KLEBCRE")
        super(KLEBCRE, self).__init__(*args, **kwargs)

    def load_data(self, data_path="data.xls", target="KLEBCRE"):
        def otras_col(df):
            df["otras"] = df["DDD_ABT_"] - df["CARBA"] - df["FQ"] - df["CEFARES"] - df["CEFAPSEU"]
            return df.drop("DDD_ABT_", axis=1)
        df = pd.read_excel(data_path)
        df = df[self.columns + [target]].copy().set_index("year_mon")
        df = otras_col(df)
        #df = remove_all_outliers(df, target=target).dropna(axis=0)

        def get_lags(df, col, lags, max_lags: int = 12):
            data = pd.concat([df[[col]], lagmat(df[[col]], max_lags, use_pandas=True)],
                             axis=1).iloc[max_lags:].copy()
            cols = ["{}.L.{}".format(col, i) if i != 0 else str(col) for i in lags]

            return data.loc[:, cols].copy()

        lags = pd.concat([get_lags(df, c, (range(1, 7) if c != target else [0, 1, 2, 6, 11, 12]))
                          for c in df.columns], axis=1)
        self.X = lags.drop(target, axis=1)
        self.y = lags[target].values
        self.n_cols = len(self.X.columns)


class PSEUDO(Earth):

    def __init__(self, *args, **kwargs):
        col_names = ['year_mon', 'CARBA', 'FQ', 'CEFARES', 'DDD_ABT_',
                     'CEFAPSEU', 'ALCBEDDA', 'OccupRat', 'COLPSEUD']#, 'swabrat']
        kwargs["columns"] = kwargs.get("columns", col_names)
        kwargs["target"] = kwargs.get("target", "PSEUDO")
        super(PSEUDO, self).__init__(*args, **kwargs)

    def load_data(self, data_path="data.xls", target="KLEBCRE"):
        def otras_col(df):
            df["otras"] = df["DDD_ABT_"] - df["CARBA"] - df["FQ"] - df["CEFARES"] - df["CEFAPSEU"]
            return df.drop("DDD_ABT_", axis=1)
        df = pd.read_excel(data_path)
        df = df[self.columns + [target]].copy().set_index("year_mon")
        df = otras_col(df)
        # df = remove_all_outliers(df, target=target).dropna(axis=0)

        def get_lags(df, col, lags, max_lags: int = 7):
            data = pd.concat([df[[col]], lagmat(df[[col]], max_lags, use_pandas=True)],
                             axis=1).iloc[max_lags:].copy()
            cols = ["{}.L.{}".format(col, i) if i != 0 else str(col) for i in lags]

            return data.loc[:, cols].copy()

        lags = pd.concat([get_lags(df, c, (range(1, 7) if c != target else [0, 1, 2, 6, 11, 12]))
                          for c in df.columns], axis=1)
        self.X = lags.drop(target, axis=1)
        self.y = lags[target].values
        self.n_cols = len(self.X.columns)
