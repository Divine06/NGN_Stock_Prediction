"""The four forecasting models plus a no-change baseline.

All models predict the SAME targets: the log return from today's close to each
of the next HORIZON trading days (h = 1..5). A predicted price is then
last_close * exp(predicted_return_h). Predicting returns (not the naira price) is
the fix for the all-SELL problem: price-level models with MinMax scaling, random
forests or RBF-SVR cannot output a price above the highest price seen in training,
so in a rising market they always predict a fall.

Common interface:
    m = Model(); m.fit(frame, train_mask); preds = m.predict(frame, rows=None)
`frame` is the full feature frame for one ticker (see features.build_features).
`predict` returns an array of shape (rows, HORIZON): column h-1 is the forecast
cumulative log return to day h (NaN where the model has no input). `rows` marks
where multi-day forecasts are needed; only ARIMA uses it (to save time).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, TARGET_COLUMNS, feature_mask, target_mask

H = config.HORIZON


class ModelUnavailable(RuntimeError):
    """Raised when a model's library (TensorFlow, statsmodels) is not installed."""


def _train_rows(frame: pd.DataFrame, train_mask: pd.Series) -> np.ndarray:
    """Training rows: features complete and every horizon's outcome known."""
    return train_mask.to_numpy() & feature_mask(frame).to_numpy() & target_mask(frame).to_numpy()


class Forecaster:
    name = "BASE"

    def fit(self, frame: pd.DataFrame, train_mask: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame, rows: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "Forecaster":
        raise NotImplementedError


# --- Baseline -------------------------------------------------------------------
class NaiveForecaster(Forecaster):
    """Random-walk baseline: the price stays where it is (return 0 for every horizon).

    Any model that cannot beat this on the test period is not adding information.
    """
    name = "NAIVE"

    def fit(self, frame, train_mask):
        return self

    def predict(self, frame, rows=None):
        ok = feature_mask(frame).to_numpy()
        return np.where(ok[:, None], 0.0, np.nan) * np.ones((len(frame), H))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "naive.json").write_text(json.dumps({"model": "naive"}))

    @classmethod
    def load(cls, directory):
        return cls()


# --- Tabular models (Random Forest, SVR) -----------------------------------------
class _TabularForecaster(Forecaster):
    filename = "model.joblib"

    def _fit_xy(self, X, Y):
        raise NotImplementedError

    def _predict_x(self, X) -> np.ndarray:
        raise NotImplementedError

    def fit(self, frame, train_mask):
        rows = _train_rows(frame, train_mask)
        X = frame.loc[rows, FEATURE_COLUMNS].to_numpy(float)
        Y = frame.loc[rows, TARGET_COLUMNS].to_numpy(float)
        self._fit_xy(X, Y)
        return self

    def predict(self, frame, rows=None):
        out = np.full((len(frame), H), np.nan)
        ok = feature_mask(frame).to_numpy()
        if ok.any():
            out[ok] = self._predict_x(frame.loc[ok, FEATURE_COLUMNS].to_numpy(float)).reshape(-1, H)
        return out

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "horizon": H}, directory / self.filename)

    @classmethod
    def load(cls, directory):
        obj = cls()
        saved = joblib.load(directory / cls.filename)
        if not isinstance(saved, dict) or saved.get("horizon") != H:
            raise RuntimeError(f"{cls.name} was trained for a different forecast horizon; retrain it")
        obj.model = saved["model"]
        return obj


class RandomForestForecaster(_TabularForecaster):
    """One multi-output forest predicting all horizons together."""
    name = "RF"

    def _fit_xy(self, X, Y):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                           random_state=config.RANDOM_STATE, n_jobs=-1)
        self.model.fit(X, Y)

    def _predict_x(self, X):
        return np.asarray(self.model.predict(X))


class SVRForecaster(_TabularForecaster):
    """RBF Support Vector Regression, one model per horizon.

    The target is standardised too: daily returns are ~0.01, smaller than SVR's
    default epsilon (0.1), which would otherwise make SVR predict a constant.
    """
    name = "SVR"

    @staticmethod
    def _one():
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
            transformer=StandardScaler())

    def _fit_xy(self, X, Y):
        self.model = [self._one().fit(X, Y[:, h]) for h in range(H)]

    def _predict_x(self, X):
        return np.column_stack([m.predict(X) for m in self.model])


# --- ARIMA ------------------------------------------------------------------------
class ARIMAForecaster(Forecaster):
    """ARIMA(p,0,q) with constant on daily log returns.

    This is equivalent to ARIMA(p,1,q) on log prices. (p, q) is picked from
    {0,1,2} x {0,1,2} by AIC on the training period. Segment boundaries are left
    as NaN; the state-space ARIMA treats them as missing values. Multi-day forecasts
    are dynamic (each day uses the previous days' forecasts), summed into
    cumulative returns.
    """
    name = "ARIMA"
    ORDERS = [(p, 0, q) for p in range(3) for q in range(3)]

    @staticmethod
    def _arima():
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailable("statsmodels is not installed (pip install statsmodels)") from exc
        return ARIMA

    def fit(self, frame, train_mask):
        ARIMA = self._arima()
        positions = np.flatnonzero(train_mask.to_numpy())
        y = frame["logret"].to_numpy(float)[: positions[-1] + 1]
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.ORDERS:
                try:
                    res = ARIMA(y, order=order, trend="c").fit()
                except Exception:
                    continue
                if np.isfinite(res.aic) and (best is None or res.aic < best[1].aic):
                    best = (order, res)
        if best is None:
            raise RuntimeError("no ARIMA order could be fitted")
        self.order = best[0]
        self.params = np.asarray(best[1].params, dtype=float)
        self.aic = float(best[1].aic)
        return self

    def predict(self, frame, rows=None):
        ARIMA = self._arima()
        y = frame["logret"].to_numpy(float)
        n = len(y)
        valid = feature_mask(frame).to_numpy()
        out = np.full((n, H), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y, order=self.order, trend="c").filter(self.params)  # fixed params, no refit
            one_step = np.asarray(res.predict(), dtype=float)   # forecast of y[t] made at t-1
            nxt = float(np.asarray(res.forecast(1))[0])        # forecast beyond the last row
            out[:, 0] = np.append(one_step[1:], nxt)            # row t -> forecast of y[t+1]
            if H > 1:
                want = np.zeros(n, bool) if rows is None else np.asarray(rows, bool).copy()
                want[-1] = True                                 # the live forecast always needs the path
                for t in np.flatnonzero(want & valid):
                    if t == n - 1:
                        steps = np.asarray(res.forecast(H), dtype=float)
                    else:
                        steps = np.asarray(res.predict(start=t + 1, end=t + H, dynamic=True), dtype=float)
                    out[t] = np.cumsum(steps[:H])
        out[~valid] = np.nan
        return out

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "arima.json").write_text(json.dumps(
            {"order": list(self.order), "params": self.params.tolist(), "aic": self.aic}))

    @classmethod
    def load(cls, directory):
        spec = json.loads((directory / "arima.json").read_text())
        obj = cls()
        obj.order = tuple(spec["order"])
        obj.params = np.asarray(spec["params"], dtype=float)
        obj.aic = spec.get("aic")
        return obj


# --- LSTM -------------------------------------------------------------------------
def make_windows(n_rows: int, valid: np.ndarray, segment: np.ndarray, window: int) -> np.ndarray:
    """Row positions i whose previous `window` rows (inclusive) are all valid and in one segment."""
    ends = []
    run = 0
    for i in range(n_rows):
        if valid[i] and (i == 0 or segment[i] == segment[i - 1]) and run > 0:
            run += 1
        elif valid[i]:
            run = 1
        else:
            run = 0
        if run >= window:
            ends.append(i)
    return np.asarray(ends, dtype=int)


class LSTMForecaster(Forecaster):
    """LSTM over the last WINDOW days of features -> returns to days 1..HORIZON (one output each).

    Features are standardised with statistics from the training rows only;
    each horizon's target is divided by its training standard deviation.
    """
    name = "LSTM"

    def __init__(self, window: int = config.WINDOW, epochs: int = config.LSTM_EPOCHS):
        self.window = window
        self.epochs = epochs

    @staticmethod
    def _keras():
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailable("TensorFlow is not installed (pip install tensorflow)") from exc
        return tf, keras

    def _windows(self, frame):
        X = frame[FEATURE_COLUMNS].to_numpy(float)
        Xs = (X - self.mean_) / self.std_
        ends = make_windows(len(frame), feature_mask(frame).to_numpy(), frame["segment"].to_numpy(), self.window)
        if len(ends) == 0:
            return ends, np.empty((0, self.window, len(FEATURE_COLUMNS)))
        return ends, np.stack([Xs[i - self.window + 1: i + 1] for i in ends])

    def fit(self, frame, train_mask):
        tf, keras = self._keras()
        tf.keras.utils.set_random_seed(config.RANDOM_STATE)
        rows = train_mask.to_numpy() & feature_mask(frame).to_numpy()
        X_train = frame.loc[rows, FEATURE_COLUMNS].to_numpy(float)
        self.mean_ = X_train.mean(axis=0)
        self.std_ = np.where(X_train.std(axis=0) > 0, X_train.std(axis=0), 1.0)

        ends, Xw = self._windows(frame)
        targets = frame[TARGET_COLUMNS].to_numpy(float)
        keep = _train_rows(frame, train_mask)[ends]
        Xw, Y = Xw[keep], targets[ends][keep]
        sd = Y.std(axis=0)
        self.y_scale_ = np.where(sd > 0, sd, 1.0)

        model = keras.Sequential([
            keras.layers.Input(shape=(self.window, len(FEATURE_COLUMNS))),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(H),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.Huber())
        model.fit(Xw, Y / self.y_scale_, epochs=self.epochs, batch_size=64, validation_split=0.15,
                  shuffle=True, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                           restore_best_weights=True)])
        self.model = model
        return self

    def predict(self, frame, rows=None):
        out = np.full((len(frame), H), np.nan)
        ends, Xw = self._windows(frame)
        if len(ends):
            out[ends] = np.asarray(self.model.predict(Xw, verbose=0)).reshape(len(ends), H) * self.y_scale_
        return out

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(str(directory / "lstm.keras"))
        joblib.dump({"window": self.window, "mean": self.mean_, "std": self.std_, "horizon": H,
                     "y_scale": self.y_scale_, "features": FEATURE_COLUMNS}, directory / "lstm_meta.joblib")

    @classmethod
    def load(cls, directory):
        _, keras = cls._keras()
        meta = joblib.load(directory / "lstm_meta.joblib")
        if list(meta["features"]) != FEATURE_COLUMNS or meta.get("horizon") != H:
            raise RuntimeError("LSTM was trained with different features or horizon; retrain it")
        obj = cls(window=meta["window"])
        obj.mean_, obj.std_, obj.y_scale_ = meta["mean"], meta["std"], np.asarray(meta["y_scale"])
        obj.model = keras.models.load_model(str(directory / "lstm.keras"))
        return obj


MODEL_CLASSES: dict[str, type[Forecaster]] = {
    "NAIVE": NaiveForecaster,
    "ARIMA": ARIMAForecaster,
    "RF": RandomForestForecaster,
    "SVR": SVRForecaster,
    "LSTM": LSTMForecaster,
}


def model_dir(ticker: str, name: str) -> Path:
    return config.MODELS_DIR / ticker.upper() / name.lower()
