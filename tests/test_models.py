import numpy as np
import pandas as pd

import _setup  # noqa: F401
from backend.ml.features import build_features, feature_mask
from backend.ml.history import load_history
from backend.ml.models import (ARIMAForecaster, LSTMForecaster, RandomForestForecaster,
                       SVRForecaster, make_windows)


def _frame(ticker="SYNUP1"):
    f = build_features(load_history(ticker, include_live=False))
    usable = feature_mask(f) & f["target"].notna()
    cut = f.index[usable.to_numpy()][int(usable.sum() * 0.8)]
    return f, usable & (f.index < cut), usable & (f.index >= cut)


def test_make_windows_respects_validity_and_segments():
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], bool)
    seg = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    assert list(make_windows(10, valid, seg, 3)) == [2, 3, 9]


def test_tabular_models_predict_returns_not_prices():
    f, tr, te = _frame()
    for cls in (RandomForestForecaster, SVRForecaster):
        p = cls().fit(f, tr).predict(f)
        assert p.shape == (len(f), 5)                           # days 1..5 ahead
        assert np.isfinite(p[te.to_numpy()]).all()
        assert np.abs(p[te.to_numpy(), 0]).max() < 0.10         # daily return scale, inside NGX band
        assert np.std(p[te.to_numpy(), 0]) > 1e-5               # not a constant (SVR target scaling)


def test_rf_learns_the_planted_signal():
    f, tr, te = _frame("SYNUP2")
    p = RandomForestForecaster().fit(f, tr).predict(f)[te.to_numpy()]
    actual = f["target"].to_numpy()[te.to_numpy()]
    ok = np.isfinite(actual)
    assert np.corrcoef(p[ok, 0], actual[ok])[0, 1] > 0.05


def test_arima_forecast_for_row_t_targets_t_plus_1():
    f, tr, te = _frame("SYNFLAT2")
    m = ARIMAForecaster().fit(f, tr)
    rows = te.to_numpy()
    p = m.predict(f, rows=rows)
    tgt = f["target"].to_numpy()
    ok = rows & np.isfinite(tgt)
    # AR structure in the data: forecast for t+1 should correlate with the actual t+1 return
    assert np.corrcoef(p[ok, 0], tgt[ok])[0, 1] > 0.05
    assert np.isfinite(p[-1]).all()                    # 5-day path beyond the last row exists
    assert np.isfinite(p[rows]).all()                  # multi-day paths for the requested rows
    i = np.flatnonzero(rows)[10]
    assert np.isclose(p[i, 0], m.predict(f)[i, 0])     # day 1 is the same either way


def test_arima_save_load_roundtrip(tmp_path):
    f, tr, _ = _frame("SYNFLAT2")
    m = ARIMAForecaster().fit(f, tr)
    m.save(tmp_path)
    m2 = ARIMAForecaster.load(tmp_path)
    assert np.allclose(m.predict(f), m2.predict(f), equal_nan=True)


def _install_keras_standin():
    """Linear model with Keras' interface: exercises LSTMForecaster's windowing,
    scaling and save/load code when TensorFlow is not installed."""
    import pickle
    import sys
    import types

    class Sequential:
        def __init__(self, layers):
            self.layers = layers

        def compile(self, **kw):
            pass

        def fit(self, X, y, **kw):
            assert X.ndim == 3 and len(X) == len(y) and np.isfinite(X).all() and np.isfinite(y).all()
            A = np.c_[X[:, -1, :], np.ones(len(X))]
            self.w = np.linalg.lstsq(A, y, rcond=None)[0]

        def predict(self, X, verbose=0):
            return (np.c_[X[:, -1, :], np.ones(len(X))] @ self.w).reshape(len(X), -1)

        def save(self, path):
            assert isinstance(path, str) and path.endswith(".keras")
            with open(path, "wb") as fh:
                pickle.dump(self.w, fh)

    def load_model(path):
        m = Sequential([])
        with open(path, "rb") as fh:
            m.w = pickle.load(fh)
        return m

    noop = lambda *a, **k: None
    keras = types.SimpleNamespace(
        Sequential=Sequential,
        layers=types.SimpleNamespace(Input=noop, LSTM=noop, Dropout=noop, Dense=noop),
        optimizers=types.SimpleNamespace(Adam=noop), losses=types.SimpleNamespace(Huber=noop),
        callbacks=types.SimpleNamespace(EarlyStopping=noop),
        models=types.SimpleNamespace(load_model=load_model),
        utils=types.SimpleNamespace(set_random_seed=noop))
    tf = types.ModuleType("tensorflow")
    tf.keras = keras
    sys.modules["tensorflow"] = tf


def test_lstm_pipeline(tmp_path):
    f, tr, te = _frame()
    try:
        import tensorflow  # noqa: F401
        real = True
    except ImportError:
        _install_keras_standin()
        real = False
    m = LSTMForecaster(epochs=3).fit(f, tr)
    p = m.predict(f)
    rows = te.to_numpy()
    assert p.shape[1] == 5 and np.isfinite(p[rows]).mean() > 0.9
    assert np.nanmax(np.abs(p[rows, 0])) < 0.10
    m.save(tmp_path)
    p2 = LSTMForecaster.load(tmp_path).predict(f)
    assert np.allclose(p, p2, equal_nan=True, atol=1e-6)
    if not real:
        import sys
        del sys.modules["tensorflow"]  # do not leak the stand-in into other tests
