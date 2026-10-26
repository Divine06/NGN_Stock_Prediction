"""Minimal stand-in for statsmodels' ARIMA, used ONLY when statsmodels is not
installed in the test environment. It fits an AR(1) with constant and exposes
the same calls the package uses (fit, filter, aic, params, predict, forecast),
so the alignment logic in ml.models.ARIMAForecaster can be tested."""
import numpy as np


class _Results:
    def __init__(self, y, params, aic):
        self.y, self.params, self.aic = y, np.asarray(params, float), aic

    def _one_step(self):
        c, phi = self.params[0], self.params[1]
        out = np.full(len(self.y), c)
        prev = self.y[:-1]
        ok = np.isfinite(prev)
        out[1:][ok] = c + phi * (prev[ok] - c)
        return out

    def _path(self, prev, steps):
        c, phi = self.params[0], self.params[1]
        out = []
        for _ in range(steps):
            prev = c + phi * (prev - c) if np.isfinite(prev) else c
            out.append(prev)
        return np.array(out)

    def forecast(self, steps=1):
        return self._path(self.y[-1], steps)

    def predict(self, start=None, end=None, dynamic=False):
        if start is None:
            return self._one_step()
        # dynamic forecast from origin start-1 (uses forecasts after the first step)
        return self._path(self.y[start - 1], end - start + 1)


class ARIMA:
    def __init__(self, endog, order=(0, 0, 0), trend="c"):
        self.y = np.asarray(endog, float)
        self.order = order

    def fit(self):
        y = self.y
        c = float(np.nanmean(y))
        phi = 0.0
        if self.order[0] >= 1:
            x, z = y[:-1] - c, y[1:] - c
            ok = np.isfinite(x) & np.isfinite(z)
            phi = float(x[ok] @ z[ok] / (x[ok] @ x[ok]))
        res = _Results(y, [c, phi, 0.0], 0.0)
        e = (y - res.predict())[np.isfinite(y)]
        var = float(np.mean(e ** 2))
        k = 2 + (self.order[0] >= 1) + self.order[2]
        res.aic = len(e) * np.log(var) + 2 * k
        res.params[2] = var
        return res

    def filter(self, params):
        return _Results(self.y, params, np.nan)
