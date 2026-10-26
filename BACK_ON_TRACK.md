# Getting the NGX prediction system back on track

Your objective:

> Historical NGX data + live prices → LSTM, Random Forest, SVR, ARIMA → plus Google News sentiment
> → prediction and decision support on a dashboard with indicators and charts.

This package restores that pipeline inside your existing backend. Nothing in your
original files was changed; everything new is in `backend/ml/`, `backend/api/main_ml.py`
and `tests/`.

```
data/raw/{TICKER}.csv (NGX daily, 2014-)  ─┐
NGN Market / Kobo every 20 min (free)    ─┴─> one daily series per stock ─> data-quality checks
   ─> 22 scale-free features ─> LSTM · RF · SVR · ARIMA (next-day return) ─> ensemble
   ─> + RSI/MACD/SMA  + Google News VADER sentiment ─> BUY / HOLD / SELL / INSUFFICIENT_DATA
   ─> SQLite `predictions` table ─> FastAPI (same routes as before) ─> React dashboard
```

---

## 0. The free, automatic setup

Once it's running, nothing is manual: prices are fetched every 20 minutes during market hours,
each day's close is saved, and the models produce a predicted price every weekday at 15:10 WAT.

```
every 20 min, 09:55-15:05 WAT, Mon-Fri          every weekday 15:10 WAT
NGN Market (free) ─┐                              daily closes ─> LSTM/RF/SVR/ARIMA ─> next-day + 5-day forecast
Kobo Terminal (free)┴─> current price ─> ngx_live.db   + indicators + Google News sentiment ─> BUY/HOLD/SELL
                                     └─> today's close added to the chart history
```

| Need | Free source | Limit vs what the system uses |
|---|---|---|
| Current price (first choice) | **NGN Market** free plan, `GET /v1/companies`: price, previous close, day high/low, volume, updated every 20 min | 3,000 calls/month; uses ~16 a day (~350/month) |
| Current price (fallback) | **Kobo Terminal** Personal tier, prices ≥30 min delayed | 100 requests/day; used only if NGN Market fails or misses a stock |
| Chart history | Your `data/raw/*.csv` (2014 →) plus every day collected automatically from now on | none |
| Predicted price | Your own models, trained on your computer | none |
| News sentiment | Google News RSS + VADER | none |

**One-time setup**
1. Get a free NGN Market key (sign up at ngnmarket.com, developer dashboard) and add both keys to the
   `.env` file in your project root:
   ```
   KOBO_API_KEY=your_kobo_key
   NGN_MARKET_API_KEY=ngm_live_your_key
   ```
   Without the NGN Market key the system simply uses Kobo (30-min delay instead of 20).
   To change the order: `LIVE_SOURCES=kobo,ngnmarket`.
2. `python -m backend.ml.train` (once; repeat monthly to include the new days).
3. Start it: `uvicorn backend.api.main_ml:app --reload` (dashboard + automatic collection), **or**
   `python -m backend.ml.collect --loop` (collection and predictions without the web server).
   Check a single fetch with `python -m backend.ml.collect`.

**What "automatic" depends on:** collection runs only while the API or the collector is running and the
computer is awake. For days the computer is off, nothing is fetched; the system bridges short gaps and
flags them. If you want it to run unattended, start `python -m backend.ml.collect --loop` with
Windows Task Scheduler at log-on and set the laptop not to sleep on trading days.

**The one thing no free API provides: past daily prices.** Your CSVs have holes in 2026 (and SEPLAT's history
is London prices). The system still runs without fixing this: predictions are marked *low confidence*
until ~50 continuous trading days have been collected automatically (about 10 weeks). To fix it
immediately instead, download the missing period once by hand (free) and import it:
- Investing.com (free account) → each stock's historical-data page, **Lagos** listing, **Currency in NGN**
  (e.g. `ng.investing.com/equities/dangcem-historical-data`), 1 Jan 2026 → today, *Download Data*
  (or copy the table into Excel and save as CSV). Their terms don't allow automated scraping, so this
  step stays manual.
- `python -m backend.ml.import_csv --folder path/to/downloads` (file names must contain the ticker).
- SEPLAT: download the full Lagos history from 2014 and run
  `python -m backend.ml.import_csv SEPLAT path/to/SEPLAT.csv --replace`.
The importer refuses a file whose prices don't match the live price or your existing history.

---

## 1. What went wrong (found in your code and data)

### Why every company got SELL
There were three separate causes, and they stacked up.

1. **Stale predictions compared with today's price.** `decisions/decision_engine.py` took the
   *last row of the LSTM test-set file* (a prediction for 30 Dec 2025) and compared it with today's
   Kobo price. In `latest_hybrid_decisions.csv`, Dangote Cement's "predicted" ₦468.86 was the LSTM's
   guess for 30 Dec 2025 (when it actually closed at ₦609), compared against a Kobo price of ₦1,066.70.
   That produced −56%. The changes ranged from −31% (TOTAL) to −98% (SEPLAT).
2. **Price-level models can't predict above their training range.** LSTM (MinMax on price), SVR
   (RBF on raw prices) and Random Forest can't output much more than the highest price they were
   trained on, and NGX rallied in 2024–2026. `python -m backend.ml.diagnose` on your CSVs: on all 10
   tickers the old-style model predicts a fall far more often than prices fell (e.g. DANGCEM: 88% of
   test days vs 4% actual, mean predicted change −28%), and its MAPE (2–30%) is worse than simply
   guessing "tomorrow = today" (0.05–1.7%).
3. **Random Forest "up vs not-up" read as "down".** `preprocess.py` sets `target = next_close > close`.
   On NGX, 83–99% of days have *no price change* for several of your stocks, so "not up" dominates.
   The RF predicted 0 for all ten companies, and the engine treated 0 as "downward movement" (−2 points).

### Model-specific bugs in `backend/models/`
- `svr_model.py` uses `close` as an input **and** as the target on the same row (leakage), with an
  unscaled target. Its predictions flatten out (DANGCEM: ₦207.78 every day at the end of the test period).
- `arima_model.py` forecasts the whole ~500-day test period in one go from the end of training, so the
  "prediction" is a flat line (DANGCEM: ₦261 every day). It should be one step ahead, day by day.
- The RF classifier scores 86–96% "accuracy" on DANGCEM/TOTAL/AIRTELAFRI with **precision 0**: it just
  predicts the majority class.

### Data problems
- **SEPLAT is not NGX data.** Your CSV ends 2025 at 284.00; Kobo shows ₦16,000.10, with a 52-week
  low of ₦5,610. It has 20 day-to-day moves beyond NGX's ±10% limit (your other nine stocks have 0–5) and
  253 trading days in 2025 (the London calendar; the others have 247). It looks like London Stock Exchange
  prices in pence. The new validator marks it INVALID until it is rebuilt.
- **2026 is patchy.** AIRTELAFRI/SEPLAT/TOTAL stop at 31 Dec 2025; NESTLE has a single 2026 row
  (18 Mar, +73% from the previous row); DANGCEM/UNILEVER/BUACEMENT have holes of 11–158 days; GTCO/ZENITH
  are missing late June to late August 2026.

### How the system drifted
`main_live.py` (the current API) replaced the models with a straight-line fit over the last 60 prices,
and its `/sentiment` route returns 0 by design. There are also two sentiment modules (`sentiment/main.py`
uses VADER; `news_sentiment.py` uses a hand-made word list) and four API entry points.

---

## 2. What the new code does

| File | Purpose |
|---|---|
| `backend/api/main_ml.py` | **The one API to run.** Same routes as `main.py` + `main_live.py`, now fed by the models |
| `backend/ml/history.py` | Merges `data/raw/*.csv` with daily closes built from the Kobo observations |
| `backend/ml/validate.py` | Data-quality checks: scale vs live price, NGX ±10% band, gaps, freshness |
| `backend/ml/features.py` | 22 scale-free features (returns, ratios, RSI, MACD/price, Bollinger %B, …) |
| `backend/ml/models.py` | LSTM, RF, SVR, ARIMA (+ no-change baseline), each predicting the change to days 1–5 ahead |
| `backend/ml/train.py` | Time-ordered train/test split, metrics vs baseline, saves models to `data/ml_models/` |
| `backend/ml/decision.py` | Forecast + indicators + sentiment → signal, with reasons; ML forecast acts as a gate |
| `backend/ml/predict.py` | Daily prediction run, stores results in the `predictions` table |
| `backend/ml/live_feed.py` | Free price feed (NGN Market first, Kobo fallback), 09:55–15:05 WAT weekdays (~16 requests/day) |
| `backend/ml/collect.py` | Runs collection + daily predictions without the web server (`--loop`) |
| `backend/ml/sentiment.py` | Google News RSS + VADER (one module instead of two) |
| `backend/ml/dashboard.py` | Builds the dashboard JSON: stock detail, indicators, chart history, decisions |
| `backend/ml/import_csv.py` | Imports downloaded daily prices (Investing.com etc.) with safety checks |
| `backend/ml/fill_gaps.py` | Lists missing date ranges; can fill them via Investo if your key has API access |
| `backend/ml/diagnose.py` | Reproduces the all-SELL cause on your CSVs (evidence for Chapter 4) |

**Fixes compared with the old pipeline**
- Models predict the next-day **return**, so they work at any price level; a predicted price is
  `last close × e^(forecast)`.
- Every model is compared with the no-change baseline on a time-ordered test period, never shuffled.
- Forecasts are always for the next session after the latest close, never an old test-set row.
- No price inputs alongside a price target (no leakage). ARIMA is one step ahead.
- Flat (no-trade) days count as neutral, not bearish; HOLD is the default when evidence is weak.
- The decision includes its reasons, and warns if all ten companies get the same BUY/SELL again.

---

## 3. Run it (from the project root, the folder that contains `backend/`)

```bash
pip install -r requirements-ml.txt          # adds nothing unusual: tensorflow, statsmodels, sklearn, fastapi…
python tests/run_tests.py                   # 42 tests on synthetic data; should all pass

python -m backend.ml.validate               # data-quality report (SEPLAT will be INVALID)
python -m backend.ml.fill_gaps --dry-run    # lists the missing date ranges per ticker
python -m backend.ml.import_csv --folder downloads/ngx              # free: Investing.com files (section 0)
python -m backend.ml.import_csv SEPLAT downloads/SEPLAT.csv --replace
python -m backend.ml.validate               # check again

python -m backend.ml.diagnose               # saves data/ml_models/diagnosis.csv
python -m backend.ml.train                  # all four models, days 1-5; writes data/ml_models/metrics_summary.csv
python -m backend.ml.predict                # first prediction run (needs internet for Google News)

uvicorn backend.api.main_ml:app --reload    # then open http://127.0.0.1:8000/docs
```

Stop using `main_live.py`: the new API does its polling, and running both would double your
requests. Don't run the API and `collect --loop` at the same time either. Your React dashboard should work unchanged because the routes are the same. The new API
also runs the models automatically every weekday at 15:10 WAT, after the delayed close arrives.

**Filling the 2026 holes for free:** use the Investing.com import in section 0 (`fill_gaps.py` needs a
working Investo API key, which their free plan doesn't include). Until the gaps are filled, predictions
are still produced but marked *low confidence*, with a note saying the inputs span a data gap.

**Frontend:** the zip also contains a corrected `frontend/src/App.jsx` and `frontend/src/styles.css`
(your previous versions are kept beside them as `App.before-ml.jsx` / `styles.before-ml.css`).
See section 9.

---

## 4. Endpoints

| Route | Returns |
|---|---|
| `GET /api/decisions`, `/api/decisions/{t}`, `/api/advisory/{t}` | Signal, predicted price/change, confidence, reasons, per-model outputs, sentiment, indicators |
| `GET /api/stocks/{t}` | Live price + all indicators (RSI, SMA20/50, EMA20, MACD, Bollinger, ATR) + prediction |
| `GET /api/stocks/{t}/history?timeframe=1d\|1w\|1m\|3m\|6m\|1y\|5y\|all` | OHLCV chart rows with indicator columns; `1d` = 20-min Kobo points |
| `GET /api/stocks/{t}/sentiment`, `/news` | VADER sentiment of recent Google News headlines |
| `GET /api/companies`, `/api/market-data`, `/api/market-status`, `/api/health`, `POST /api/refresh` | As before (market hours corrected to 10:00–14:30 WAT) |
| `GET /api/predictions`, `/api/predictions/{t}`, `POST /api/predictions/run` | Stored predictions; run now |
| `GET /api/predictions/accuracy` | Live track record: each prediction vs the next close actually collected |
| `GET /api/model-metrics`, `/api/model-metrics/{t}` | Test-period metrics per ticker and model |
| `GET /api/data-quality`, `/api/data-coverage/{t}` | Validation results and gaps |

---

## 5. First results on your data (RF and SVR only, run during this session)

TensorFlow and statsmodels weren't available in my environment, so LSTM and ARIMA must be run on your
machine with `python -m backend.ml.train`. Test periods are the last 20% of each stock's usable days.

| Ticker | Test period | MAPE: no-change / RF / SVR | RMSE skill vs no-change: RF / SVR | Confidence |
|---|---|---|---|---|
| GTCO | Nov 2023 – Jun 2026 | 1.52 / 1.53 / 1.55% | **+2.72% / +1.59%** | medium |
| ZENITHBANK | Jan 2024 – Jun 2026 | 1.63 / 1.66 / 1.71% | **+2.47%** / −0.82% | medium |
| TOTAL | Sep 2023 – Dec 2025 | 0.26 / 0.44 / 0.41% | +0.36% / +0.02% | medium |
| BUACEMENT | Nov 2024 – Dec 2025 | 0.72 / 1.09 / 0.91% | −0.10% / 0.00% | medium |
| MTNN | Sep 2024 – Dec 2025 | 0.97 / 1.17 / 1.11% | −0.93% / −2.82% | low |
| DANGCEM | Aug 2023 – Dec 2025 | 0.62 / 0.93 / 0.79% | −1.85% / −1.66% | low |
| NESTLE | Aug 2023 – Dec 2025 | 0.43 / 0.61 / 0.58% | −0.98% / −1.06% | low |
| UNILEVER | Oct 2023 – Jan 2026 | 1.22 / 1.51 / 1.45% | −2.51% / −2.21% | low |
| AIRTELAFRI | Sep 2024 – Dec 2025 | 0.05 / 0.21 / 0.19% | −6.71% / −2.40% | low |
| SEPLAT | excluded: history is not NGX data | | | |

How to read this honestly in Chapter 4: the old 11.6% average LSTM MAPE was inflated by the price-level
problem; the realistic bar for daily stock prices is the no-change guess. The models add a little
information on the actively traded banks and essentially none on stocks that rarely trade (AIRTELAFRI
didn't change price on 98.7% of its test days). That is a normal, defensible finding, and it's why
the decision engine weights the forecast by its measured confidence and defaults to HOLD.

The signals over the test periods are mixed (e.g. GTCO: 161 BUY, 373 HOLD, 41 SELL), with no all-SELL output.

---

### 5-day forecast (RF + SVR ensemble, your data)

Typical error (MAPE) of the forecast for each day ahead, against simply assuming no change:

| Ticker | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | No-change, day 5 |
|---|---|---|---|---|---|---|
| GTCO | 1.51% | 2.38% | 3.00% | 3.56% | 4.00% | 3.82% |
| ZENITHBANK | 1.66% | 2.57% | 3.30% | 3.81% | 4.30% | 4.06% |
| TOTAL | 0.40% | 0.74% | 1.07% | 1.37% | 1.66% | 1.25% |
| DANGCEM | 0.81% | 1.49% | 2.11% | 2.66% | 3.18% | 2.78% |
| MTNN | 1.10% | 1.91% | 2.64% | 3.27% | 3.87% | 3.55% |

Errors grow with each day ahead, and beyond day 1 the models rarely beat "no change" (RMSE skill turns
negative for most stocks by day 3). Report this honestly: the 5-day path is mostly a gentle drift, and
the useful part is the **likely range**, which shows how far prices typically move. LSTM and ARIMA results
will be added when you train on your machine.

## 6. What this means for Chapters 1–3

- **Chapter 1:** the original aim and objectives stand (historical NGX data + live prices → four models
  + sentiment → decision support). Describe the live source accurately: "Kobo Terminal, delayed by at
  least 30 minutes". Name where your 2014–2025 dataset actually came from, plus Investo for updates.
- **Chapter 2:** keep LSTM, RF, SVR, ARIMA and VADER; add the random-walk (no-change) baseline and why
  daily price forecasts must be judged against it.
- **Chapter 3:** targets = log return from today's close to each of the next 5 trading days; 22 scale-free features; time-ordered 80/20 split; scalers
  fitted on training data only; metrics MAE/RMSE/MAPE, directional accuracy and skill vs baseline;
  decision rule (below); data validation and gap handling; daily 15:10 WAT prediction run.

**5-day forecast (for section 8.x)**
- Each model predicts the cumulative log return to day h for h = 1..5: RF as one multi-output forest, SVR as
  one model per day, LSTM with 5 outputs, ARIMA by dynamic 5-step forecasting. The path is the average.
- Each day's forecast is evaluated separately on the test period (MAPE, directional accuracy, skill vs no change).
- "Likely range" = forecast price × e^(10th and 90th percentile of the ensemble's test-period errors for that day),
  i.e. where 80% of past outcomes fell relative to the forecast.
- Each day's cumulative change is capped by the NGX ±10% daily band compounded over h days.
- Dates are the next 5 weekdays; NGX public holidays are not skipped.
- The BUY/HOLD/SELL signal uses the day-1 forecast only. The horizon can be changed with `NGX_HORIZON`
  (e.g. 10); retrain afterwards.

**Decision rule (exact, for section 8.5)**
- Forecast vote = `tanh(forecast / max(0.25 × 20-day volatility, 0.2%))` × confidence (high 1.0, medium 0.6,
  low 0.3; confidence comes from the test-period comparison with the baseline).
- Technical vote = mean of RSI (<30 → +1, >70 → −1), MACD histogram sign, price vs SMA50 and SMA20 vs SMA50,
  each neutral within 0.5%.
- Sentiment vote = mean VADER compound of the last 7 days' headlines ÷ 0.5, clipped to ±1; used only with ≥3 headlines.
- Score = weighted mean of the available votes (0.5 / 0.3 / 0.2). BUY ≥ +0.25, SELL ≤ −0.25, else HOLD.
- **Gate:** BUY needs a forecast ≥ 0 and SELL a forecast ≤ 0; otherwise HOLD.
- INSUFFICIENT_DATA if the data fails validation, the models aren't trained or the latest close is over 7 days old.

**Chapter 4 material the system produces:** `diagnosis.csv` (old vs new behaviour), `metrics_summary.csv`,
`signal_backtest` in each `data/ml_models/{T}/metrics.json`, `/api/data-quality`, and the live record at
`/api/predictions/accuracy`.

---

## 7. Old files: keep for the report, don't run

| File | Status |
|---|---|
| `api/main_live.py`, `api/main.py`, `api/main_backup.py`, `backend/main.py` | Replaced by `api/main_ml.py` |
| `live_pipeline.py` (`build_trend`), `decisions/decision_engine.py` | Replaced by `ml/predict.py` + `ml/decision.py` |
| `models/*.py`, `data/model_data/`, `data/predictions/`, `data/decisions/` | The first (price-level) experiment; cite it in Chapter 4 as what failed and why |
| `news_sentiment.py`, `sentiment/main.py` | Replaced by `ml/sentiment.py` (VADER) |
| `data/stock_data.py` (yfinance) | Not used; probably where the London-listed SEPLAT data came from |
| **Still used:** `market_data.py` (Kobo), `data/historical_updater.py` (Investo), `data/raw/*.csv` | |

---

## 8. Limits to state

- Live prices are delayed (NGN Market refreshes every 20 minutes; Kobo is ≥30 minutes behind); forecasts are
  for the next session and are made after the close.
- Sentiment is applied at decision time, not used as a training feature: Google News RSS only returns
  recent headlines, so there's no headline history to train on.
- Predictions made across a data gap are marked low confidence until the gap is filled.
- Decision support for an academic project, not financial advice.

---

## 9. Frontend changes (`frontend/src/App.jsx`, `styles.css`)

Same design, same routes; it now shows what the project is about:

- **Current price** with its source (NGN Market / Kobo) and time; the dashboard still loads if live prices
  are temporarily unavailable (it falls back to the latest stored prices instead of an error screen).
- **Predicted next close** and a **5-day forecast**, model confidence and signal score. The chart draws the
  5-day path as a dashed purple line with a shaded likely range that widens each day, and a "Next 5
  trading days" table lists each day's predicted close, range, typical error and the no-change error.
- **Chart:** daily closes with SMA 20 / SMA 50 / Bollinger toggles, an RSI panel, data-gap notes, and a
  "Today (20-min)" view of the live observations. Timeframes up to 5 years and full history.
- **Technical indicators:** RSI, SMA 20/50, EMA 20, MACD, Bollinger bands, ATR with bullish/bearish tags.
- **Model table:** each model's (LSTM, RF, SVR, ARIMA) next close and day-5 forecast, plus its test-period MAPE, skill vs
  the no-change baseline and directional accuracy, and a **Re-run models** button.
- **Why this signal:** the decision engine's reasons, with data-quality warnings highlighted.
- **News:** article dates and VADER scores now display; the sentiment actually used in the signal is shown.
- Fixed: error details printed as `[object Object]`; a failed price refresh hid its own error message;
  slow responses for a previously clicked company could overwrite the current one; React "setState in
  effect" lint errors (your ESLint config now reports 0 problems, the old file had 4).

Run: `cd frontend`, `npm run dev`. If the API isn't at `http://127.0.0.1:8000`, copy `.env.example` to
`.env` and set `VITE_API_BASE`. Delete the old `dist/` folder (or run `npm run build`) before deploying.
Unused files you can delete: `src/App_backup.jsx`, `src/styles_backup.css`, `src/App.css`, `src/index.css`.
