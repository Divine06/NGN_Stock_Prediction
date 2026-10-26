import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import "./styles.css";

// Set VITE_API_BASE in frontend/.env when the API is not on this machine.
const API_BASE = (import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000")
  .replace(/\/$/, "");
const REFRESH_MS = 20 * 60 * 1000; // the backend fetches prices every 20 minutes
const MODEL_ORDER = ["LSTM", "RF", "SVR", "ARIMA"];
const MODEL_NAMES = { LSTM: "LSTM", RF: "Random Forest", SVR: "SVR", ARIMA: "ARIMA", ENSEMBLE: "Ensemble (mean)", NAIVE: "No-change baseline" };

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      const d = body.detail;
      detail = d ? ` — ${typeof d === "object" ? d.message || JSON.stringify(d) : d}` : "";
    } catch {
      // The response may not contain JSON.
    }
    const err = new Error(`Request failed: ${response.status} ${response.statusText}${detail}`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

// Decisions always list every company; live prices are optional (e.g. before the first poll).
function fetchDashboard() {
  return Promise.allSettled([
    fetchJSON(`${API_BASE}/api/market-data`),
    fetchJSON(`${API_BASE}/api/decisions`),
  ]);
}

const money = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : `₦${Number(value).toLocaleString("en-NG", { maximumFractionDigits: 2 })}`;

const pct = (value, digits = 2) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(digits)}%`;

const num = (value, digits = 2) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toLocaleString("en-NG", { maximumFractionDigits: digits, minimumFractionDigits: digits });

const safeNumber = (value) => {
  const number = Number(value);
  return value == null || value === "" || !Number.isFinite(number) ? null : number;
};

const tone = (value) => {
  const n = safeNumber(value);
  return n == null ? "" : n >= 0 ? "positive" : "negative";
};

const shortDate = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso).slice(0, 10) : d.toLocaleDateString("en-NG", { day: "numeric", month: "short", year: "numeric" });
};

function getNGXMarketStatus() {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Africa/Lagos",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date());

  const getPart = (type) => parts.find((part) => part.type === type)?.value;
  const weekday = getPart("weekday");
  const minutesNow = Number(getPart("hour")) * 60 + Number(getPart("minute"));
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  // NGX trading hours: 9:00 AM - 4:00 PM WAT, Monday - Friday (matches backend market_data.get_market_status)
  const isTradingHours = minutesNow >= 540 && minutesNow < 960;
  const isOpen = isWeekday && isTradingHours;

  return {
    isOpen,
    label: isOpen ? "MARKET OPEN" : "MARKET CLOSED",
    message: isOpen ? "Prices update every 20 minutes" : "NGX trades 09:00–4:00 WAT, Mon–Fri",
  };
}

function SignalBadge({ value, large = false }) {
  const signal = String(value || "INSUFFICIENT_DATA").toUpperCase();
  const label = signal === "INSUFFICIENT_DATA" ? "NO SIGNAL" : signal;
  return <span className={`signal-badge ${signal.toLowerCase()} ${large ? "signal-badge-large" : ""}`}>{label}</span>;
}

function StatCard({ label, value, helper, icon, tone: cardTone = "" }) {
  return (
    <article className={`stat-card ${cardTone}`}>
      <div className="stat-card-top">
        <span className="stat-label">{label}</span>
        <span className="stat-icon" aria-hidden="true">{icon}</span>
      </div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-helper">{helper}</span>
    </article>
  );
}

function rsiTag(rsi) {
  const r = safeNumber(rsi);
  if (r == null) return null;
  if (r > 70) return { text: "Overbought", cls: "tag-sell" };
  if (r < 30) return { text: "Oversold", cls: "tag-buy" };
  return { text: "Neutral", cls: "tag-neutral" };
}

function crossTag(a, b, upText = "Bullish", downText = "Bearish") {
  const x = safeNumber(a);
  const y = safeNumber(b);
  if (x == null || y == null) return null;
  if (y !== 0 && Math.abs(x / y - 1) < 0.005) return { text: "Level", cls: "tag-neutral" };
  return x > y ? { text: upText, cls: "tag-buy" } : { text: downText, cls: "tag-sell" };
}

function IndicatorCard({ label, value, helper, tag }) {
  return (
    <article className="indicator-card">
      <span className="indicator-label">{label}</span>
      <strong>{value}</strong>
      <span className="indicator-foot">
        {tag && <span className={`indicator-tag ${tag.cls}`}>{tag.text}</span>}
        {helper && <small>{helper}</small>}
      </span>
    </article>
  );
}

function App() {
  const [stocks, setStocks] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [feed, setFeed] = useState({ source: "", note: "" });
  const [selectedTicker, setSelectedTicker] = useState("AIRTELAFRI");
  // Data for the selected company. Loading is derived from which ticker/timeframe is on screen,
  // so no state is set synchronously inside effects and slow replies cannot overwrite newer ones.
  const [selected, setSelected] = useState({ ticker: null, histKey: null, stock: null, history: [], meta: { gaps: [] }, metrics: null, error: "" });
  const latestRequest = useRef("");
  const [timeframe, setTimeframe] = useState("1y");
  const [overlays, setOverlays] = useState({ sma20: true, sma50: true, bands: false });
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsError, setNewsError] = useState("");
  const [showAllNews, setShowAllNews] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [marketStatus, setMarketStatus] = useState(getNGXMarketStatus);
  const [searchTerm, setSearchTerm] = useState("");

  const applyDashboard = useCallback(([marketRes, decisionRes]) => {
    if (decisionRes.status === "fulfilled") {
      const decisionRows = decisionRes.value.decisions || [];
      const market = marketRes.status === "fulfilled" ? marketRes.value : null;
      const live = new Map((market?.stocks || []).map((s) => [s.ticker, s]));
      setDecisions(decisionRows);
      setStocks(decisionRows.map((d) => {
        const q = live.get(d.ticker) || {};
        return {
          ticker: d.ticker,
          name: q.name || d.name,
          sector: q.sector || d.sector,
          price: q.price ?? d.current_price,
          change_pct: q.change_pct ?? d.change_pct,
          source: q.source || d.price_source,
        };
      }));
      setFeed(market
        ? { source: market.source || "Live feed", note: market.refresh_error ? "Last fetch had problems; showing stored prices" : "" }
        : { source: "Stored prices", note: "Live prices unavailable; showing the latest stored prices" });
      setLastUpdated(market?.last_refresh || decisionRes.value.last_refresh || new Date().toISOString());
      setError("");
    } else {
      setError(decisionRes.reason?.message || "Could not load the dashboard. Is the API running?");
    }
    setLoading(false);
  }, []);

  const loadDashboard = useCallback(() => fetchDashboard().then(applyDashboard), [applyDashboard]);

  const loadSelected = useCallback(async (ticker, period) => {
    if (!ticker) return;
    const key = `${ticker}|${period}`;
    latestRequest.current = key;
    const [stockRes, histRes, metricsRes] = await Promise.allSettled([
      fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(ticker)}`),
      fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(ticker)}/history?timeframe=${encodeURIComponent(period)}`),
      fetchJSON(`${API_BASE}/api/model-metrics/${encodeURIComponent(ticker)}`),
    ]);
    if (latestRequest.current !== key) return; // the user has already moved on
    const hist = histRes.status === "fulfilled" ? histRes.value : null;
    setSelected({
      ticker,
      histKey: key,
      stock: stockRes.status === "fulfilled" ? stockRes.value : null,
      history: hist ? (Array.isArray(hist) ? hist : hist.data || hist.history || []) : [],
      meta: hist ? { gaps: hist.gaps || [], note: hist.note } : { gaps: [] },
      metrics: metricsRes.status === "fulfilled" ? metricsRes.value : null,
      error: hist ? "" : histRes.reason?.message || "Could not load price history.",
    });
  }, []);

  function selectTicker(ticker) {
    setSelectedTicker(ticker);
    setShowAllNews(false);
    setNews(null);
    setNewsError("");
  }

  useEffect(() => {
    let active = true;
    fetchDashboard().then((result) => { if (active) applyDashboard(result); });
    return () => { active = false; };
  }, [applyDashboard]);

  useEffect(() => { loadSelected(selectedTicker, timeframe); }, [selectedTicker, timeframe, loadSelected]);

  useEffect(() => {
    const id = setInterval(() => {
      loadDashboard();
      loadSelected(selectedTicker, timeframe);
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [loadDashboard, loadSelected, selectedTicker, timeframe]);

  useEffect(() => {
    const interval = setInterval(() => setMarketStatus(getNGXMarketStatus()), 60000);
    return () => clearInterval(interval);
  }, []);

  const selectedStock = selected.ticker === selectedTicker ? selected.stock : null;
  const metrics = selected.ticker === selectedTicker ? selected.metrics : null;
  const historyLoading = selected.histKey !== `${selectedTicker}|${timeframe}`;
  const history = useMemo(() => (historyLoading ? [] : selected.history), [historyLoading, selected.history]);
  const historyMeta = historyLoading ? { gaps: [] } : selected.meta;
  const historyError = historyLoading ? "" : selected.error;

  const counts = useMemo(() => {
    const c = { BUY: 0, HOLD: 0, SELL: 0, INSUFFICIENT_DATA: 0 };
    decisions.forEach((d) => { c[d.recommendation] = (c[d.recommendation] || 0) + 1; });
    return c;
  }, [decisions]);

  const decision = decisions.find((item) => item.ticker === selectedTicker) || selectedStock?.prediction || null;
  const stockRow = stocks.find((item) => item.ticker === selectedTicker);
  const indicators = selectedStock?.indicators || {};
  const evidence = Array.isArray(decision?.evidence) ? decision.evidence
    : typeof decision?.reasons === "string" ? decision.reasons.split(" | ") : decision?.reasons || [];
  const warning = evidence.find((e) => /data gap|Data quality|not trained|days old/i.test(e));

  const filteredStocks = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return stocks;
    return stocks.filter((stock) =>
      [stock.name, stock.ticker, stock.sector].some((value) => String(value || "").toLowerCase().includes(term))
    );
  }, [stocks, searchTerm]);

  const intraday = timeframe === "1d";
  const chartData = useMemo(() => {
    const rows = history
      .map((row) => {
        const raw = String(row.timestamp ?? row.observed_at ?? row.date ?? "");
        const label = intraday ? raw.slice(11, 16) : raw.slice(0, 10);
        return {
          date: label,
          sortKey: raw,
          price: safeNumber(row.close ?? row.price),
          sma20: safeNumber(row.sma_20),
          sma50: safeNumber(row.sma_50),
          bbUpper: safeNumber(row.bb_upper),
          bbLower: safeNumber(row.bb_lower),
          rsi: safeNumber(row.rsi_14),
          forecast: null,
        };
      })
      .filter((row) => row.date && row.price != null && row.price > 0)
      .sort((a, b) => a.sortKey.localeCompare(b.sortKey));

    // Predicted next close drawn as a dashed segment from the last close to the target date.
    const predicted = safeNumber(decision?.predicted_price);
    const target = decision?.target_date;
    if (!intraday && rows.length && predicted != null && target && target > rows[rows.length - 1].date) {
      rows[rows.length - 1] = { ...rows[rows.length - 1], forecast: rows[rows.length - 1].price };
      rows.push({ date: target, sortKey: target, price: null, forecast: predicted });
    }
    return rows;
  }, [history, decision, intraday]);

  const hasRsi = chartData.some((r) => r.rsi != null);

  async function forceRefresh() {
    setRefreshing(true);
    let refreshError = "";
    try {
      await fetchJSON(`${API_BASE}/api/refresh`, { method: "POST" });
    } catch (e) {
      refreshError = `Live refresh failed (${e.message}). Showing the latest stored prices.`;
    }
    await loadDashboard();
    await loadSelected(selectedTicker, timeframe);
    if (refreshError) setError(refreshError); // after the reload, which clears old errors
    setRefreshing(false);
  }

  async function rerunModels() {
    setRerunning(true);
    try {
      await fetchJSON(`${API_BASE}/api/predictions/run`, { method: "POST" });
      await loadDashboard();
      await loadSelected(selectedTicker, timeframe);
    } catch (e) {
      setError(`Model run failed: ${e.message}`);
    } finally {
      setRerunning(false);
    }
  }

  async function loadNews() {
    if (!selectedTicker) return;
    setNewsLoading(true);
    setNewsError("");
    try {
      setNews(await fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(selectedTicker)}/news`));
    } catch (e) {
      setNewsError(e.message || "Could not load news.");
    } finally {
      setNewsLoading(false);
    }
  }

  if (loading) {
    return (
      <main className="app-shell loading-shell">
        <div className="loading-card">
          <div className="loading-mark">N</div>
          <span className="eyebrow">NGX MARKET INTELLIGENCE</span>
          <h1>Preparing your dashboard</h1>
          <p>Connecting to the prediction service…</p>
          <span className="loading-line" />
        </div>
      </main>
    );
  }

  const modelRows = [...MODEL_ORDER, "ENSEMBLE", "NAIVE"].filter((m) =>
    m === "ENSEMBLE" ? decision?.predicted_price != null || metrics?.metrics?.ENSEMBLE
      : m === "NAIVE" ? metrics?.metrics?.NAIVE : decision?.models?.[m] || metrics?.metrics?.[m]);
  const metricFor = (m) => metrics?.metrics?.[m];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-label="NGX">N</div>
          <div className="brand-copy">
            <p className="eyebrow">MARKET INTELLIGENCE PLATFORM</p>
            <h1>Nigeria Stock Prediction System</h1>
            <p className="brand-subtitle">LSTM · Random Forest · SVR · ARIMA + news sentiment</p>
          </div>
        </div>
        <div className="header-actions">
          <div className={`market-status ${marketStatus.isOpen ? "is-open" : "is-closed"}`} role="status" aria-live="polite">
            <span className="market-status-dot" />
            <span className="market-status-copy">
              <strong>{marketStatus.label}</strong>
              <small>{marketStatus.message}</small>
            </span>
          </div>
         
          <button className="button button-secondary refresh-button" type="button" onClick={forceRefresh} disabled={refreshing}>
            <span className={refreshing ? "refresh-symbol spinning" : "refresh-symbol"}>↻</span>
            {refreshing ? "Updating…" : "Refresh prices"}
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <span className="error-mark">!</span>
          <div><strong>Unable to update dashboard</strong><p>{error}</p></div>
          <button className="icon-button" onClick={() => setError("")} aria-label="Dismiss error">×</button>
        </div>
      )}

      <section className="welcome-row">
        <div>
          <p className="eyebrow">MARKET OVERVIEW</p>
          <h2>Machine-learning forecasts for ten NGX stocks.</h2>
          {feed.note && <p className="section-subtitle">{feed.note}</p>}
        </div>
        <div className="updated-card">
          <span className="updated-icon">◷</span>
          <div><span>Last price refresh</span><strong>{lastUpdated ? new Date(lastUpdated).toLocaleString("en-NG", { dateStyle: "medium", timeStyle: "short" }) : "Not available"}</strong></div>
        </div>
      </section>

      <section className="summary-grid" aria-label="Market summary">
        <StatCard label="Companies tracked" value={decisions.length} helper={counts.INSUFFICIENT_DATA ? `${counts.INSUFFICIENT_DATA} without a signal yet` : "All companies have a signal"} icon="▦" />
        <StatCard label="BUY signals" value={counts.BUY} helper="Models expect a rise, indicators agree" icon="↗" tone="stat-buy" />
        <StatCard label="HOLD signals" value={counts.HOLD} helper="Weak or mixed evidence" icon="Ⅱ" tone="stat-hold" />
        <StatCard label="SELL signals" value={counts.SELL} helper="Models expect a fall, indicators agree" icon="↘" tone="stat-sell" />
      </section>

      <section className="workspace-grid">
        <aside className="panel watchlist-panel">
          <div className="panel-heading watchlist-heading">
            <div>
              <h2>Company watchlist</h2>
              <p className="panel-subtitle">Choose a company to inspect its forecast.</p>
            </div>
            <span className="count-pill">{filteredStocks.length} / {stocks.length}</span>
          </div>
          <div className="watchlist-tools">
            <label className="search-box">
              <span aria-hidden="true">⌕</span>
              <input value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search company or ticker" aria-label="Search companies" />
              {searchTerm && <button type="button" onClick={() => setSearchTerm("")} aria-label="Clear search">×</button>}
            </label>
          </div>
          <div className="stock-list">
            {filteredStocks.map((stock) => {
              const d = decisions.find((item) => item.ticker === stock.ticker) || {};
              const active = selectedTicker === stock.ticker;
              return (
                <button key={stock.ticker} className={`stock-row ${active ? "active" : ""}`} onClick={() => selectTicker(stock.ticker)} aria-pressed={active}>
                  <span className={`stock-symbol ${active ? "selected-symbol" : ""}`}>{stock.ticker.slice(0, 2)}</span>
                  <span className="stock-info">
                    <strong>{stock.name || stock.ticker}</strong>
                    <small>{stock.ticker}{stock.sector ? ` · ${stock.sector}` : ""}</small>
                  </span>
                  <span className="stock-value">
                    <strong>{money(stock.price)}</strong>
                    <small className={tone(stock.change_pct)}>{pct(stock.change_pct)}</small>
                  </span>
                  <span className="stock-signal-row">
                    {d.predicted_change_pct != null && <small className={`row-forecast ${tone(d.predicted_change_pct)}`}>forecast {pct(d.predicted_change_pct)}</small>}
                    <SignalBadge value={d.recommendation} />
                  </span>
                </button>
              );
            })}
            {!filteredStocks.length && <div className="empty-list"><strong>No matching companies</strong><p>Try another company name or ticker symbol.</p></div>}
          </div>
          <div className="watchlist-footer"><span className="footer-dot" /> Forecasts are for the next trading session.</div>
        </aside>

        <div className="main-column">
          <section className="panel detail-panel">
            {decision ? (
              <>
                <div className="detail-top">
                  <div className="company-heading">
                    <span className="company-avatar">{String(selectedTicker).slice(0, 2)}</span>
                    <div>
                      <p className="eyebrow">{decision.sector || stockRow?.sector || "NGX EQUITY"}</p>
                      <h2>{decision.name || stockRow?.name || selectedTicker}</h2>
                      <p className="company-ticker">{selectedTicker} <span>·</span> Nigerian Exchange</p>
                    </div>
                  </div>
                  <SignalBadge value={decision.recommendation} large />
                </div>

                <div className="price-hero">
                  <div>
                    <span className="metric-overline">CURRENT PRICE</span>
                    <div className="hero-price">{money(stockRow?.price ?? decision.current_price)}</div>
                    <span className="price-source">
                      {stockRow?.source || decision.price_source || "—"}
                      {decision.market_data_timestamp ? ` · ${new Date(decision.market_data_timestamp).toLocaleString("en-NG", { dateStyle: "medium", timeStyle: "short" })}` : ""}
                    </span>
                  </div>
                  <div className={`daily-change ${tone(stockRow?.change_pct) === "positive" ? "positive-bg" : tone(stockRow?.change_pct) === "negative" ? "negative-bg" : ""}`}>
                    <span className="change-arrow">{safeNumber(stockRow?.change_pct) == null ? "–" : safeNumber(stockRow?.change_pct) >= 0 ? "↗" : "↘"}</span>
                    <strong>{pct(stockRow?.change_pct)}</strong>
                    <small>Today's change</small>
                  </div>
                </div>

                {warning && (
                  <div className="data-warning" role="note">
                    <span className="data-warning-mark">!</span>
                    <p>{warning}</p>
                  </div>
                )}

                <div className="metrics-grid">
                  <article className="metric-card featured-metric">
                    <span className="metric-label">Predicted next close</span>
                    <strong>{money(decision.predicted_price)}</strong>
                    <small>{decision.target_date ? `For ${shortDate(decision.target_date)}` : "No forecast yet"}</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Expected change</span>
                    <strong className={tone(decision.predicted_change_pct)}>{pct(decision.predicted_change_pct)}</strong>
                    <small>{decision.forecast_base_price != null ? `From last close ${money(decision.forecast_base_price)}` : "Versus last close"}</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Model confidence</span>
                    <strong>{decision.confidence || "Not available"}</strong>
                    <small>{decision.models_agreeing ? `${decision.models_agreeing} models agree on direction` : "From test-period accuracy"}</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Signal score</span>
                    <strong className={tone(decision.score)}>{decision.score == null ? "—" : `${decision.score > 0 ? "+" : ""}${Number(decision.score).toFixed(2)}`}</strong>
                    <small>BUY ≥ +0.25 · SELL ≤ −0.25</small>
                  </article>
                </div>

                <div className="chart-section">
                  <div className="chart-heading">
                    <div>
                      <p className="eyebrow">PRICE HISTORY & FORECAST</p>
                      <h3>{intraday ? "Today's 20-minute prices" : "Daily closing prices"}</h3>
                    </div>
                    <div className="chart-controls">
                      {!intraday && (
                        <div className="overlay-toggles" role="group" aria-label="Chart overlays">
                          {[["sma20", "SMA 20"], ["sma50", "SMA 50"], ["bands", "Bollinger"]].map(([key, label]) => (
                            <button key={key} type="button" className={`toggle-chip ${overlays[key] ? "on" : ""}`} aria-pressed={overlays[key]}
                              onClick={() => setOverlays((o) => ({ ...o, [key]: !o[key] }))}>{label}</button>
                          ))}
                        </div>
                      )}
                      <label className="select-wrap">
                        <span className="sr-only">Chart timeframe</span>
                        <select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
                          <option value="1d">Today (20-min)</option>
                          <option value="1w">1 week</option>
                          <option value="1m">1 month</option>
                          <option value="3m">3 months</option>
                          <option value="6m">6 months</option>
                          <option value="1y">1 year</option>
                          <option value="5y">5 years</option>
                          <option value="all">All history</option>
                        </select>
                      </label>
                    </div>
                  </div>
                  <div className="chart-area">
                    {historyLoading ? (
                      <div className="chart-state"><span className="mini-spinner" /><p>Loading price history…</p></div>
                    ) : historyError ? (
                      <div className="chart-state"><strong>History unavailable</strong><p>{historyError}</p></div>
                    ) : chartData.filter((r) => r.price != null).length < 2 ? (
                      <div className="chart-state">
                        <span className="chart-state-icon">⌁</span>
                        <strong>{intraday ? "No 20-minute prices for today yet" : "Not enough prices for this period"}</strong>
                        <p>{intraday ? "Prices are collected every 20 minutes during market hours." : "Choose a longer timeframe."}</p>
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 12, right: 18, left: 4, bottom: 4 }}>
                          <CartesianGrid strokeDasharray="3 5" stroke="#e8eee9" vertical={false} />
                          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#7a887f" }} axisLine={false} tickLine={false} minTickGap={28} />
                          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "#7a887f" }} axisLine={false} tickLine={false} width={76}
                            tickFormatter={(value) => Number(value).toLocaleString("en-NG", { maximumFractionDigits: 0 })} />
                          <Tooltip
                            contentStyle={{ border: "1px solid #e3ebe5", borderRadius: 12, boxShadow: "0 12px 30px rgba(27, 55, 38, .10)" }}
                            labelStyle={{ color: "#65766a", fontSize: 12 }}
                            labelFormatter={(label) => (intraday ? `Time: ${label}` : `Date: ${label}`)}
                            formatter={(value, name) => [money(value), name]}
                          />
                          {!intraday && overlays.bands && <Line type="monotone" dataKey="bbUpper" name="Bollinger upper" stroke="#b9c7bd" strokeDasharray="4 4" dot={false} isAnimationActive={false} />}
                          {!intraday && overlays.bands && <Line type="monotone" dataKey="bbLower" name="Bollinger lower" stroke="#b9c7bd" strokeDasharray="4 4" dot={false} isAnimationActive={false} />}
                          {!intraday && overlays.sma50 && <Line type="monotone" dataKey="sma50" name="SMA 50" stroke="#c48a2c" strokeWidth={1.6} dot={false} isAnimationActive={false} />}
                          {!intraday && overlays.sma20 && <Line type="monotone" dataKey="sma20" name="SMA 20" stroke="#5b8fd1" strokeWidth={1.6} dot={false} isAnimationActive={false} />}
                          <Line type="monotone" dataKey="price" name="Close" stroke="#23734b" strokeWidth={2.5} dot={false}
                            activeDot={{ r: 5, fill: "#23734b", stroke: "#fff", strokeWidth: 2 }} connectNulls={false} isAnimationActive={false} />
                          <Line type="linear" dataKey="forecast" name="Predicted close" stroke="#7b4bd1" strokeWidth={2.2} strokeDasharray="6 4"
                            dot={{ r: 5, fill: "#7b4bd1", stroke: "#fff", strokeWidth: 2 }} connectNulls isAnimationActive={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                  <div className="chart-legend">
                    <span className="legend-line" /> Close
                    {!intraday && overlays.sma20 && <><span className="legend-line legend-sma20" /> SMA 20</>}
                    {!intraday && overlays.sma50 && <><span className="legend-line legend-sma50" /> SMA 50</>}
                    {!intraday && overlays.bands && <><span className="legend-line legend-bands" /> Bollinger</>}
                    {!intraday && decision.predicted_price != null && <><span className="legend-line legend-forecast" /> Predicted close</>}
                    <span className="chart-legend-note">
                      {historyMeta.gaps?.length ? `Data gaps: ${historyMeta.gaps.slice(-2).map((g) => `${g.from} → ${g.to}`).join(", ")}` : historyMeta.note || ""}
                    </span>
                  </div>

                  {!intraday && hasRsi && (
                    <div className="rsi-area">
                      <span className="rsi-title">RSI (14)</span>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 6, right: 18, left: 4, bottom: 0 }}>
                          <XAxis dataKey="date" hide />
                          <YAxis domain={[0, 100]} ticks={[30, 70]} tick={{ fontSize: 10, fill: "#7a887f" }} axisLine={false} tickLine={false} width={76} />
                          <ReferenceLine y={70} stroke="#e7b7b2" strokeDasharray="3 3" />
                          <ReferenceLine y={30} stroke="#b5dcc1" strokeDasharray="3 3" />
                          <Tooltip formatter={(v) => [num(v, 1), "RSI"]} labelFormatter={(l) => `Date: ${l}`} />
                          <Line type="monotone" dataKey="rsi" stroke="#6b5bd1" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                <div className="section-block">
                  <div className="section-block-heading">
                    <div>
                      <p className="eyebrow">TECHNICAL INDICATORS</p>
                      <h3>Latest values {selectedStock?.indicator_timestamp ? `(${shortDate(selectedStock.indicator_timestamp)})` : ""}</h3>
                    </div>
                  </div>
                  {selectedStock?.indicator_note && <p className="block-note">{selectedStock.indicator_note}</p>}
                  <div className="indicator-grid">
                    <IndicatorCard label="RSI (14)" value={num(indicators.rsi_14, 1)} tag={rsiTag(indicators.rsi_14)} helper="<30 oversold · >70 overbought" />
                    <IndicatorCard label="SMA 20 / SMA 50" value={`${money(indicators.sma_20)} / ${money(indicators.sma_50)}`} tag={crossTag(indicators.sma_20, indicators.sma_50)} helper="Short vs long trend" />
                    <IndicatorCard label="EMA 20" value={money(indicators.ema_20)} tag={crossTag(stockRow?.price ?? decision.current_price, indicators.ema_20, "Price above", "Price below")} />
                    <IndicatorCard label="MACD / signal" value={`${num(indicators.macd)} / ${num(indicators.macd_signal)}`} tag={crossTag(indicators.macd, indicators.macd_signal)} helper={`Histogram ${num(indicators.macd_histogram)}`} />
                    <IndicatorCard label="Bollinger bands" value={`${money(indicators.bb_lower)} – ${money(indicators.bb_upper)}`} helper={`Middle ${money(indicators.bb_middle)}`} />
                    <IndicatorCard label="ATR (14)" value={money(indicators.atr_14)} helper="Average daily range" />
                  </div>
                </div>

                <div className="section-block">
                  <div className="section-block-heading">
                    <div>
                      <p className="eyebrow">MACHINE-LEARNING MODELS</p>
                      <h3>Next-session forecast by model</h3>
                    </div>
                    <button className="button button-secondary" type="button" onClick={rerunModels} disabled={rerunning}>
                      <span className={rerunning ? "refresh-symbol spinning" : "refresh-symbol"}>↻</span>
                      {rerunning ? "Running models…" : "Re-run models"}
                    </button>
                  </div>
                  {modelRows.length ? (
                    <div className="table-wrap">
                      <table className="model-table">
                        <thead>
                          <tr>
                            <th>Model</th><th>Predicted close</th><th>Change</th>
                            <th title="Mean absolute percentage error on the test period">Test MAPE</th>
                            <th title="RMSE improvement over predicting no change">Skill vs no-change</th>
                            <th title="Share of up/down days called correctly">Direction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {modelRows.map((m) => {
                            const out = m === "ENSEMBLE" ? { price: decision.predicted_price, return_pct: decision.predicted_change_pct } : decision.models?.[m];
                            const mt = metricFor(m);
                            const isBest = metrics?.best_model === m;
                            return (
                              <tr key={m} className={m === "ENSEMBLE" ? "row-ensemble" : m === "NAIVE" ? "row-baseline" : ""}>
                                <td>{MODEL_NAMES[m] || m}{isBest && <span className="best-chip">best</span>}</td>
                                <td>{m === "NAIVE" ? money(decision.forecast_base_price) : money(out?.price)}</td>
                                <td className={tone(out?.return_pct)}>{m === "NAIVE" ? "0.00%" : pct(out?.return_pct)}</td>
                                <td>{mt ? `${num(mt.price_mape_pct)}%` : "—"}</td>
                                <td className={m === "NAIVE" ? "" : tone(mt?.rmse_skill_vs_naive_pct)}>{mt && m !== "NAIVE" ? pct(mt.rmse_skill_vs_naive_pct) : "—"}</td>
                                <td>{mt?.dir_acc != null ? `${num(mt.dir_acc * 100, 1)}%` : "—"}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="block-note">
                      {metrics?.status === "INVALID_DATA"
                        ? "Models were not trained for this company because its historical prices failed the data-quality check (see below)."
                        : "No model output yet. Train the models (python -m backend.ml.train), then press “Re-run models”."}
                    </p>
                  )}
                  {metrics?.test_start && (
                    <p className="table-foot">
                      Test period {shortDate(metrics.test_start)} – {shortDate(metrics.test_end)} ({metrics.test_days} trading days, never seen in training).
                      Skill above 0 means the model beat simply predicting no change.
                    </p>
                  )}
                </div>

                <div className="section-block">
                  <div className="section-block-heading">
                    <div>
                      <p className="eyebrow">DECISION SUPPORT</p>
                      <h3>Why {decision.recommendation === "INSUFFICIENT_DATA" ? "there is no signal" : `the signal is ${decision.recommendation}`}</h3>
                    </div>
                    {decision.prediction_created_at && <small className="block-meta">Computed {new Date(decision.prediction_created_at).toLocaleString("en-NG", { dateStyle: "medium", timeStyle: "short" })}</small>}
                  </div>
                  <ul className="reason-list">
                    {evidence.map((e, i) => <li key={i} className={e === warning ? "reason-warning" : ""}>{e}</li>)}
                  </ul>
                </div>

                <div className="estimate-info">
                  <div className="estimate-info-icon">i</div>
                  <div className="estimate-info-content">
                    <h4>How this forecast is made</h4>
                    <p>
                      Daily NGX prices since 2014, plus prices collected every 20 minutes from a free delayed feed, are
                      turned into returns and technical indicators. LSTM, Random Forest, SVR and ARIMA each forecast the
                      next session's return; their average gives the predicted close. The signal combines that forecast
                      (weighted by how well the models did on unseen test data) with RSI, MACD, moving averages and
                      Google News sentiment. A BUY is never shown when the models expect a fall, nor a SELL when they
                      expect a rise. This is decision support for a final-year project, not financial advice.
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-detail"><span className="chart-state-icon">⌁</span><strong>Select a company</strong><p>Choose a stock from the watchlist to view its price, forecast and history.</p></div>
            )}
          </section>

          <section className="panel news-panel">
            <div className="news-heading">
              <div>
                <p className="eyebrow">NEWS & SENTIMENT</p>
                <h2>{selectedTicker || "Company"} market news</h2>
                <p className="panel-subtitle">Google News headlines scored with VADER (−1 negative to +1 positive).</p>
              </div>
              <button className="button button-primary" type="button" onClick={loadNews} disabled={newsLoading || !selectedTicker}>
                {newsLoading ? <><span className="mini-spinner light-spinner" /> Loading…</> : <><span>↻</span> {news ? "Refresh news" : "Load news"}</>}
              </button>
            </div>

            {decision && (
              <p className="news-inline-note">
                Sentiment used in the signal: <strong className={tone(decision.sentiment_score)}>{decision.sentiment_label || "Neutral"} ({num(decision.sentiment_score)})</strong>
                {" "}from {decision.headline_count ?? 0} headline{decision.headline_count === 1 ? "" : "s"} in the last 7 days
                {(decision.headline_count ?? 0) < 3 ? " (fewer than 3, so it was left out of the score)" : ""}.
              </p>
            )}

            {newsLoading && !news && <div className="news-empty"><span className="mini-spinner" /><p>Loading company headlines…</p></div>}
            {!newsLoading && newsError && (
              <div className="news-status"><strong>News temporarily unavailable</strong><p>{newsError}</p><small>No sentiment conclusion is drawn from missing news.</small></div>
            )}
            {!news && !newsLoading && !newsError && (
              <div className="news-empty"><span className="news-empty-icon">↗</span><strong>Explore the latest headlines</strong><p>Load the last 30 days of news for {selectedTicker || "the selected company"}.</p><button className="text-button" type="button" onClick={loadNews}>Load company news <span>→</span></button></div>
            )}
            {news && (
              <>
                <div className="news-summary">
                  <div><span>Provider status</span><strong>{news.status || "Unknown"}</strong></div>
                  <div><span>Articles (30 days)</span><strong>{news.article_count ?? news.articles?.length ?? 0}</strong></div>
                  <div><span>Average sentiment</span><strong className={tone(news.sentiment?.score)}>{news.sentiment ? `${news.sentiment.overall} (${num(news.sentiment.score)})` : "Unavailable"}</strong></div>
                </div>
                {news.status === "no_articles" && <p className="news-inline-note">No matching articles were returned for this company.</p>}
                {news.message && <p className="news-inline-note">{news.message}</p>}
                <div className="news-list">
                  {(news.articles || []).slice(0, showAllNews ? undefined : 5).map((article, index) => {
                    const url = article.url || article.link;
                    const title = article.title || article.headline || "Untitled article";
                    const published = article.published || article.published_at || article.date;
                    const sentiment = typeof article.sentiment === "object" ? article.sentiment?.label : article.sentiment;
                    return (
                      <article className="news-item" key={url || `${selectedTicker}-${index}`}>
                        <div className="news-item-marker">↗</div>
                        <div className="news-item-content">
                          <h3>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</h3>
                          <p>{article.source || "News source"}{published ? ` · ${shortDate(published)}` : ""}</p>
                          {sentiment && <span className={`news-tag ${String(sentiment).toLowerCase()}`}>{sentiment}{article.score != null ? ` ${num(article.score)}` : ""}</span>}
                        </div>
                      </article>
                    );
                  })}
                </div>
                {(news.articles || []).length > 5 && (
                  <button className="news-more-button" type="button" onClick={() => setShowAllNews((current) => !current)} aria-expanded={showAllNews}>
                    {showAllNews ? "Show less" : `Show all ${(news.articles || []).length} articles`}
                    <span aria-hidden="true">{showAllNews ? "↑" : "↓"}</span>
                  </button>
                )}
              </>
            )}
          </section>
        </div>
      </section>

      <footer className="app-footer">
        <div className="footer-brand"><span className="footer-logo">N</span><span>NGX Stock Trend & Decision Support</span></div>
        <p>Final Year BSc Project · Not financial advice</p>
        <p>Last refresh: {lastUpdated ? new Date(lastUpdated).toLocaleString("en-NG") : "—"}</p>
      </footer>
    </main>
  );
}

export default App;
