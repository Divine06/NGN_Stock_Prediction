import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail ? ` — ${body.detail}` : "";
    } catch {
      // The response may not contain JSON.
    }
    throw new Error(`Request failed: ${response.status} ${response.statusText}${detail}`);
  }
  return response.json();
}

const money = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : `₦${Number(value).toLocaleString("en-NG", { maximumFractionDigits: 2 })}`;

const pct = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(2)}%`;

const safeNumber = (value) => {
  const number = Number(value);
  return value == null || value === "" || !Number.isFinite(number) ? null : number;
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
  const hour = Number(getPart("hour"));
  const minute = Number(getPart("minute"));
  const minutesNow = hour * 60 + minute;
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  const isTradingHours = minutesNow >= 600 && minutesNow < 870;
  const isOpen = isWeekday && isTradingHours;

  return {
    isOpen,
    label: isOpen ? "MARKET OPEN" : "MARKET CLOSED",
    message: isOpen ? "Within configured trading hours" : "Outside configured trading hours",
  };
}

function SignalBadge({ value, large = false }) {
  const signal = String(value || "INSUFFICIENT_DATA").toUpperCase();
  const label = signal === "INSUFFICIENT_DATA" ? "COLLECTING DATA" : signal;
  return <span className={`signal-badge ${signal.toLowerCase()} ${large ? "signal-badge-large" : ""}`}>{label}</span>;
}

function StatCard({ label, value, helper, icon, tone = "" }) {
  return (
    <article className={`stat-card ${tone}`}>
      <div className="stat-card-top">
        <span className="stat-label">{label}</span>
        <span className="stat-icon" aria-hidden="true">{icon}</span>
      </div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-helper">{helper}</span>
    </article>
  );
}

function App() {
  const [stocks, setStocks] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState("AIRTELAFRI");
  const [selectedStock, setSelectedStock] = useState(null);
  const [history, setHistory] = useState([]);
  const [timeframe, setTimeframe] = useState("all");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsError, setNewsError] = useState("");
  const [showAllNews, setShowAllNews] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [marketStatus, setMarketStatus] = useState(getNGXMarketStatus);
  const [searchTerm, setSearchTerm] = useState("");

  const loadDashboard = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);
    else setLoading(true);

    try {
      const [market, decisionData] = await Promise.all([
        fetchJSON(`${API_BASE}/api/market-data`),
        fetchJSON(`${API_BASE}/api/decisions`),
      ]);
      const rows = market.stocks || [];
      setStocks(rows);
      setDecisions(decisionData.decisions || []);
      setLastUpdated(market.last_refresh || new Date().toISOString());
      setError("");
    } catch (e) {
      setError(e.message || "Could not load market data.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const loadSelected = useCallback(async (ticker, period = timeframe) => {
    if (!ticker) return;
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const [stock, hist] = await Promise.all([
        fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(ticker)}`),
        fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(ticker)}/history?timeframe=${encodeURIComponent(period)}`),
      ]);
      setSelectedStock(stock);
      const rows = Array.isArray(hist)
        ? hist
        : hist.data || hist.history || hist.observations || [];
      setHistory(rows);
    } catch (e) {
      setSelectedStock(null);
      setHistory([]);
      setHistoryError(e.message || "Could not load selected company history.");
    } finally {
      setHistoryLoading(false);
    }
  }, [timeframe]);

  useEffect(() => { setShowAllNews(false); }, [selectedTicker]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  useEffect(() => {
    if (selectedTicker) loadSelected(selectedTicker, timeframe);
  }, [selectedTicker, timeframe, loadSelected]);

  useEffect(() => {
    const id = setInterval(() => loadDashboard(true), 20 * 60 * 1000);
    return () => clearInterval(id);
  }, [loadDashboard]);

  useEffect(() => {
    const updateMarketStatus = () => setMarketStatus(getNGXMarketStatus());
    updateMarketStatus();
    const interval = setInterval(updateMarketStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const counts = useMemo(() => ({
    BUY: decisions.filter((item) => item.recommendation === "BUY").length,
    HOLD: decisions.filter((item) => item.recommendation === "HOLD").length,
    SELL: decisions.filter((item) => item.recommendation === "SELL").length,
    INSUFFICIENT_DATA: decisions.filter((item) => item.recommendation === "INSUFFICIENT_DATA").length,
  }), [decisions]);

  const selectedDecision = decisions.find((item) => item.ticker === selectedTicker) || selectedStock;
  const selectedMarketStock = stocks.find((item) => item.ticker === selectedTicker);

  const filteredStocks = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return stocks;
    return stocks.filter((stock) =>
      [stock.name, stock.ticker, stock.sector].some((value) =>
        String(value || "").toLowerCase().includes(term)
      )
    );
  }, [stocks, searchTerm]);

  const chartData = useMemo(() => history
    .map((row) => {
      const rawDate = row.timestamp ?? row.observed_at ?? row.date ?? "";
      const date = String(rawDate).slice(0, 10);
      const price = safeNumber(row.close ?? row.price);
      return { date, price };
    })
    .filter((row) => row.date && row.price != null && row.price > 0)
    .sort((a, b) => a.date.localeCompare(b.date)), [history]);

  async function forceRefresh() {
    setRefreshing(true);
    try {
      await fetchJSON(`${API_BASE}/api/refresh`, { method: "POST" });
      await loadDashboard(true);
      if (selectedTicker) await loadSelected(selectedTicker, timeframe);
    } catch (e) {
      setError(e.message || "Refresh failed.");
    } finally {
      setRefreshing(false);
    }
  }

  async function loadNews() {
    if (!selectedTicker) return;
    setNewsLoading(true);
    setNewsError("");
    try {
      const result = await fetchJSON(`${API_BASE}/api/stocks/${encodeURIComponent(selectedTicker)}/news`);
      setNews(result);
    } catch (e) {
      setNewsError(e.message || "Could not load news.");
    } finally {
      setNewsLoading(false);
    }
  }

  useEffect(() => {
    setNews(null);
    setNewsError("");
  }, [selectedTicker]);

  if (loading) {
    return (
      <main className="app-shell loading-shell">
        <div className="loading-card">
          <div className="loading-mark">N</div>
          <span className="eyebrow">NGX MARKET INTELLIGENCE</span>
          <h1>Preparing your dashboard</h1>
          <p>Connecting to the market data service…</p>
          <span className="loading-line" />
        </div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-label="NGX">N</div>
          <div className="brand-copy">
            <p className="eyebrow">MARKET INTELLIGENCE PLATFORM</p>
            <h1>Nigeria Stock Prediction System</h1>
            <p className="brand-subtitle">Trend prediction & decision support</p>
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
          <span className="feed-pill"><span className="feed-dot" /> Kobo delayed feed</span>
          <button className="button button-secondary refresh-button" type="button" onClick={forceRefresh} disabled={refreshing}>
            <span className={refreshing ? "refresh-symbol spinning" : "refresh-symbol"}>↻</span>
            {refreshing ? "Updating…" : "Refresh data"}
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
          <h2>Good analysis starts with clear signals.</h2>
        </div>
        <div className="updated-card">
          <span className="updated-icon">◷</span>
          <div><span>Last data refresh</span><strong>{lastUpdated ? new Date(lastUpdated).toLocaleString("en-NG", { dateStyle: "medium", timeStyle: "short" }) : "Not available"}</strong></div>
        </div>
      </section>

      <section className="summary-grid" aria-label="Market summary">
        <StatCard label="Companies tracked" value={stocks.length} helper="Companies with usable quotes" icon="▦" />
        <StatCard label="BUY signals" value={counts.BUY} helper="Positive trend threshold reached" icon="↗" tone="stat-buy" />
        <StatCard label="HOLD signals" value={counts.HOLD} helper="Movement within the neutral range" icon="Ⅱ" tone="stat-hold" />
        <StatCard label="SELL signals" value={counts.SELL} helper="Negative trend threshold reached" icon="↘" tone="stat-sell" />
      </section>

      <section className="workspace-grid">
        <aside className="panel watchlist-panel">
          <div className="panel-heading watchlist-heading">
            <div>
              
              <h2>Company watchlist</h2>
              <p className="panel-subtitle">Choose a company to inspect its signals.</p>
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
              const decision = decisions.find((item) => item.ticker === stock.ticker) || stock.trend || {};
              const active = selectedTicker === stock.ticker;
              const change = safeNumber(stock.change_pct);
              return (
                <button key={stock.ticker} className={`stock-row ${active ? "active" : ""}`} onClick={() => setSelectedTicker(stock.ticker)} aria-pressed={active}>
                  <span className={`stock-symbol ${active ? "selected-symbol" : ""}`}>{stock.ticker.slice(0, 2)}</span>
                  <span className="stock-info">
                    <strong>{stock.name || stock.ticker}</strong>
                    <small>{stock.ticker}{stock.sector ? ` · ${stock.sector}` : ""}</small>
                  </span>
                  <span className="stock-value">
                    <strong>{money(stock.price ?? stock.current_price)}</strong>
                    <small className={change == null ? "" : change >= 0 ? "positive" : "negative"}>{pct(stock.change_pct)}</small>
                  </span>
                  <span className="stock-signal-row"><SignalBadge value={decision.recommendation} /></span>
                </button>
              );
            })}
            {!filteredStocks.length && <div className="empty-list"><strong>No matching companies</strong><p>Try another company name or ticker symbol.</p></div>}
          </div>
          <div className="watchlist-footer"><span className="footer-dot" /> Signals are experimental and may change as new observations arrive.</div>
        </aside>

        <div className="main-column">
          <section className="panel detail-panel">
            {selectedDecision ? (
              <>
                <div className="detail-top">
                  <div className="company-heading">
                    <span className="company-avatar">{String(selectedTicker).slice(0, 2)}</span>
                    <div>
                      <p className="eyebrow">{selectedDecision.sector || selectedMarketStock?.sector || "NGX EQUITY"}</p>
                      <h2>{selectedDecision.name || selectedMarketStock?.name || selectedTicker}</h2>
                      <p className="company-ticker">{selectedTicker} <span>·</span> Nigerian Exchange</p>
                    </div>
                  </div>
                  <SignalBadge value={selectedDecision.recommendation} large />
                </div>

                <div className="price-hero">
                  <div>
                    <span className="metric-overline">LATEST AVAILABLE PRICE</span>
                    <div className="hero-price">{money(selectedDecision.price ?? selectedDecision.current_price ?? selectedMarketStock?.price)}</div>
                  </div>
                  <div className={`daily-change ${safeNumber(selectedDecision.change_pct ?? selectedMarketStock?.change_pct) == null ? "" : safeNumber(selectedDecision.change_pct ?? selectedMarketStock?.change_pct) >= 0 ? "positive-bg" : "negative-bg"}`}>
                    <span className="change-arrow">{safeNumber(selectedDecision.change_pct ?? selectedMarketStock?.change_pct) >= 0 ? "↗" : "↘"}</span>
                    <strong>{pct(selectedDecision.change_pct ?? selectedMarketStock?.change_pct)}</strong>
                    <small>Provider daily change</small>
                  </div>
                </div>

                <div className="metrics-grid">
                  <article className="metric-card featured-metric">
                    <span className="metric-label">Next-step estimate</span>
                    <strong>{money(selectedDecision.predicted_price)}</strong>
                    <small>Experimental estimate</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Estimated movement</span>
                    <strong className={safeNumber(selectedDecision.predicted_change_pct) == null ? "" : safeNumber(selectedDecision.predicted_change_pct) >= 0 ? "positive" : "negative"}>{pct(selectedDecision.predicted_change_pct)}</strong>
                    <small>Versus current price</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Trend confidence</span>
                    <strong>{selectedDecision.confidence || "Not available"}</strong>
                    <small>Based on available records</small>
                  </article>
                  <article className="metric-card">
                    <span className="metric-label">Observations</span>
                    <strong>{selectedDecision.observations ?? "—"}</strong>
                    <small>Records used by the trend</small>
                  </article>
                </div>

                <div className="chart-section">
                  <div className="chart-heading">
                    <div>
                      <p className="eyebrow">PRICE HISTORY</p>
                      <h3>Price observations</h3>
                    </div>
                    <label className="select-wrap">
                      <span className="sr-only">Chart timeframe</span>
                      <select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
                        <option value="1d">1 day</option>
                        <option value="1w">1 week</option>
                        <option value="1m">1 month</option>
                        <option value="3m">3 months</option>
                        <option value="6m">6 months</option>
                        <option value="1y">1 year</option>
                        <option value="all">All collected</option>
                      </select>
                    </label>
                  </div>
                  <div className="chart-area">
                    {historyLoading ? (
                      <div className="chart-state"><span className="mini-spinner" /><p>Loading price history…</p></div>
                    ) : historyError ? (
                      <div className="chart-state"><strong>History unavailable</strong><p>{historyError}</p></div>
                    ) : chartData.length < 2 ? (
                      <div className="chart-state">
                        <span className="chart-state-icon">⌁</span>
                        <strong>{chartData.length === 0 ? "No historical prices available" : "Not enough points to draw a line"}</strong>
                        <p>{chartData.length === 0 ? "Check the history endpoint and selected timeframe." : "Only one valid price record was returned for this period."}</p>
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 12, right: 18, left: 4, bottom: 4 }}>
                          <CartesianGrid strokeDasharray="3 5" stroke="#e8eee9" vertical={false} />
                          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#7a887f" }} axisLine={false} tickLine={false} minTickGap={28} />
                          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "#7a887f" }} axisLine={false} tickLine={false} width={76} tickFormatter={(value) => Number(value).toLocaleString("en-NG", { maximumFractionDigits: 0 })} />
                          <Tooltip
                            contentStyle={{ border: "1px solid #e3ebe5", borderRadius: 12, boxShadow: "0 12px 30px rgba(27, 55, 38, .10)" }}
                            labelStyle={{ color: "#65766a", fontSize: 12 }}
                            labelFormatter={(label) => `Date: ${label}`}
                            formatter={(value) => [money(value), "Observed price"]}
                          />
                          <Line type="monotone" dataKey="price" name="Observed price" stroke="#23734b" strokeWidth={2.7} dot={false} activeDot={{ r: 5, fill: "#23734b", stroke: "#fff", strokeWidth: 2 }} connectNulls={false} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                <div className="estimate-info">
                  <div className="estimate-info-icon">i</div>
                  <div className="estimate-info-content">
                    <h4>How to interpret this estimate</h4>
                    <p>
                      The system analyses historical NGX stock prices, the latest available
                      market-price updates, technical indicators and financial news sentiment.
                      These inputs are evaluated by the forecasting models and decision-support
                      engine to estimate the stock's next-spec price trend and generate a
                      BUY, HOLD or SELL signal.
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-detail"><span className="chart-state-icon">⌁</span><strong>Select a company</strong><p>Choose a stock from the watchlist to view its quote, trend estimate, and price history.</p></div>
            )}
          </section>

          <section className="panel news-panel">
            <div className="news-heading">
              <div>
                <p className="eyebrow">NEWS & SENTIMENT</p>
                <h2>{selectedTicker || "Company"} market news</h2>
                <p className="panel-subtitle">Headlines and experimental sentiment from your configured provider.</p>
              </div>
              <button className="button button-primary" type="button" onClick={loadNews} disabled={newsLoading || !selectedTicker}>
                {newsLoading ? <><span className="mini-spinner light-spinner" /> Loading…</> : <><span>↻</span> Refresh news</>}
              </button>
            </div>

            {newsLoading && !news && <div className="news-empty"><span className="mini-spinner" /><p>Loading company headlines…</p></div>}
            {!newsLoading && newsError && (
              <div className="news-status"><strong>News temporarily unavailable</strong><p>{newsError}</p><small>No sentiment conclusion is drawn from missing news.</small></div>
            )}
            {!news && !newsLoading && !newsError && (
              <div className="news-empty"><span className="news-empty-icon">↗</span><strong>Explore the latest headlines</strong><p>Load news for {selectedTicker || "the selected company"} to review article sources and available sentiment.</p><button className="text-button" type="button" onClick={loadNews}>Load company news <span>→</span></button></div>
            )}
            {news && (
              <>
                <div className="news-summary">
                  <div><span>Provider status</span><strong>{news.status || "Unknown"}</strong></div>
                  <div><span>Articles returned</span><strong>{news.article_count ?? news.articles?.length ?? 0}</strong></div>
                  <div><span>Sentiment</span><strong>{news.sentiment == null ? "Unavailable" : typeof news.sentiment === "object" ? news.sentiment.label || news.sentiment.sentiment || "Available" : news.sentiment}</strong></div>
                </div>
                {news.status === "no_articles" && <p className="news-inline-note">No matching articles were returned for this company.</p>}
                {news.message && <p className="news-inline-note">{news.message}</p>}
                <div className="news-list">
                  {(news.articles || []).slice(0, showAllNews ? undefined : 5).map((article, index) => {
                    const url = article.url || article.link;
                    const title = article.title || article.headline || "Untitled article";
                    return (
                      <article className="news-item" key={url || `${selectedTicker}-${index}`}>
                        <div className="news-item-marker">↗</div>
                        <div className="news-item-content">
                          <h3>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</h3>
                          <p>{article.source || article.domain || "News source"}{(article.published_at || article.publishedAt || article.date) ? ` · ${article.published_at || article.publishedAt || article.date}` : ""}</p>
                          {article.sentiment && <span className="news-tag">{typeof article.sentiment === "object" ? article.sentiment.label || "Sentiment analysed" : article.sentiment}</span>}
                        </div>
                      </article>
                    );
                  })}
                </div>
                {(news.articles || []).length > 5 && (
                  <button className="news-more-button" type="button"
                    onClick={() => setShowAllNews((current) => !current)}
                    aria-expanded={showAllNews}>
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
        <p>Final Year Bsc Project</p>
        <p>Last refresh: {lastUpdated ? new Date(lastUpdated).toLocaleString("en-NG") : "—"}</p>
      </footer>
    </main>
  );
}

export default App;