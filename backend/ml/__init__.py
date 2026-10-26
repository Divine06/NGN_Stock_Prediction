"""ML prediction layer for the NGX trend prediction and decision support system.

Pipeline:  NGX daily history (CSV / Kobo backfill) + Kobo live daily closes
           -> scale-free features -> LSTM / RF / SVR / ARIMA (next-day return)
           -> ensemble forecast + technical indicators + Google News sentiment
           -> BUY / HOLD / SELL / INSUFFICIENT_DATA with evidence.
"""
