# Low-frequency data summary for cryptoAlpha

## Panel overview
- Rows: 197,923
- Columns: 27
- Symbols: 19
- Start: 2025-01-01 00:00:00
- End: 2026-03-11 00:00:00
- Unique timestamps: 10,417
- Inferred frequency: 0 days 01:00:00
- Avg symbols per timestamp: 19.00

## Sources used in the low-frequency panel
- **CoinGecko / COIN_MARKET_CHART/hourly**: Spot market price, market cap, total volume
  - Columns: price | market_cap | total_volume
  - Avg non-null ratio: 0.999
  - Model factors: not used directly in default Ridge factors
- **Coinglass / FUTURES_FUNDING_RATE_HISTORY**: Funding-rate history for derivatives sentiment/carry
  - Columns: funding_open | funding_high | funding_low | funding_close
  - Avg non-null ratio: 1.000
  - Model factors: funding_z_24
- **Coinglass / FUTURES_GLOBAL_LS_ACCOUNT_RATIO**: Global long-short account ratio
  - Columns: global_account_long_percent | global_account_short_percent | global_account_long_short_ratio
  - Avg non-null ratio: 0.984
  - Model factors: long_short_ratio_z_24
- **Coinglass / FUTURES_OPEN_INTEREST**: Open interest for leverage and positioning
  - Columns: oi_open | oi_high | oi_low | oi_close
  - Avg non-null ratio: 0.984
  - Model factors: oi_change_24h
- **Coinglass / FUTURES_PRICE_HISTORY**: Perpetual futures OHLCV price panel
  - Columns: open | high | low | close | volume_usd
  - Avg non-null ratio: 1.000
  - Model factors: mom_24h | mom_6h | volume_ratio_24
- **Coinglass / FUTURES_TAKER_BUY_SELL_VOLUME**: Aggressive buy/sell flow
  - Columns: taker_buy_volume_usd | taker_sell_volume_usd
  - Avg non-null ratio: 1.000
  - Model factors: taker_imbalance
- **Cryptoracle / selected social/sentiment metrics**: Social activity and sentiment indicators
  - Columns: active_community_count | mention_count | positive_sentiment_ratio | negative_sentiment_ratio
  - Avg non-null ratio: 0.800
  - Model factors: active_community_count_z_24

## Symbol coverage (top 10 by row count)

| symbol | n_rows | start_datetime | end_datetime | coverage_ratio_vs_global_timestamps | close_non_null_ratio | funding_non_null_ratio | oi_non_null_ratio | social_non_null_ratio | market_cap_non_null_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADAUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9994240184314102 |
| APTUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9992320245752135 |
| ATOMUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9993280215033119 |
| AVAXUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9993280215033119 |
| BCHUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9992320245752135 |
| BNBUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.0 | 0.9992320245752135 |
| BTCUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9993280215033119 |
| DOGEUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9994240184314102 |
| DOTUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9993280215033119 |
| ETCUSDT | 10417 | 2025-01-01 00:00:00 | 2026-03-11 00:00:00 | 1.0 | 1.0 | 1.0 | 0.9842565037918787 | 0.9858884515695497 | 0.9992320245752135 |