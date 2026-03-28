from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def prepare_forward_returns(
    panel_df: pd.DataFrame,
    price_col: str = "close",
    forward_hours: int = 24,
) -> pd.DataFrame:
    """
    Build forward returns from panel data.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Must contain ['datetime', 'symbol', price_col]
    price_col : str
        Price column used to compute returns
    forward_hours : int
        Forward return horizon in hours

    Returns
    -------
    pd.DataFrame
        ['datetime', 'symbol', 'fwd_ret']
    """
    if forward_hours <= 0:
        raise ValueError("forward_hours must be positive")

    required = {"datetime", "symbol", price_col}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"panel_df missing columns: {missing}")

    df = panel_df[["datetime", "symbol", price_col]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    df["fwd_ret"] = (
        df.groupby("symbol")[price_col]
        .shift(-forward_hours) / df[price_col] - 1.0
    )
    return df[["datetime", "symbol", "fwd_ret"]]


def prepare_hourly_returns(
    panel_df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build 1-hour forward returns from panel data.

    Returns
    -------
    pd.DataFrame
        ['datetime', 'symbol', 'fwd_ret_1h']
    """
    required = {"datetime", "symbol", price_col}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"panel_df missing columns: {missing}")

    df = panel_df[["datetime", "symbol", price_col]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    df["fwd_ret_1h"] = (
        df.groupby("symbol")[price_col]
        .shift(-1) / df[price_col] - 1.0
    )
    return df[["datetime", "symbol", "fwd_ret_1h"]]


def select_rebalance_timestamps(
    timestamps: list[pd.Timestamp],
    rebalance_every_hours: int,
) -> list[pd.Timestamp]:
    """
    Select timestamps according to rebalancing interval.
    """
    if rebalance_every_hours <= 0:
        raise ValueError("rebalance_every_hours must be positive")

    ts = sorted(pd.to_datetime(pd.Series(timestamps)).drop_duplicates().tolist())
    if not ts:
        return []

    selected = [ts[0]]
    last = ts[0]

    for t in ts[1:]:
        delta_h = (t - last).total_seconds() / 3600.0
        if delta_h >= rebalance_every_hours:
            selected.append(t)
            last = t

    return selected


def build_top_bottom_weights(
    signal_df: pd.DataFrame,
    quantile: float = 0.1,
) -> pd.DataFrame:
    """
    Build equal-weight top-bottom portfolio weights on each rebalance timestamp.

    Parameters
    ----------
    signal_df : pd.DataFrame
        ['datetime', 'symbol', 'score']
    quantile : float
        top/bottom fraction

    Returns
    -------
    pd.DataFrame
        ['datetime', 'symbol', 'score', 'weight']
    """
    required = {"datetime", "symbol", "score"}
    missing = required - set(signal_df.columns)
    if missing:
        raise ValueError(f"signal_df missing columns: {missing}")

    if not (0 < quantile < 0.5):
        raise ValueError("quantile must be in (0, 0.5)")

    df = signal_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    def _assign(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("score").copy()
        n = len(g)
        q = int(np.floor(n * quantile))

        g["weight"] = 0.0
        if q > 0:
            g.loc[g.index[:q], "weight"] = -1.0 / q
            g.loc[g.index[-q:], "weight"] = 1.0 / q
        return g

    out = df.groupby("datetime", group_keys=False).apply(_assign)
    return out.reset_index(drop=True)


def expand_weights_to_all_hours(
    weight_df: pd.DataFrame,
    returns_df_1h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill rebalance weights to all hourly timestamps.

    Parameters
    ----------
    weight_df : pd.DataFrame
        ['datetime', 'symbol', 'weight']
    returns_df_1h : pd.DataFrame
        ['datetime', 'symbol', 'fwd_ret_1h']

    Returns
    -------
    pd.DataFrame
        ['datetime', 'symbol', 'weight', 'fwd_ret_1h']
    """
    required_w = {"datetime", "symbol", "weight"}
    required_r = {"datetime", "symbol", "fwd_ret_1h"}

    missing_w = required_w - set(weight_df.columns)
    missing_r = required_r - set(returns_df_1h.columns)

    if missing_w:
        raise ValueError(f"weight_df missing columns: {missing_w}")
    if missing_r:
        raise ValueError(f"returns_df_1h missing columns: {missing_r}")

    w = weight_df[["datetime", "symbol", "weight"]].copy()
    r = returns_df_1h[["datetime", "symbol", "fwd_ret_1h"]].copy()

    w["datetime"] = pd.to_datetime(w["datetime"])
    r["datetime"] = pd.to_datetime(r["datetime"])

    out = r.merge(w, on=["datetime", "symbol"], how="left")
    out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    out["weight"] = out.groupby("symbol")["weight"].ffill().fillna(0.0)
    return out


def compute_turnover(weight_panel: pd.DataFrame) -> pd.Series:
    """
    Turnover = sum_i |w_t - w_{t-1}|
    """
    wp = weight_panel.sort_index().fillna(0.0)
    return wp.diff().abs().sum(axis=1)


def build_execution_target_table(
    bt_result: dict,
    panel_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    initial_portfolio_value: float = 1.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build a rebalance-level execution table for a single symbol.

    Returns
    -------
    pd.DataFrame
        ['datetime', 'current_weight', 'target_weight', 'delta_weight',
         'portfolio_value', 'price', 'target_qty']
    """
    if initial_portfolio_value <= 0:
        raise ValueError("initial_portfolio_value must be positive")

    required_bt = {"weight_df", "cumret"}
    missing_bt = required_bt - set(bt_result.keys())
    if missing_bt:
        raise ValueError(f"bt_result missing keys: {missing_bt}")

    required_panel = {"datetime", "symbol", price_col}
    missing_panel = required_panel - set(panel_df.columns)
    if missing_panel:
        raise ValueError(f"panel_df missing columns: {missing_panel}")

    weight_df = bt_result["weight_df"].copy()
    weight_df["datetime"] = pd.to_datetime(weight_df["datetime"])

    symbol_weight_df = (
        weight_df.loc[weight_df["symbol"] == symbol, ["datetime", "weight"]]
        .sort_values("datetime")
        .rename(columns={"weight": "target_weight"})
        .reset_index(drop=True)
    )

    if symbol_weight_df.empty:
        raise ValueError(f"No weights found for symbol={symbol}")

    symbol_weight_df["current_weight"] = symbol_weight_df["target_weight"].shift(1).fillna(0.0)
    symbol_weight_df["delta_weight"] = (
        symbol_weight_df["target_weight"] - symbol_weight_df["current_weight"]
    )

    portfolio_value = bt_result["cumret"].rename("cumret").reset_index()
    portfolio_value.columns = ["datetime", "cumret"]
    portfolio_value["datetime"] = pd.to_datetime(portfolio_value["datetime"])
    portfolio_value["portfolio_value"] = initial_portfolio_value * portfolio_value["cumret"]

    price_df = (
        panel_df.loc[panel_df["symbol"] == symbol, ["datetime", price_col]]
        .copy()
        .rename(columns={price_col: "price"})
    )
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])

    out = symbol_weight_df.merge(
        portfolio_value[["datetime", "portfolio_value"]],
        on="datetime",
        how="left",
    ).merge(
        price_df,
        on="datetime",
        how="left",
    )

    out["target_qty"] = out["delta_weight"] * out["portfolio_value"] / out["price"]

    return out[
        [
            "datetime",
            "current_weight",
            "target_weight",
            "delta_weight",
            "portfolio_value",
            "price",
            "target_qty",
        ]
    ]


def backtest_long_short_portfolio(
    pred_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    quantile: float = 0.1,
    rebalance_every_hours: int = 24,
    portfolio_forward_hours: int = 24,
    price_col: str = "close",
) -> dict:
    """
    End-to-end portfolio backtest from model score to returns.

    Notes
    -----
    This function now uses path-based hourly PnL accumulation as the main
    backtest logic:
        pnl_t = weight_t * fwd_ret_1h_t

    So:
    - rebalance_every_hours controls how often weights are refreshed
    - portfolio_forward_hours is kept mainly for summary / compatibility /
      signal evaluation reference, but main portfolio PnL no longer uses
      direct H-hour forward return multiplication.
    """
    pred = pred_df.copy()
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    returns_df = prepare_forward_returns(
        panel_df=panel_df,
        price_col=price_col,
        forward_hours=portfolio_forward_hours,
    )

    hourly_returns_df = prepare_hourly_returns(
        panel_df=panel_df,
        price_col=price_col,
    )

    rebalance_ts = select_rebalance_timestamps(
        timestamps=pred["datetime"].drop_duplicates().tolist(),
        rebalance_every_hours=rebalance_every_hours,
    )

    rebalance_signal = pred[pred["datetime"].isin(rebalance_ts)].copy()

    weight_df = build_top_bottom_weights(
        signal_df=rebalance_signal,
        quantile=quantile,
    )

    # For reference / signal evaluation only
    period_df = weight_df.merge(
        returns_df,
        on=["datetime", "symbol"],
        how="left",
    )
    period_df["pnl_fwd"] = period_df["weight"] * period_df["fwd_ret"]

    # Main trading path: hourly holding pnl
    holding_df = expand_weights_to_all_hours(
        weight_df=weight_df,
        returns_df_1h=hourly_returns_df,
    )

    if rebalance_ts:
        active_start = pd.Timestamp(rebalance_ts[0])
        active_end = pd.Timestamp(rebalance_ts[-1]) + pd.Timedelta(hours=rebalance_every_hours - 1)
        holding_df = holding_df[
            holding_df["datetime"].between(active_start, active_end)
        ].reset_index(drop=True)

    holding_df["pnl"] = holding_df["weight"] * holding_df["fwd_ret_1h"]

    portfolio_return = (
        holding_df.groupby("datetime")["pnl"]
        .sum()
        .sort_index()
    )

    cumret = (1.0 + portfolio_return.fillna(0.0)).cumprod()

    weight_panel = (
        holding_df.pivot(index="datetime", columns="symbol", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = compute_turnover(weight_panel)

    long_count = (weight_panel > 0).sum(axis=1)
    short_count = (weight_panel < 0).sum(axis=1)
    gross_exposure = weight_panel.abs().sum(axis=1)
    net_exposure = weight_panel.sum(axis=1)

    return {
        "rebalance_every_hours": rebalance_every_hours,
        "portfolio_forward_hours": portfolio_forward_hours,
        "rebalance_timestamps": rebalance_ts,
        "rebalance_signal": rebalance_signal,
        "weight_df": weight_df,
        "period_df": period_df,          # reference only
        "holding_df": holding_df,        # main hourly path
        "weight_panel": weight_panel,
        "portfolio_return": portfolio_return,  # now hourly pnl series
        "cumret": cumret,
        "turnover": turnover,
        "long_count": long_count,
        "short_count": short_count,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
    }


def summarize_portfolio_result(
    bt_result: dict,
    annualization_hours: int = 24 * 365,
) -> dict:
    """
    Summarize portfolio performance.

    Note
    ----
    Since the main portfolio_return is now hourly path-based return,
    annualization uses annualization_hours directly.
    """
    ret = bt_result["portfolio_return"].dropna()
    if ret.empty:
        return {}

    cumret = bt_result["cumret"].dropna()
    dd = cumret / cumret.cummax() - 1.0

    mean_ret = ret.mean()
    vol = ret.std()

    periods_per_year = annualization_hours
    sharpe = np.sqrt(periods_per_year) * mean_ret / (vol + EPS)

    summary = {
        "rebalance_every_hours": bt_result["rebalance_every_hours"],
        "portfolio_forward_hours": bt_result["portfolio_forward_hours"],
        "n_periods": int(ret.shape[0]),
        "mean_ret": float(mean_ret),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "cum_return_last": float(cumret.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "avg_turnover": float(bt_result["turnover"].mean()),
        "avg_long_count": float(bt_result["long_count"].mean()),
        "avg_short_count": float(bt_result["short_count"].mean()),
        "avg_gross_exposure": float(bt_result["gross_exposure"].mean()),
        "avg_net_exposure": float(bt_result["net_exposure"].mean()),
    }
    return summary