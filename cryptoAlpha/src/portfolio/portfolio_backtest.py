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
    Mainly used for plotting intra-holding PnL path if needed.
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
            g.iloc[:q, g.columns.get_loc("weight")] = -1.0 / q
            g.iloc[-q:, g.columns.get_loc("weight")] = 1.0 / q
        return g

    out = df.groupby("datetime", group_keys=False).apply(_assign)
    return out.reset_index(drop=True)


def expand_weights_to_all_hours(
    weight_df: pd.DataFrame,
    returns_df_1h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill rebalance weights to all hourly timestamps.
    This is useful for plotting hourly exposure / turnover path.

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

    Parameters
    ----------
    pred_df : pd.DataFrame
        ['datetime', 'symbol', 'score']
    panel_df : pd.DataFrame
        ['datetime', 'symbol', price_col]
    quantile : float
        top/bottom bucket fraction
    rebalance_every_hours : int
        e.g. 1 / 4 / 24
    portfolio_forward_hours : int
        forward holding / evaluation horizon for portfolio return
    price_col : str
        close price column

    Returns
    -------
    dict
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

    # Main portfolio return is computed on rebalance timestamps
    period_df = weight_df.merge(
        returns_df,
        on=["datetime", "symbol"],
        how="left",
    )
    period_df["pnl"] = period_df["weight"] * period_df["fwd_ret"]

    portfolio_return = (
        period_df.groupby("datetime")["pnl"]
        .sum()
        .sort_index()
    )

    cumret = (1.0 + portfolio_return.fillna(0.0)).cumprod()

    # Hourly holding path for turnover / exposure visualization
    holding_df = expand_weights_to_all_hours(
        weight_df=weight_df,
        returns_df_1h=hourly_returns_df,
    )

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
        "period_df": period_df,
        "holding_df": holding_df,
        "weight_panel": weight_panel,
        "portfolio_return": portfolio_return,
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
    annualization_hours refers to the number of hours in one year.
    Effective annualization factor is adjusted by portfolio_forward_hours.
    """
    ret = bt_result["portfolio_return"].dropna()
    if ret.empty:
        return {}

    cumret = bt_result["cumret"].dropna()
    dd = cumret / cumret.cummax() - 1.0

    mean_ret = ret.mean()
    vol = ret.std()

    portfolio_forward_hours = bt_result["portfolio_forward_hours"]
    periods_per_year = annualization_hours / portfolio_forward_hours

    sharpe = np.sqrt(periods_per_year) * mean_ret / (vol + EPS)

    summary = {
        "rebalance_every_hours": bt_result["rebalance_every_hours"],
        "portfolio_forward_hours": portfolio_forward_hours,
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