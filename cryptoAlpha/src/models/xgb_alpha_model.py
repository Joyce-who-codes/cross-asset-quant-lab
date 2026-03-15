from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb


class XGBAlphaModel:

    def __init__(
        self,
        horizon: int = 24,
        train_window: int = 180,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.horizon = horizon
        self.train_window = train_window

        self.model_params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    def build_label(self, panel_df: pd.DataFrame, price_col="close"):

        df = panel_df.copy()

        df["future_return"] = (
            df.groupby("symbol")[price_col]
            .shift(-self.horizon) / df[price_col] - 1
        )

        return df

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "future_return",
    ) -> pd.DataFrame:

        return self._fit_predict_impl(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            daily_model_dir=None,
        )

    def fit_predict_save_daily_models(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        daily_model_dir: str | Path,
        label_col: str = "future_return",
    ) -> pd.DataFrame:

        return self._fit_predict_impl(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            daily_model_dir=Path(daily_model_dir),
        )

    def _fit_predict_impl(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        daily_model_dir: Path | None,
    ) -> pd.DataFrame:

        df = df.sort_values(["datetime", "symbol"])

        dates = sorted(df["datetime"].unique())

        preds = []

        if daily_model_dir is not None:
            daily_model_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.train_window, len(dates)):

            train_dates = dates[i - self.train_window:i]
            test_date = dates[i]

            train_df = df[df["datetime"].isin(train_dates)]
            test_df = df[df["datetime"] == test_date]

            train_df = train_df.dropna(subset=feature_cols + [label_col])
            test_df = test_df.dropna(subset=feature_cols)

            if len(train_df) == 0:
                continue

            X_train = train_df[feature_cols].values
            y_train = train_df[label_col].values

            model = xgb.XGBRegressor(**self.model_params)
            model.fit(X_train, y_train)

            if daily_model_dir is not None:
                test_ts = pd.Timestamp(test_date)
                model_fp = daily_model_dir / f"xgb_alpha_{test_ts:%Y%m%d}.json"
                model.save_model(model_fp)

            X_test = test_df[feature_cols].values

            test_df = test_df.copy()
            test_df["score"] = model.predict(X_test)

            preds.append(test_df[["datetime", "symbol", "score"]])

        if not preds:
            raise ValueError("No predictions were generated. Check train_window and input data.")

        pred_df = pd.concat(preds).reset_index(drop=True)

        return pred_df