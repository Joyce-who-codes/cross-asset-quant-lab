from __future__ import annotations

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

        df = df.sort_values(["datetime", "symbol"])

        dates = sorted(df["datetime"].unique())

        preds = []

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

            X_test = test_df[feature_cols].values

            test_df = test_df.copy()
            test_df["score"] = model.predict(X_test)

            preds.append(test_df[["datetime", "symbol", "score"]])

        pred_df = pd.concat(preds).reset_index(drop=True)

        return pred_df