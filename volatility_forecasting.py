"""Forecast three-month equity volatility with small, classical ML models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
HORIZON = 3
TEST_FRACTION = 0.25
TICKERS = ["IBM", "AAPL", "MSFT", "XRX", "AMZN", "DELL", "GOOGL", "ADBE"]
MARKET = "^GSPC"

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Stocks.csv"
RESULTS_DIR = ROOT / "results"

FEATURES = [
    "return_1m",
    "abs_return_1m",
    "squared_return_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "volatility_3m",
    "volatility_6m",
    "volatility_12m",
    "vol_ratio_3_12",
    "market_abs_return_1m",
    "market_volatility_3m",
    "market_volatility_6m",
    "market_volatility_12m",
    "market_momentum_3m",
]


def load_prices(path=DATA_PATH):
    """Load monthly adjusted prices and remove empty/duplicate sample rows."""
    prices = pd.read_csv(path, comment="#", parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()
    prices = prices[prices.index.day == 1]
    return prices[~prices.index.duplicated(keep="first")]


def make_panel(prices):
    """Create one row per stock and month using information available at month t."""
    returns = prices.pct_change(fill_method=None)
    rows = []

    for ticker in TICKERS:
        stock_return = returns[ticker]
        frame = pd.DataFrame(index=prices.index)
        frame["ticker"] = ticker
        frame["return_1m"] = stock_return
        frame["abs_return_1m"] = stock_return.abs()
        frame["squared_return_1m"] = stock_return**2

        for window in (3, 6, 12):
            frame[f"momentum_{window}m"] = prices[ticker].pct_change(
                window, fill_method=None
            )
            frame[f"volatility_{window}m"] = (
                stock_return.rolling(window).std() * np.sqrt(12)
            )

        frame["vol_ratio_3_12"] = (
            frame["volatility_3m"] / frame["volatility_12m"]
        )
        frame["market_abs_return_1m"] = returns[MARKET].abs()

        for window in (3, 6, 12):
            frame[f"market_volatility_{window}m"] = (
                returns[MARKET].rolling(window).std() * np.sqrt(12)
            )

        frame["market_momentum_3m"] = prices[MARKET].pct_change(
            3, fill_method=None
        )

        # Annualized root-mean-square return over months t+1, t+2 and t+3.
        future_squared_returns = pd.concat(
            [stock_return.shift(-step) ** 2 for step in range(1, HORIZON + 1)],
            axis=1,
        )
        frame["target_volatility"] = np.sqrt(
            12 * future_squared_returns.mean(axis=1)
        )
        frame.loc[
            future_squared_returns.isna().any(axis=1), "target_volatility"
        ] = np.nan
        rows.append(frame.reset_index())

    panel = pd.concat(rows, ignore_index=True)
    required = FEATURES + ["target_volatility"]
    return (
        panel.dropna(subset=required)
        .sort_values(["Date", "ticker"])
        .reset_index(drop=True)
    )


def purged_train_test_split(panel):
    """Chronological 75/25 split with a three-month gap before the test set."""
    dates = np.sort(panel["Date"].unique())
    split_index = int((1 - TEST_FRACTION) * len(dates))
    test_start = dates[split_index]
    train_cutoff = dates[split_index - HORIZON]

    train = panel[panel["Date"] < train_cutoff].copy()
    test = panel[panel["Date"] >= test_start].copy()
    return train, test


def linear_preprocessor():
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), FEATURES),
            ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
        ]
    )


def tree_preprocessor():
    return ColumnTransformer(
        [
            ("numeric", "passthrough", FEATURES),
            (
                "ticker",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["ticker"],
            ),
        ]
    )


def log_target(regressor):
    """Fit log(volatility), then transform predictions back to volatility."""
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log,
        inverse_func=np.exp,
    )


def build_models():
    """Three deliberately small models: linear, bagged trees and boosted trees."""
    ridge = Pipeline(
        [
            ("preprocess", linear_preprocessor()),
            # Chosen from a small training-only time-series CV grid.
            ("regressor", Ridge(alpha=300.0)),
        ]
    )
    random_forest = Pipeline(
        [
            ("preprocess", tree_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=5,
                    min_samples_leaf=10,
                    max_features=0.7,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        [
            ("preprocess", tree_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=10,
                    loss="squared_error",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Ridge": log_target(ridge),
        "Random Forest": log_target(random_forest),
        "Gradient Boosting": log_target(gradient_boosting),
    }


def regression_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "r2": r2_score(actual, predicted),
    }


def cross_validate(train, models):
    """Expanding-window validation; the gap keeps future targets out of training."""
    feature_columns = FEATURES + ["ticker"]
    dates = np.sort(train["Date"].unique())
    splitter = TimeSeriesSplit(n_splits=4, gap=HORIZON)
    rows = []

    for fold, (train_dates_idx, valid_dates_idx) in enumerate(
        splitter.split(dates), start=1
    ):
        fold_train = train[train["Date"].isin(dates[train_dates_idx])]
        fold_valid = train[train["Date"].isin(dates[valid_dates_idx])]
        actual = fold_valid["target_volatility"]

        baseline_prediction = fold_valid["volatility_6m"]
        rows.append(
            {
                "model": "Historical 6m",
                "fold": fold,
                **regression_metrics(actual, baseline_prediction),
            }
        )

        for name, model in models.items():
            fitted = clone(model).fit(
                fold_train[feature_columns], fold_train["target_volatility"]
            )
            prediction = fitted.predict(fold_valid[feature_columns])
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    **regression_metrics(actual, prediction),
                }
            )

    cv = pd.DataFrame(rows)
    return (
        cv.groupby("model")[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
        .round(4)
    )


def fit_and_evaluate(train, test, models):
    feature_columns = FEATURES + ["ticker"]
    actual = test["target_volatility"].to_numpy()
    metric_rows = []
    prediction_frames = []
    fitted_models = {}

    candidates = {"Historical 6m": test["volatility_6m"].to_numpy()}
    for name, model in models.items():
        fitted_models[name] = clone(model).fit(
            train[feature_columns], train["target_volatility"]
        )
        candidates[name] = fitted_models[name].predict(test[feature_columns])

    for name, prediction in candidates.items():
        metric_rows.append({"model": name, **regression_metrics(actual, prediction)})
        prediction_frames.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].to_numpy(),
                    "ticker": test["ticker"].to_numpy(),
                    "actual": actual,
                    "prediction": prediction,
                    "model": name,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, fitted_models


def save_plots(metrics, predictions, fitted_models):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = metrics.sort_values("mae", ascending=False)
    bars = ax.barh(ordered["model"], 100 * ordered["mae"], color="#376996")
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_xlim(0, 100 * ordered["mae"].max() * 1.12)
    ax.set_xlabel("Test MAE (volatility percentage points; lower is better)")
    ax.set_title("Out-of-sample model comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    best_name = metrics.iloc[0]["model"]
    best = predictions[predictions["model"] == best_name]
    monthly = best.groupby("Date")[["actual", "prediction"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, 100 * monthly["actual"], label="Actual", linewidth=1.8)
    ax.plot(
        monthly.index,
        100 * monthly["prediction"],
        label=f"Predicted ({best_name})",
        linewidth=1.8,
    )
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Average predicted and realized volatility in the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    boosting_pipeline = fitted_models["Gradient Boosting"].regressor_
    preprocessor = boosting_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = boosting_pipeline.named_steps["regressor"].feature_importances_
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .assign(
            feature=lambda data: data["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("ticker__ticker_", "ticker=", regex=False)
        )
        .sort_values("importance", ascending=False)
    )
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#E07A5F")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Gradient Boosting feature importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    panel = make_panel(load_prices())
    train, test = purged_train_test_split(panel)
    models = build_models()

    cv_metrics = cross_validate(train, models)
    metrics, predictions, fitted_models = fit_and_evaluate(train, test, models)

    metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    cv_metrics.to_csv(RESULTS_DIR / "cross_validation_metrics.csv")
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_plots(metrics, predictions, fitted_models)

    print(f"Panel rows: {len(panel):,}")
    print(
        f"Train: {train['Date'].min():%Y-%m} to {train['Date'].max():%Y-%m} "
        f"({len(train):,} rows)"
    )
    print(
        f"Test:  {test['Date'].min():%Y-%m} to {test['Date'].max():%Y-%m} "
        f"({len(test):,} rows)"
    )
    print("\nOut-of-sample test metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMean expanding-window CV metrics")
    print(cv_metrics.xs("mean", axis=1, level=1).to_string())


if __name__ == "__main__":
    main()
