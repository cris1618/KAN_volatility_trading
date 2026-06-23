import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kan_volatility.backtest import compute_strategy_returns, summarize_backtest
from kan_volatility.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)
from kan_volatility.strategy import (
    compute_buy_and_hold_weights,
    compute_threshold_weights,
    compute_volatility_target_weights,
)


DATASET_FILE = "spy_volatility_dataset.csv"

PREDICTION_FILES = [
    "baseline_test_predictions.csv",
    "mlp_test_predictions.csv",
    "kan_test_predictions.csv",
    "best_kan_test_predictions.csv",
]

TARGET_VOLATILITY = 0.10
TRANSACTION_COST_BPS = 1.0


def load_spy_returns() -> pd.DataFrame:
    """
    Load raw SPY prices and compute simple daily returns.
    """
    path = RAW_DATA_DIR / "SPY.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run scripts/01_download_data.py first."
        )

    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    data["spy_return"] = data["adj_close"].pct_change()

    return data[["date", "spy_return"]].dropna()


def load_predictions() -> pd.DataFrame:
    """
    Load model test predictions from all available prediction files.
    """
    frames = []

    for filename in PREDICTION_FILES:
        path = TABLES_DIR / filename

        if not path.exists():
            print(f"Skipping missing prediction file: {path}")
            continue

        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"])
        frames.append(data)

    if not frames:
        raise FileNotFoundError(
            "No prediction files found. Run baseline, MLP, KAN, and tuning scripts first."
        )

    predictions = pd.concat(frames, ignore_index=True, sort=False)

    return predictions


def pivot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long prediction format to wide format.

    Model predictions are log future realized volatility.
    We exponentiate them to get annualized volatility forecasts.
    """
    true_values = (
        predictions[["date", "y_true"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )

    pred_wide = predictions.pivot(
        index="date",
        columns="model",
        values="y_pred",
    ).reset_index()

    data = true_values.merge(pred_wide, on="date", how="inner")

    data["actual_future_vol"] = np.exp(data["y_true"])

    model_columns = [
        column
        for column in data.columns
        if column not in {"date", "y_true", "actual_future_vol"}
    ]

    for column in model_columns:
        data[f"{column}_predicted_vol"] = np.exp(data[column])

    return data.sort_values("date")


def run_model_backtests(
    prediction_data: pd.DataFrame,
    spy_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run volatility-targeting and threshold strategies for every model.
    """
    data = prediction_data.merge(spy_returns, on="date", how="inner")
    data = data.sort_values("date")
    data = data.set_index("date")

    predicted_vol_columns = [
        column
        for column in data.columns
        if column.endswith("_predicted_vol")
    ]

    all_summaries = []
    all_curves = []

    # Buy and hold benchmark as its own strategy.
    buy_hold_weights = compute_buy_and_hold_weights(
        index=data.index,
        weight=1.0,
    )

    buy_hold_results = compute_strategy_returns(
        asset_returns=data["spy_return"],
        weights=buy_hold_weights,
        transaction_cost_bps=0.0,
    )

    buy_hold_summary = summarize_backtest(
        results=buy_hold_results,
        strategy_name="buy_and_hold_spy",
    )

    all_summaries.append(buy_hold_summary)

    buy_hold_curve = buy_hold_results[
        ["equity_curve", "benchmark_equity_curve"]
    ].copy()
    buy_hold_curve["strategy"] = "buy_and_hold_spy"
    all_curves.append(buy_hold_curve.reset_index())

    for predicted_vol_column in predicted_vol_columns:
        model_name = predicted_vol_column.replace("_predicted_vol", "")

        predicted_vol = data[predicted_vol_column]

        vol_target_weights = compute_volatility_target_weights(
            predicted_volatility=predicted_vol,
            target_volatility=TARGET_VOLATILITY,
            max_weight=1.0,
            min_weight=0.0,
        )

        vol_target_results = compute_strategy_returns(
            asset_returns=data["spy_return"],
            weights=vol_target_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )

        vol_target_strategy_name = f"{model_name}_vol_target"

        vol_target_summary = summarize_backtest(
            results=vol_target_results,
            strategy_name=vol_target_strategy_name,
        )

        all_summaries.append(vol_target_summary)

        curve = vol_target_results[
            ["equity_curve", "benchmark_equity_curve", "weight"]
        ].copy()
        curve["strategy"] = vol_target_strategy_name
        all_curves.append(curve.reset_index())

        threshold_weights = compute_threshold_weights(
            predicted_volatility=predicted_vol,
            high_volatility_threshold=0.25,
            low_risk_weight=0.50,
            normal_risk_weight=1.00,
        )

        threshold_results = compute_strategy_returns(
            asset_returns=data["spy_return"],
            weights=threshold_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )

        threshold_strategy_name = f"{model_name}_threshold"

        threshold_summary = summarize_backtest(
            results=threshold_results,
            strategy_name=threshold_strategy_name,
        )

        all_summaries.append(threshold_summary)

        curve = threshold_results[
            ["equity_curve", "benchmark_equity_curve", "weight"]
        ].copy()
        curve["strategy"] = threshold_strategy_name
        all_curves.append(curve.reset_index())

    summary = pd.DataFrame(all_summaries)
    curves = pd.concat(all_curves, ignore_index=True, sort=False)

    return summary, curves


def plot_equity_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    top_n: int = 6,
) -> None:
    """
    Plot buy-and-hold and top strategies by Sharpe ratio.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values("sharpe", ascending=False)

    selected_strategies = ["buy_and_hold_spy"]

    for strategy in ranked["strategy"]:
        if strategy not in selected_strategies:
            selected_strategies.append(strategy)

        if len(selected_strategies) >= top_n:
            break

    plt.figure(figsize=(14, 6))

    for strategy in selected_strategies:
        strategy_curve = curves[curves["strategy"] == strategy].copy()
        strategy_curve = strategy_curve.sort_values("date")

        plt.plot(
            strategy_curve["date"],
            strategy_curve["equity_curve"],
            label=strategy,
            alpha=0.90,
        )

    plt.title("Volatility Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "backtest_equity_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_strategy_weights(curves: pd.DataFrame) -> None:
    """
    Plot strategy weights for selected volatility-targeting strategies.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    weight_data = curves.dropna(subset=["weight"]).copy()

    selected = weight_data[
        weight_data["strategy"].str.contains("vol_target", case=False)
    ].copy()

    if selected.empty:
        print("No volatility-targeting strategy weights found to plot.")
        return

    # Limit to a few strategies to keep plot readable.
    selected_names = selected["strategy"].drop_duplicates().tolist()[:6]
    selected = selected[selected["strategy"].isin(selected_names)]

    plt.figure(figsize=(14, 6))

    for strategy, group in selected.groupby("strategy"):
        group = group.sort_values("date")

        plt.plot(
            group["date"],
            group["weight"],
            label=strategy,
            alpha=0.85,
        )

    plt.title("Volatility-Targeting Strategy Weights")
    plt.xlabel("Date")
    plt.ylabel("SPY portfolio weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "backtest_strategy_weights.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def plot_return_drawdown(summary: pd.DataFrame) -> None:
    """
    Plot annualized return vs maximum drawdown.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    plt.scatter(
        summary["max_drawdown"],
        summary["annualized_return"],
        alpha=0.8,
    )

    for _, row in summary.iterrows():
        plt.annotate(
            row["strategy"],
            (row["max_drawdown"], row["annualized_return"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title("Strategy Return vs Drawdown")
    plt.xlabel("Maximum drawdown")
    plt.ylabel("Annualized return")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "backtest_return_vs_drawdown.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    print("Backtesting volatility-based trading strategies")
    print("================================================")

    spy_returns = load_spy_returns()
    predictions = load_predictions()
    prediction_data = pivot_predictions(predictions)

    print(f"SPY return rows:      {len(spy_returns)}")
    print(f"Prediction rows:      {len(prediction_data)}")
    print()

    summary, curves = run_model_backtests(
        prediction_data=prediction_data,
        spy_returns=spy_returns,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "backtest_strategy_summary.csv"
    curves_path = TABLES_DIR / "backtest_equity_curves.csv"

    summary = summary.sort_values("sharpe", ascending=False)

    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)

    print("Backtest summary ranked by Sharpe:")
    print(summary.to_string(index=False))
    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Saved curves to:  {curves_path}")
    print()

    plot_equity_curves(curves, summary)
    plot_strategy_weights(curves)
    plot_return_drawdown(summary)

    print()
    print("Backtest complete.")


if __name__ == "__main__":
    main()