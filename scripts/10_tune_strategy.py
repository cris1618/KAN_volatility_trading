import itertools

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from kan_volatility.backtest import compute_strategy_returns, summarize_backtest
from kan_volatility.config import RAW_DATA_DIR, TABLES_DIR, FIGURES_DIR
from kan_volatility.strategy import (
    apply_rebalance_buffer,
    compute_buy_and_hold_weights,
    compute_threshold_weights,
    compute_volatility_target_weights,
    smooth_weights,
)

PREDICTION_FILES = [
    "baseline_test_predictions.csv",
    "mlp_test_predictions.csv",
    "kan_test_predictions.csv",
    "best_kan_test_predictions.csv",
]

MODELS_TO_TUNE = [
    "efficient_kan_16_8_grid5",
    "random_forest",
    "ridge_alpha_10",
    "best_efficient_kan_16_8_grid3_wd1em04",
]

TARGET_VOLATILITY_GRID = [0.08, 0.10, 0.12, 0.15]
TRANSACTION_COST_GRID = [1.0, 5.0, 10.0]
SMOOTHING_ALPHA_GRID = [1.0, 0.50, 0.25, 0.10]
REBALANCE_BUFFER_GRID = [0.00, 0.025, 0.05]
THRESHOLD_GRID = [0.20, 0.25, 0.30]

def load_spy_returns() -> pd.DataFrame:
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
            "No prediction files found. Run the model training scripts first."
        )
    
    return pd.concat(frames, ignore_index=True, sort=False)

def pivot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    pred_wide = predictions.pivot(
        index="date",
        columns="model",
        values="y_pred",
    ).reset_index()

    pred_wide["date"] = pd.to_datetime(pred_wide["date"])

    return pred_wide.sort_values("date")

def get_available_models(prediction_data: pd.DataFrame) -> list[str]:
    available = [
        model 
        for model in MODELS_TO_TUNE
        if model in prediction_data.columns
    ]

    if not available:
        available = [
            column 
            for column in prediction_data.columns
            if column != "date"
        ]

        print("None of MODELS_TO_TUNE were found exactly.")
        print("Falling back to all available models:")
        for model in available:
            print(f"  - {model}")
    
    return available

def prepare_backset_data() -> tuple[pd.DataFrame, list[str]]:
    spy_returns = load_spy_returns()
    predictions = load_predictions()
    prediction_data = pivot_predictions(predictions)

    data = prediction_data.merge(spy_returns, on="date", how="inner")
    data = data.sort_values("date").set_index("date")

    available_models = get_available_models(data.reset_index())

    print("Available models selected for strategy tuning:")
    for model in available_models:
        print(f" - {model}")
    print()

    return data, available_models

def evaluate_strategy(
    asset_returns: pd.Series,
    weights: pd.Series,
    strategy_name: str,
    transaction_cost_bps: float,
    metadata: dict,
) -> tuple[dict, pd.DataFrame]:
    results = compute_strategy_returns(
        asset_returns=asset_returns,
        weights=weights,
        transaction_cost_bps=transaction_cost_bps,
    )

    summary = summarize_backtest(
        results=results,
        strategy_name=strategy_name,
    )

    summary.update(metadata)

    curve = results[
        [
            "equity_curve",
            "benchmark_equity_curve",
            "weight",
            "turnover",
            "strategy_return",
            "benchmark_return",
        ]
    ].copy()

    curve["strategy"] = strategy_name

    for key, value in metadata.items():
        curve[key] = value
    
    return summary, curve.reset_index()

def tune_vol_target_strategies(
    data: pd.DataFrame,
    model_names: list[str],
) -> tuple[list[dict], list[pd.DataFrame]]:
    summaries = []
    curves = []

    configs = list(
        itertools.product(
            model_names,
            TARGET_VOLATILITY_GRID,
            TRANSACTION_COST_GRID,
            SMOOTHING_ALPHA_GRID,
            REBALANCE_BUFFER_GRID,
        )
    )

    print(f"Vol-target configurations: {len(configs)}")

    for (
        model_name,
        target_volatility,
        transaction_cost_bps,
        smoothing_alpha,
        rebalance_buffer,
    ) in configs:
        predicted_volatility = np.exp(data[model_name])

        raw_weights = compute_volatility_target_weights(
            predicted_volatility=predicted_volatility,
            target_volatility=target_volatility,
            max_weight=1.0,
            min_weight=0.0,
        )

        smoothed_weights = smooth_weights(
            raw_weights=raw_weights,
            smoothing_alpha=smoothing_alpha,
        )

        final_weights = apply_rebalance_buffer(
            target_weights=smoothed_weights,
            rebalance_buffer=rebalance_buffer,
        )

        strategy_name = (
            f"{model_name}_vol_target"
            f"_tv{target_volatility:.2f}"
            f"_tc{transaction_cost_bps:.0f}"
            f"_a{smoothing_alpha:.2f}"
            f"_buf{rebalance_buffer:.3f}"
        )

        metadata = {
            "model": model_name,
            "strategy_type": "vol_target",
            "target_volatility": target_volatility,
            "transaction_cost_bps": transaction_cost_bps,
            "smoothing_alpha": smoothing_alpha,
            "rebalance_buffer": rebalance_buffer,
            "threshold": np.nan,
        }

        summary, curve = evaluate_strategy(
            asset_returns=data["spy_return"],
            weights=final_weights,
            strategy_name=strategy_name,
            transaction_cost_bps=transaction_cost_bps,
            metadata=metadata,
        )

        summaries.append(summary)
        curves.append(curve)
    
    return summaries, curves

def tune_threshold_strategies(
    data: pd.DataFrame,
    model_names: list[str],
) -> tuple[list[dict], list[pd.DataFrame]]:
    summaries = []
    curves = []

    configs = list(
        itertools.product(
            model_names,
            THRESHOLD_GRID,
            TRANSACTION_COST_GRID,
            SMOOTHING_ALPHA_GRID,
            REBALANCE_BUFFER_GRID,
        )
    )

    print(f"Threshold configurations: {len(configs)}")

    for (
        model_name,
        threshold,
        transaction_cost_bps,
        smoothing_alpha,
        rebalance_buffer,
    ) in configs:
        predicted_volatility = np.exp(data[model_name])

        raw_weights = compute_threshold_weights(
            predicted_volatility=predicted_volatility,
            high_volatility_threshold=threshold,
            low_risk_weight=0.50,
            normal_risk_weight=1.00,
        )

        smoothed_weights = smooth_weights(
            raw_weights=raw_weights,
            smoothing_alpha=smoothing_alpha,
        )

        final_weights = apply_rebalance_buffer(
            target_weights=smoothed_weights,
            rebalance_buffer=rebalance_buffer,
        )

        strategy_name = (
            f"{model_name}_threshold"
            f"_thr{threshold:.2f}"
            f"_tc{transaction_cost_bps:.0f}"
            f"_a{smoothing_alpha:.2f}"
            f"_buf{rebalance_buffer:.3f}"
        )

        metadata = {
            "model": model_name,
            "strategy_type": "threshold",
            "target_volatility": np.nan,
            "transaction_cost_bps": transaction_cost_bps,
            "smoothing_alpha": smoothing_alpha,
            "rebalance_buffer": rebalance_buffer,
            "threshold": threshold,
        }

        summary, curve = evaluate_strategy(
            asset_returns=data["spy_return"],
            weights=final_weights,
            strategy_name=strategy_name,
            transaction_cost_bps=transaction_cost_bps,
            metadata=metadata,
        )

        summaries.append(summary)
        curves.append(curve)
    
    return summaries, curves

def add_buy_and_hold(
    data: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    weights = compute_buy_and_hold_weights(
        index=data.index,
        weight=1.0,
    )

    metadata = {
        "model": "buy_and_hold",
        "strategy_type": "buy_and_hold",
        "target_volatility": np.nan,
        "transaction_cost_bps": 0.0,
        "smoothing_alpha": np.nan,
        "rebalance_buffer": np.nan,
        "threshold": np.nan,
    }

    return evaluate_strategy(
        asset_returns=data["spy_return"],
        weights=weights,
        strategy_name="buy_and_hold_spy",
        transaction_cost_bps=0.0,
        metadata=metadata,
    )

def plot_top_equity_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str = "sharpe",
    top_n: int = 8,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values(metric, ascending=False)

    selected_strategies = ["buy_and_hold_spy"]

    for strategy in ranked["strategy"]:
        if strategy not in selected_strategies:
            selected_strategies.append(strategy)

        if len(selected_strategies) >= top_n:
            break

    plt.figure(figsize=(15, 7))

    for strategy in selected_strategies:
        strategy_curve = curves[curves["strategy"] == strategy].copy()
        strategy_curve = strategy_curve.sort_values("date")

        plt.plot(
            strategy_curve["date"],
            strategy_curve["equity_curve"],
            label=strategy,
            alpha=0.90,
        )

    plt.title(f"Top Strategy Equity Curves by {metric}")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"strategy_tuning_top_equity_curves_by_{metric}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")   

def plot_return_vs_drawdown(summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    plt.scatter(
        summary["max_drawdown"],
        summary["annualized_return"],
        alpha=0.6,
    )

    best = summary.sort_values("sharpe", ascending=False).head(10)

    for _, row in best.iterrows():
        plt.annotate(
            row["model"],
            (row["max_drawdown"], row["annualized_return"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title("Strategy Tuning: Return vs Drawdown")
    plt.xlabel("Maximum drawdown")
    plt.ylabel("Annualized return")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "strategy_tuning_return_vs_drawdown.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def save_top_tables(summary: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    top_by_sharpe = summary.sort_values("sharpe", ascending=False).head(25)
    top_by_sortino = summary.sort_values("sortino", ascending=False).head(25)
    top_by_drawdown = summary.sort_values("max_drawdown", ascending=False).head(25)

    top_by_sharpe.to_csv(TABLES_DIR / "strategy_tuning_top_by_sharpe.csv", index=False)
    top_by_sortino.to_csv(TABLES_DIR / "strategy_tuning_top_by_sortino.csv", index=False)
    top_by_drawdown.to_csv(
        TABLES_DIR / "strategy_tuning_top_by_drawdown.csv",
        index=False,
    )

def plot_sharpe_vs_turnover(summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = summary[summary["strategy_type"] != "buy_and_hold"].copy()

    plt.figure(figsize=(10, 7))

    plt.scatter(
        plot_data["total_turnover"],
        plot_data["sharpe"],
        alpha=0.6,
    )

    best = plot_data.sort_values("sharpe", ascending=False).head(10)

    for _, row in best.iterrows():
        plt.annotate(
            row["model"],
            (row["total_turnover"], row["sharpe"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title("Strategy Tuning: Sharpe vs Total Turnover")
    plt.xlabel("Total turnover")
    plt.ylabel("Sharpe ratio")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "strategy_tuning_sharpe_vs_turnover.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

def main() -> None:
    print("Tuning volatility strategy parameters")
    print("=====================================")

    data, model_names = prepare_backset_data()

    all_summaries = []
    all_curves = []

    buy_hold_summary, buy_hold_curve = add_buy_and_hold(data)
    all_summaries.append(buy_hold_summary)
    all_curves.append(buy_hold_curve)

    vol_target_summaries, vol_target_curves = tune_vol_target_strategies(
        data=data,
        model_names=model_names,
    )

    threshold_summaries, threshold_curves = tune_threshold_strategies(
        data=data,
        model_names=model_names,
    )

    all_summaries.extend(vol_target_summaries)
    all_summaries.extend(threshold_summaries)

    all_curves.extend(vol_target_curves)
    all_curves.extend(threshold_curves)

    bad_items = [
    (i, type(item), item)
    for i, item in enumerate(all_summaries)
    if not isinstance(item, dict)
    ]

    if bad_items:
        print("Bad summary items found:")
        for item in bad_items[:5]:
            print(item)
        raise TypeError("all_summaries contains non-dict items.")

    summary = pd.DataFrame.from_records(all_summaries)
    curves = pd.concat(all_curves, ignore_index=True, sort=False)

    summary = summary.sort_values("sharpe", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "strategy_tuning_summary.csv"
    curves_path = TABLES_DIR / "strategy_tuning_curves.csv"

    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)

    save_top_tables(summary)

    print()
    print("Top 20 strategies by Sharpe:")
    print(summary.head(20).to_string(index=False))
    print()
    print(f"Saved full strategy tuning summary to: {summary_path}")
    print(f"Saved full strategy tuning curves to:  {curves_path}")
    print()

    plot_top_equity_curves(curves, summary, metric="sharpe", top_n=8)
    plot_sharpe_vs_turnover(summary)
    plot_return_vs_drawdown(summary)

    print()
    print("Strategy tuning complete.")


if __name__ == "__main__":
    main()

