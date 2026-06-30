import numpy as np
import pandas as pd

from kan_volatility.config import TABLES_DIR

ALL_MODELS_SUMMARY_FILE = "all_models_summary.csv"
STRATEGY_TUNING_SUMMARY_FILE = "strategy_tuning_summary.csv"

def load_csv(filename: str) -> pd.DataFrame:
    path = TABLES_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure the previous scripts have been run."
        )
    
    return pd.read_csv(path)

def format_percent(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    
    return f"{100 * value:.{decimals}f}%"

def format_float(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return ""
    
    return f"{value:.{decimals}f}"

def clean_model_name(name: str) -> str:
    replacements = {
        "naive_spy_rv_5d": "Naive RV(5d)",
        "linear_regression": "Linear Regression",
        "ridge_alpha_1": "Ridge α=1",
        "ridge_alpha_10": "Ridge α=10",
        "random_forest": "Random Forest",
        "mlp_64_32": "MLP [64, 32]",
        "efficient_kan_16_8_grid5": "EfficientKAN [16, 8], grid=5",
        "best_efficient_kan_16_8_grid3_wd1em04": (
            "Tuned EfficientKAN [16, 8], grid=3"
        ),
    }

    return replacements.get(name, name)

def create_forecasting_model_ranking(models: pd.DataFrame) -> pd.DataFrame:
    """
    Create clean forecasting model ranking table.
    """
    ranking = models.copy()

    ranking = ranking.sort_values("test_rmse").reset_index(drop=True)
    ranking.insert(0, "rank_by_test_rmse", np.arange(1, len(ranking) + 1))

    output = pd.DataFrame(
        {
            "Rank": ranking["rank_by_test_rmse"],
            "Model": ranking["model"].map(clean_model_name),
            "Validation RMSE": ranking["val_rmse"].map(lambda x: format_float(x, 4)),
            "Test RMSE": ranking["test_rmse"].map(lambda x: format_float(x, 4)),
            "Test MAE": ranking["test_mae"].map(lambda x: format_float(x, 4)),
            "Test R²": ranking["test_r2"].map(lambda x: format_float(x, 4)),
            "Test Directional Accuracy": ranking[
                "test_directional_accuracy"
            ].map(lambda x: format_percent(x, 2)),
        }
    )

    return output

def create_top_strategy_ranking(strategies: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Create clean top-strategy ranking table by Sharpe ratio.
    """
    ranking = strategies.sort_values("sharpe", ascending=False).head(top_n).copy()
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, "rank_by_sharpe", np.arange(1, len(ranking) + 1))

    output = pd.DataFrame(
        {
            "Rank": ranking["rank_by_sharpe"],
            "Strategy": ranking["strategy"],
            "Model": ranking["model"].map(clean_model_name),
            "Type": ranking["strategy_type"],
            "Annualized Return": ranking["annualized_return"].map(format_percent),
            "Annualized Volatility": ranking["annualized_volatility"].map(
                format_percent
            ),
            "Sharpe": ranking["sharpe"].map(lambda x: format_float(x, 3)),
            "Sortino": ranking["sortino"].map(lambda x: format_float(x, 3)),
            "Max Drawdown": ranking["max_drawdown"].map(format_percent),
            "Average Weight": ranking["average_weight"].map(format_percent),
            "Total Turnover": ranking["total_turnover"].map(lambda x: format_float(x, 2)),
            "Target Volatility": ranking["target_volatility"].map(format_percent),
            "Transaction Cost bps": ranking["transaction_cost_bps"].map(
                lambda x: format_float(x, 1)
            ),
            "Smoothing α": ranking["smoothing_alpha"].map(lambda x: format_float(x, 2)),
            "Rebalance Buffer": ranking["rebalance_buffer"].map(format_percent),
            "Threshold": ranking["threshold"].map(format_percent),
        }
    )

    return output

def get_strategy_row(strategies: pd.DataFrame, strategy_name_contains: str) -> pd.Series:
    matches = strategies[
        strategies["strategy"].str.contains(strategy_name_contains, case=False, na=False)
    ].copy()

    if matches.empty:
        raise ValueError(f"No strategy found containing: {strategy_name_contains}")
    
    return matches.sort_values("sharpe", ascending=False).iloc[0]

def get_buy_and_hold_row(strategies: pd.DataFrame) -> pd.Series:
    matches = strategies[strategies["strategy"] == "buy_and_hold_spy"].copy()

    if matches.empty:
        raise ValueError("Could not find buy_and_hold_spy in strategy results.")
    
    return matches.iloc[0]


def create_strategy_comparison_table(
    strategies: pd.DataFrame,
    comparison_strategy: pd.Series,
    comparison_label: str,
) -> pd.DataFrame:
    """
    Compare buy-and-hold against one selected strategy.
    """
    buy_hold = get_buy_and_hold_row(strategies)

    rows = [
        ("Buy-and-Hold SPY", buy_hold),
        (comparison_label, comparison_strategy),
    ]

    output = pd.DataFrame(
        {
            "Strategy": [label for label, _ in rows],
            "Annualized Return": [
                format_percent(row["annualized_return"]) for _, row in rows
            ],
            "Annualized Volatility": [
                format_percent(row["annualized_volatility"]) for _, row in rows
            ],
            "Sharpe": [format_float(row["sharpe"], 3) for _, row in rows],
            "Sortino": [format_float(row["sortino"], 3) for _, row in rows],
            "Max Drawdown": [format_percent(row["max_drawdown"]) for _, row in rows],
            "Average Weight": [format_percent(row["average_weight"]) for _, row in rows],
            "Total Turnover": [
                format_float(row["total_turnover"], 2) for _, row in rows
            ],
        }
    )

    return output

def create_key_conclusions_table(
    models: pd.DataFrame,
    strategies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a qualitative conclusions table for the report.
    """
    best_forecasting = models.sort_values("test_rmse").iloc[0]
    best_strategy = strategies.sort_values("sharpe", ascending=False).iloc[0]
    buy_hold = get_buy_and_hold_row(strategies)

    best_kan_strategy = get_strategy_row(
        strategies,
        "efficient_kan_16_8_grid5_vol_target",
    )

    best_ridge_threshold = get_strategy_row(
        strategies,
        "ridge_alpha_10_threshold",
    )

    conclusions = [
        {
            "Finding": "Best forecasting model by test RMSE",
            "Conclusion": clean_model_name(best_forecasting["model"]),
            "Evidence": (
                f"Test RMSE = {format_float(best_forecasting['test_rmse'], 4)}, "
                f"Test MAE = {format_float(best_forecasting['test_mae'], 4)}"
            ),
        },
        {
            "Finding": "Best strategy by Sharpe ratio",
            "Conclusion": best_strategy["strategy"],
            "Evidence": (
                f"Sharpe = {format_float(best_strategy['sharpe'], 3)}, "
                f"Sortino = {format_float(best_strategy['sortino'], 3)}, "
                f"Max drawdown = {format_percent(best_strategy['max_drawdown'])}"
            ),
        },
        {
            "Finding": "Best EfficientKAN strategy vs buy-and-hold",
            "Conclusion": (
                "EfficientKAN volatility targeting reduced risk and improved "
                "risk-adjusted performance, but did not maximize raw return."
            ),
            "Evidence": (
                f"EfficientKAN Sharpe = {format_float(best_kan_strategy['sharpe'], 3)} "
                f"vs buy-and-hold Sharpe = {format_float(buy_hold['sharpe'], 3)}; "
                f"drawdown = {format_percent(best_kan_strategy['max_drawdown'])} "
                f"vs {format_percent(buy_hold['max_drawdown'])}"
            ),
        },
        {
            "Finding": "Best simple strategy",
            "Conclusion": (
                "Ridge threshold strategy preserved more upside with lower turnover, "
                "but provided less drawdown protection than EfficientKAN vol targeting."
            ),
            "Evidence": (
                f"Ridge threshold annualized return = "
                f"{format_percent(best_ridge_threshold['annualized_return'])}, "
                f"Sharpe = {format_float(best_ridge_threshold['sharpe'], 3)}, "
                f"turnover = {format_float(best_ridge_threshold['total_turnover'], 2)}"
            ),
        },
        {
            "Finding": "Forecast accuracy vs trading utility",
            "Conclusion": (
                "The best forecasting model was not necessarily the best trading model."
            ),
            "Evidence": (
                "Model ranking by RMSE differed from strategy ranking by Sharpe and "
                "drawdown-adjusted performance."
            ),
        },
    ]

    return pd.DataFrame(conclusions)

def save_table(table: pd.DataFrame, filename: str) -> None:
    output_path = TABLES_DIR / filename
    table.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main() -> None:
    print("Generating report-ready tables")
    print("==============================")

    models = load_csv(ALL_MODELS_SUMMARY_FILE)
    strategies = load_csv(STRATEGY_TUNING_SUMMARY_FILE)

    forecasting_ranking = create_forecasting_model_ranking(models)
    top_strategy_ranking = create_top_strategy_ranking(strategies, top_n=20)

    best_kan_strategy = get_strategy_row(
        strategies,
        "efficient_kan_16_8_grid5_vol_target",
    )

    ridge_threshold_strategy = get_strategy_row(
        strategies,
        "ridge_alpha_10_threshold",
    )

    best_kan_vs_buy_hold = create_strategy_comparison_table(
        strategies=strategies,
        comparison_strategy=best_kan_strategy,
        comparison_label="Best EfficientKAN Vol-Target Strategy",
    )

    ridge_threshold_vs_buy_hold = create_strategy_comparison_table(
        strategies=strategies,
        comparison_strategy=ridge_threshold_strategy,
        comparison_label="Best Ridge Threshold Strategy",
    )

    key_conclusions = create_key_conclusions_table(
        models=models,
        strategies=strategies,
    )

    save_table(forecasting_ranking, "report_forecasting_model_ranking.csv")
    save_table(top_strategy_ranking, "report_top_strategy_ranking.csv")
    save_table(best_kan_vs_buy_hold, "report_best_kan_vs_buy_hold.csv")
    save_table(ridge_threshold_vs_buy_hold, "report_ridge_threshold_vs_buy_hold.csv")
    save_table(key_conclusions, "report_key_conclusions.csv")

    print()
    print("Forecasting model ranking:")
    print(forecasting_ranking.to_string(index=False))
    print()
    print("Top strategy ranking:")
    print(top_strategy_ranking.head(10).to_string(index=False))
    print()
    print("Best EfficientKAN strategy vs buy-and-hold:")
    print(best_kan_vs_buy_hold.to_string(index=False))
    print()
    print("Best Ridge threshold strategy vs buy-and-hold:")
    print(ridge_threshold_vs_buy_hold.to_string(index=False))
    print()
    print("Key conclusions:")
    print(key_conclusions.to_string(index=False))

    print()
    print("Report table generation complete.")

if __name__ == "__main__":
    main()


