import argparse
import os
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

from data.akshare_data import get_stock_data
from data.trend_stage import prepare_trend_features, TrendParams
from features.calculators import enrich_features
from utils.config_loader import load_feature_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
from buy_points.rules import build_candidate_mask_by_rule
from generate_buy_point_samples import compute_future_label


DEFAULT_PARAMS = TrendParams()


def get_sample_feature_columns(config_path: str = "configs/features.yaml") -> list[str]:
    config = load_feature_config(config_path)
    cols = config.get("all_features", [])
    if not cols:
        raise ValueError("features.yaml 中未配置 all_features")
    return cols


def load_buy_points(
    symbol: str,
    start_date: str,
    end_date: str,
    buy_point_name: str | None = None,
    buy_point_config_path: str = "configs/buy_points.yaml",
    feature_config_path: str = "configs/features.yaml",
    positive_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = get_stock_data(symbol, start_date, end_date)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    buy_point_config = load_buy_point_config(buy_point_config_path)
    buy_point_def = get_buy_point_definition(buy_point_config, buy_point_name)

    rule_name = buy_point_def["candidate_rule"]
    label_rule = buy_point_def["label_rule"]

    trend_df = prepare_trend_features(raw_df, DEFAULT_PARAMS)
    feature_columns = get_sample_feature_columns(feature_config_path)
    data = enrich_features(trend_df, feature_columns)

    candidate_mask = build_candidate_mask_by_rule(data, rule_name)
    data["candidate_bottom"] = candidate_mask.astype(int)

    rows = []
    for idx in range(len(data)):
        if int(data.loc[idx, "candidate_bottom"]) != 1:
            continue

        label_info = compute_future_label(
            data,
            idx,
            future_return_days=label_rule["future_return_days"],
            future_drawdown_days=label_rule["future_drawdown_days"],
            target_return_pct=label_rule["target_return_pct"],
            max_drawdown_pct=label_rule["max_drawdown_pct"],
        )

        if label_info["label"] is None:
            continue

        row = {
            "date": pd.Timestamp(data.loc[idx, "date"]),
            "label": int(label_info["label"]),
            "future_max_return_pct": label_info["future_max_return_pct"],
            "future_max_drawdown_pct": label_info["future_max_drawdown_pct"],
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if positive_only and not result.empty:
        result = result[result["label"] == 1].copy()

    return raw_df, result


def plot_buy_points(
    symbol: str,
    start_date: str,
    end_date: str,
    buy_point_name: str | None = None,
    buy_point_config_path: str = "configs/buy_points.yaml",
    feature_config_path: str = "configs/features.yaml",
    output_path: str | None = None,
    positive_only: bool = True,
    show_plot: bool = True,
):
    raw_df, buy_points = load_buy_points(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        buy_point_name=buy_point_name,
        buy_point_config_path=buy_point_config_path,
        feature_config_path=feature_config_path,
        positive_only=positive_only,
    )

    if raw_df.empty:
        raise ValueError("未获取到股票数据")

    plot_df = raw_df.copy().rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df = plot_df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

    marker = pd.Series(index=plot_df.index, dtype=float)
    if not buy_points.empty:
        source = raw_df.set_index(pd.to_datetime(raw_df["date"]))
        for dt in buy_points["date"]:
            ts = pd.Timestamp(dt)
            if ts in source.index:
                marker.loc[ts] = float(source.loc[ts, "low"]) * 0.98

    addplots = []
    if marker.notna().any():
        addplots.append(
            mpf.make_addplot(marker, type="scatter", markersize=120, marker="^", color="red")
        )

    title = f"{symbol} 买点标注 {start_date}~{end_date}"
    if buy_point_name:
        title += f" | {buy_point_name}"

    kwargs = dict(
    type="candle",
    volume=True,
    mav=(5, 10, 20, 60),
    style="charles",
    title=title,
    ylabel="Price",
    ylabel_lower="Volume",
    figsize=(16, 9),
    warn_too_much_data=10000,
    returnfig=True,
)

    if addplots:
        kwargs["addplot"] = addplots

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        kwargs["savefig"] = output_path

    fig, _ = mpf.plot(plot_df, **kwargs)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return buy_points


def main():
    parser = argparse.ArgumentParser(description="按股票代码和时间区间绘制买点标注K线图")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--buy-point", type=str, default=None, help="买点定义名称，如 buy_point_1 / buy_point_2")
    parser.add_argument("--buy-point-config", type=str, default="configs/buy_points.yaml")
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--output", type=str, default=None, help="可选，保存图片路径")
    parser.add_argument("--include-all-candidates", action="store_true", help="显示所有候选买点，不只正样本")
    parser.add_argument("--no-show", action="store_true", help="不直接展示图形，仅保存")
    args = parser.parse_args()

    buy_points = plot_buy_points(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        buy_point_name=args.buy_point,
        buy_point_config_path=args.buy_point_config,
        feature_config_path=args.feature_config,
        output_path=args.output,
        positive_only=not args.include_all_candidates,
        show_plot=not args.no_show,
    )

    print("=" * 80)
    print(f"股票: {args.symbol}")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"买点策略: {args.buy_point or 'default_buy_point'}")
    print(f"买点数量: {len(buy_points)}")
    if not buy_points.empty:
        print(buy_points.to_string(index=False))
    if args.output:
        print(f"输出图片: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()