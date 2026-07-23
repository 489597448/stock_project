import pandas as pd
from data.akshare_data import get_stock_data
from data.trend_stage import TrendParams, get_all_a_stock_symbols, prepare_trend_features
from features.calculators import enrich_features
from utils.config_loader import load_feature_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
from buy_points.rules import build_candidate_mask_by_rule

DEFAULT_PARAMS = TrendParams()

def get_sample_feature_columns(config_path: str = "configs/features.yaml") -> list[str]:
    config = load_feature_config(config_path)
    cols = config.get("all_features", [])
    if not cols:
        raise ValueError("features.yaml 中未配置 all_features")
    return cols

buy_point_name = "buy_point_3"
start_date = "20210101"
end_date = "20260717"

bp_config = load_buy_point_config("configs/buy_points.yaml")
bp_def = get_buy_point_definition(bp_config, buy_point_name)
rule_name = bp_def["candidate_rule"]

feature_columns = get_sample_feature_columns("configs/features.yaml")
symbols = get_all_a_stock_symbols()

rows = []

for idx, symbol in enumerate(symbols, 1):
    try:
        print(f"[{idx}/{len(symbols)}] 检查 {symbol}", flush=True)
        df = get_stock_data(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 90:
            continue

        trend_df = prepare_trend_features(df, DEFAULT_PARAMS)
        data = enrich_features(trend_df, feature_columns)

        candidate_mask = build_candidate_mask_by_rule(data, rule_name)
        if bool(candidate_mask.iloc[-1]):
            row = data.iloc[-1]
            rows.append({
                "symbol": str(symbol).zfill(6),
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "close": row["close"],
            })
    except Exception as e:
        print(f"  -> 跳过 {symbol}: {e}", flush=True)

"""
result = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
result.to_csv("output/buy_point_3_candidates_today.csv", index=False, encoding="utf-8-sig")

print("=" * 80)
print(f"今日满足 {buy_point_name} 候选点的股票数量: {len(result)}")
if not result.empty:
    print(result.to_string(index=False))
print("输出文件: output/buy_point_3_candidates_today.csv")
print("=" * 80)
"""
result = pd.DataFrame(rows)

if not result.empty and "symbol" in result.columns:
    result = result.sort_values("symbol").reset_index(drop=True)
else:
    result = pd.DataFrame(columns=["symbol", "date", "close"])

result.to_csv("output/buy_point_3_candidates_today.csv", index=False, encoding="utf-8-sig")

print("=" * 80)
print(f"今日满足 {buy_point_name} 候选点的股票数量: {len(result)}")
if not result.empty:
    print(result.to_string(index=False))
else:
    print("今天没有股票满足 buy_point_3 候选条件")
print("输出文件: output/buy_point_3_candidates_today.csv")
print("=" * 80)
