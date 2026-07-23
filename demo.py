import joblib
import pandas as pd
import numpy as np
from data.akshare_data import get_stock_data
from data.trend_stage import TrendParams, prepare_trend_features
from features.calculators import enrich_features
from utils.config_loader import load_feature_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
from buy_points.rules import build_candidate_mask_by_rule

# ============ 参数设置 ============
symbol = "600584"
start_date = "20240101"
end_date = "20260710"
model_path = "output/buy_point_2_v2_lgbm_model.joblib"
buy_point_name = "buy_point_2"
threshold = 0.75
# ================================

# 加载模型
payload = joblib.load(model_path)
model = payload["model"]
feature_cols = payload["feature_cols"]

# 加载配置
feature_config = load_feature_config("configs/features.yaml")
all_features = feature_config.get("all_features", [])
bp_config = load_buy_point_config("configs/buy_points.yaml")
bp_def = get_buy_point_definition(bp_config, buy_point_name)
rule_name = bp_def["candidate_rule"]

# 获取数据
raw_df = get_stock_data(symbol, start_date, end_date)
if raw_df is None or raw_df.empty:
    print("无数据")
    exit()

# 计算特征
trend_df = prepare_trend_features(raw_df, TrendParams())
data = enrich_features(trend_df, all_features)
data = data.replace([np.inf, -np.inf], np.nan)

# 筛选候选买点
candidate_mask = build_candidate_mask_by_rule(data, rule_name)
data["candidate_bottom"] = candidate_mask.astype(int)

# 只对候选买点打分
candidates = data[data["candidate_bottom"] == 1].copy()

if candidates.empty:
    print("该股票在此区间内没有候选买点")
    exit()

# 确保特征列存在
for col in feature_cols:
    if col not in candidates.columns:
        candidates[col] = np.nan

# 模型预测
scores = model.predict_proba(candidates[feature_cols])[:, 1]
candidates["pred_score"] = scores
candidates["pred_label"] = (scores >= threshold).astype(int)

# 筛选预测为正的买点
positive_preds = candidates[candidates["pred_label"] == 1].copy()

print("=" * 90)
print(f"股票: {symbol}")
print(f"区间: {start_date} ~ {end_date}")
print(f"买点定义: {buy_point_name} ({rule_name})")
print(f"预测阈值: {threshold}")
print(f"候选买点总数: {len(candidates)}")
print(f"预测为正的买点数: {len(positive_preds)}")
print("=" * 90)

if not positive_preds.empty:
    display_cols = ["date", "close", "pred_score", "candidate_bottom"]
    if "label" in positive_preds.columns:
        display_cols.append("label")
    if "future_max_return_pct" in positive_preds.columns:
        display_cols.append("future_max_return_pct")
    if "future_max_drawdown_pct" in positive_preds.columns:
        display_cols.append("future_max_drawdown_pct")
    
    display_cols = [c for c in display_cols if c in positive_preds.columns]
    print(positive_preds[display_cols].to_string(index=False))
else:
    print("没有预测为正的买点")
    print("\n所有候选买点的分数:")
    print(candidates[["date", "close", "pred_score"]].to_string(index=False))
