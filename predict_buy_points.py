import argparse
import os
import time

import joblib
import pandas as pd

from data.trend_stage import get_all_a_stock_symbols
from data.local_data_store import ensure_symbol_feature_data
from utils.config_loader import load_feature_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
from buy_points.rules import build_candidate_mask_by_rule


def get_sample_feature_columns(config_path: str = "configs/features.yaml") -> list[str]:
    config = load_feature_config(config_path)
    cols = config.get("all_features", [])
    if not cols:
        raise ValueError("features.yaml 中未配置 all_features")
    return cols


def load_model_payload(model_path: str):
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload and "feature_cols" in payload:
        return payload
    raise ValueError("模型文件格式不正确，缺少 model/feature_cols")


def build_latest_feature_row(
    symbol: str,
    start_date: str,
    end_date: str,
    feature_config_path: str = "configs/features.yaml",
    buy_point_rule_name: str = "pivot_confirm_v1",
) -> pd.DataFrame:
    feature_columns = get_sample_feature_columns(feature_config_path)
    data = ensure_symbol_feature_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        feature_config_path=feature_config_path,
        auto_update=False,
    )
    if data is None or data.empty or len(data) < 90:
        return pd.DataFrame()

    # 清理无穷大
    import numpy as np
    data = data.replace([np.inf, -np.inf], np.nan)

    candidate_mask = build_candidate_mask_by_rule(data, buy_point_rule_name)

    latest = data.tail(1).copy()
    latest["candidate_bottom"] = int(candidate_mask.iloc[-1])
    latest["symbol"] = str(symbol).zfill(6)
    latest["date"] = pd.to_datetime(latest["date"])
    return latest


def predict_for_symbol(
    symbol: str,
    model_payload: dict,
    start_date: str,
    end_date: str,
    candidate_only: bool = True,
    score_threshold: float = 0.5,
    feature_config_path: str = "configs/features.yaml",
    buy_point_rule_name: str = "pivot_confirm_v1",
) -> dict | None:
    latest = build_latest_feature_row(
        symbol,
        start_date,
        end_date,
        feature_config_path=feature_config_path,
        buy_point_rule_name=buy_point_rule_name,
    )
    if latest.empty:
        return None

    if candidate_only and int(latest["candidate_bottom"].iloc[-1]) != 1:
        return None

    feature_cols = model_payload["feature_cols"]
    model = model_payload["model"]

    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = None

    X = latest[feature_cols]
    score = float(model.predict_proba(X)[:, 1][0])
    pred = int(score >= score_threshold)

    row = latest.iloc[0]
    return {
        "symbol": str(symbol).zfill(6),
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "pred_score": round(score, 6),
        "pred_label": pred,
        "candidate_bottom": int(row["candidate_bottom"]),
        "close": round(float(row["close"]), 3) if pd.notna(row["close"]) else None,
        "ma20": round(float(row["ma20"]), 3) if pd.notna(row["ma20"]) else None,
        "ma60": round(float(row["ma60"]), 3) if pd.notna(row["ma60"]) else None,
        "ret_5": round(float(row["ret_5"]), 3) if pd.notna(row["ret_5"]) else None,
        "ret_20": round(float(row["ret_20"]), 3) if pd.notna(row["ret_20"]) else None,
        "vol_ratio_5": round(float(row["vol_ratio_5"]), 3) if pd.notna(row["vol_ratio_5"]) else None,
        "turnover_today": round(float(row["turnover_today"]), 3) if pd.notna(row.get("turnover_today")) else None,
        "turnover_ratio_5": round(float(row["turnover_ratio_5"]), 3) if pd.notna(row.get("turnover_ratio_5")) else None,
        "trend_score": int(row["trend_score"]) if pd.notna(row["trend_score"]) else None,
        "breakout_strength": round(float(row["breakout_strength"]), 3) if pd.notna(row.get("breakout_strength")) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="买点模型预测与选股")
    parser.add_argument("--mode", choices=["single", "market"], default="single")
    parser.add_argument("--symbol", type=str, default="600584")
    parser.add_argument("--model-path", type=str, default="output/buy_point_2_v2_lgbm_model.joblib")
    parser.add_argument("--start", type=str, default="20210101")
    parser.add_argument("--end", type=str, default="20260710")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.75, help="预测阈值，同时控制pred_label和输出过滤")
    parser.add_argument("--all-latest", action="store_true", help="不过滤candidate_bottom，直接输出所有最新样本")
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--buy-point", type=str, default=None, help="买点定义名称")
    parser.add_argument("--buy-point-config", type=str, default="configs/buy_points.yaml")
    parser.add_argument("--output", type=str, default="output/buy_point_live_predictions.csv")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    payload = load_model_payload(args.model_path)

    # 优先从模型元数据读取配置
    buy_point_name = args.buy_point or payload.get("buy_point_name")
    feature_config_path = args.feature_config or payload.get("feature_config_path", "configs/features.yaml")
    score_threshold = args.min_score

    buy_point_config = load_buy_point_config(args.buy_point_config)
    buy_point_def = get_buy_point_definition(buy_point_config, buy_point_name)
    rule_name = buy_point_def["candidate_rule"]

    print(f"模型文件: {args.model_path}")
    print(f"模型类型: {payload.get('model_type', 'unknown')}")
    print(f"当前买点定义: {buy_point_def['name']}")
    print(f"候选规则: {rule_name}")
    print(f"特征配置: {feature_config_path}")
    print(f"预测阈值: {score_threshold}")
    print(f"Top N: {args.top_n}")
    print("-" * 80, flush=True)

    candidate_only = not args.all_latest
    rows: list[dict] = []

    if args.mode == "single":
        row = predict_for_symbol(
            args.symbol,
            payload,
            args.start,
            args.end,
            candidate_only=candidate_only,
            score_threshold=score_threshold,
            feature_config_path=feature_config_path,
            buy_point_rule_name=rule_name,
        )
        if row and row["pred_score"] >= score_threshold:
            rows.append(row)
    else:
        symbols = get_all_a_stock_symbols()
        if args.limit and args.limit > 0:
            symbols = symbols[:args.limit]
        total = len(symbols)
        for idx, symbol in enumerate(symbols, start=1):
            try:
                print(f"[{idx}/{total}] 预测 {symbol}", flush=True)
                row = predict_for_symbol(
                    symbol,
                    payload,
                    args.start,
                    args.end,
                    candidate_only=candidate_only,
                    score_threshold=score_threshold,
                    feature_config_path=feature_config_path,
                    buy_point_rule_name=rule_name,
                )
                if row and row["pred_score"] >= score_threshold:
                    rows.append(row)
            except Exception as e:
                print(f"  -> 跳过 {symbol}: {e}", flush=True)
            finally:
                time.sleep(args.delay)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pred_score", "trend_score"], ascending=[False, False]).reset_index(drop=True)
        if args.top_n > 0:
            result = result.head(args.top_n).copy()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"输出记录数: {len(result)}")
    if not result.empty:
        print(result.to_string(index=False))
    else:
        print("无满足条件的结果")
    print(f"输出文件: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()