import argparse
import os
import time
import joblib
import pandas as pd
import numpy as np

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


def load_multi_models(model_paths: list[str]) -> list[dict]:
    payloads = []
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        payload = joblib.load(path)
        if not (isinstance(payload, dict) and "model" in payload and "feature_cols" in payload):
            raise ValueError(f"模型文件格式不正确: {path}")
        payloads.append(payload)
    return payloads


def predict_one_stock_multi_models(
    symbol: str,
    payloads: list[dict],
    start_date: str,
    end_date: str,
    score_threshold: float,
    all_feature_cols: list[str],
    buy_point_configs: dict[str, str],
    skip_candidate_filter: bool = False,
    feature_config_path: str = "configs/features.yaml",
) -> list[dict]:
    # 1. 直接读取本地最新特征缓存
    data = ensure_symbol_feature_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        feature_config_path=feature_config_path,
        auto_update=False,
    )
    if data is None or data.empty or len(data) < 90:
        return []

    # 2. 清理无穷大并取最新一行
    data = data.replace([np.inf, -np.inf], np.nan)

    latest = data.tail(1).copy()
    latest["symbol"] = str(symbol).zfill(6)
    latest["date"] = pd.to_datetime(latest["date"])

    results = []

    # 4. 针对每个模型分别打分
    for payload in payloads:
        bp_name = payload.get("buy_point_name", "default")
        model = payload["model"]
        feature_cols = payload["feature_cols"]
        
        # 确保特征列存在
        for col in feature_cols:
            if col not in latest.columns:
                latest[col] = np.nan

        # 如果该模型有对应的买点规则，先过一层候选筛选
        rule_name = buy_point_configs.get(bp_name)
        if rule_name and not skip_candidate_filter:
            candidate_mask = build_candidate_mask_by_rule(data, rule_name)
            is_candidate = int(candidate_mask.iloc[-1])
            if is_candidate != 1:
                continue  # 不符合该模型的候选规则，跳过该模型
        else:
            is_candidate = 1 if skip_candidate_filter else 0

        # 模型打分
        try:
            #score = float(model.predict_proba(latest[feature_cols])[:, 1][0])
            X = latest.loc[:, feature_cols].copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            X.columns = [str(c) for c in feature_cols]

            if hasattr(model, "feature_name_") and model.feature_name_ is not None:
                train_feature_names = [str(c) for c in model.feature_name_]
                if len(train_feature_names) == len(X.columns):
                    X.columns = train_feature_names
                    X = X.loc[:, train_feature_names]

            score = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            continue

        if score >= score_threshold:
            row = latest.iloc[0]
            results.append({
                "buy_point": bp_name,
                "symbol": str(symbol).zfill(6),
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "pred_score": round(score, 6),
                "candidate_bottom": is_candidate,
                "close": round(float(row["close"]), 3) if pd.notna(row["close"]) else None,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="多模型联合预测（单次拉取数据）")
    parser.add_argument("--mode", choices=["single", "market"], default="market")
    parser.add_argument("--symbol", type=str, default="600584")
    parser.add_argument("--model-paths", type=str, required=True, help="逗号分隔的模型路径")
    parser.add_argument("--start", type=str, default="20210101")
    parser.add_argument("--end", type=str, default="20260714")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--buy-point-config", type=str, default="configs/buy_points.yaml")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-candidate-filter", action="store_true", help="全局跳过候选买点过滤，直接对所有股票做模型打分")
    args = parser.parse_args()

    # 加载模型
    model_paths = [p.strip() for p in args.model_paths.split(",")]
    payloads = load_multi_models(model_paths)
    print(f"已加载 {len(payloads)} 个模型: {[p.get('buy_point_name', 'N/A') for p in payloads]}")
    print(f"是否跳过候选过滤: {args.skip_candidate_filter}")

    # 读取全局特征列表
    all_feature_cols = get_sample_feature_columns(args.feature_config)

    # 读取买点配置
    bp_config_data = load_buy_point_config(args.buy_point_config)
    buy_point_rules = {}
    for payload in payloads:
        bp_name = payload.get("buy_point_name")
        if bp_name:
            try:
                bp_def = get_buy_point_definition(bp_config_data, bp_name)
                buy_point_rules[bp_name] = bp_def["candidate_rule"]
            except Exception:
                pass

    # 准备输出文件
    os.makedirs(args.output_dir, exist_ok=True)
    output_files = {p.get("buy_point_name", f"model_{i}"): os.path.join(args.output_dir, f"buy_point_{p.get('buy_point_name', f'model_{i}')}_live_predictions.csv") for i, p in enumerate(payloads)}
    
    # resume 逻辑
    done_symbols = set()
    if args.resume:
        for bp_name, f_path in output_files.items():
            if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
                try:
                    existing = pd.read_csv(f_path, dtype={"symbol": str})
                    existing["symbol"] = existing["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                    done_symbols.update(existing["symbol"].tolist())
                except Exception:
                    pass
        print(f"resume 已启用：跳过 {len(done_symbols)} 只已处理股票")

    rows_all = {bp: [] for bp in output_files.keys()}

    if args.mode == "single":
        results = predict_one_stock_multi_models(args.symbol, payloads, args.start, args.end, args.min_score, all_feature_cols, buy_point_rules, args.skip_candidate_filter, args.feature_config)
        for r in results:
            if r["buy_point"] in rows_all:
                rows_all[r["buy_point"]].append(r)
    else:
        symbols = get_all_a_stock_symbols()
        if args.limit > 0:
            symbols = symbols[:args.limit]
        
        total = len(symbols)
        processed = 0
        
        for idx, symbol in enumerate(symbols, start=1):
            symbol = str(symbol).zfill(6)
            if symbol in done_symbols:
                continue

            print(f"[{idx}/{total}] 预测 {symbol}", flush=True)
            try:
                results = predict_one_stock_multi_models(symbol, payloads, args.start, args.end, args.min_score, all_feature_cols, buy_point_rules, args.skip_candidate_filter, args.feature_config)
                for r in results:
                    if r["buy_point"] in rows_all:
                        rows_all[r["buy_point"]].append(r)
            except Exception as e:
                print(f"  -> 跳过 {symbol}: {e}", flush=True)
            
            processed += 1
            if processed % 100 == 0:
                for bp_name, r_list in rows_all.items():
                    temp_df = pd.DataFrame(r_list)
                    if not temp_df.empty:
                        temp_df = temp_df.sort_values("pred_score", ascending=False).head(args.top_n * 2)
                    temp_df.to_csv(output_files[bp_name], index=False, encoding="utf-8-sig")
                print(f"  -> 已自动保存", flush=True)

            time.sleep(args.delay)

    # 最终保存
    for bp_name, r_list in rows_all.items():
        result_df = pd.DataFrame(r_list)
        if not result_df.empty:
            result_df = result_df.sort_values("pred_score", ascending=False).head(args.top_n).reset_index(drop=True)
        
        # resume 合并旧数据
        if args.resume and os.path.exists(output_files[bp_name]) and os.path.getsize(output_files[bp_name]) > 0:
            try:
                existing = pd.read_csv(output_files[bp_name], dtype={"symbol": str})
                existing["symbol"] = existing["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                result_df = pd.concat([existing, result_df], ignore_index=True)
                result_df = result_df.drop_duplicates(subset=["symbol"], keep="last")
                result_df = result_df.sort_values("pred_score", ascending=False).head(args.top_n).reset_index(drop=True)
            except Exception:
                pass

        result_df.to_csv(output_files[bp_name], index=False, encoding="utf-8-sig")
        print(f"[{bp_name}] 输出记录数: {len(result_df)} -> {output_files[bp_name]}")


if __name__ == "__main__":
    main()
