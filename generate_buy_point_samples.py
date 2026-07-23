import argparse
import os
import time

import numpy as np
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


def compute_future_label(
    df: pd.DataFrame,
    idx: int,
    future_return_days: int = 20,
    future_drawdown_days: int = 10,
    target_return_pct: float = 15.0,
    max_drawdown_pct: float = 8.0,
) -> dict:
    future_max_end = min(len(df) - 1, idx + future_return_days)
    future_dd_end = min(len(df) - 1, idx + future_drawdown_days)
    if idx >= len(df) - 1:
        return {"future_max_return_pct": None, "future_max_drawdown_pct": None, "label": None}

    entry_close = float(df.loc[idx, "close"])
    future_high = float(df.loc[idx + 1:future_max_end, "high"].max()) if idx + 1 <= future_max_end else entry_close
    future_low = float(df.loc[idx + 1:future_dd_end, "low"].min()) if idx + 1 <= future_dd_end else entry_close

    future_max_return_pct = (future_high / entry_close - 1) * 100
    future_max_drawdown_pct = (future_low / entry_close - 1) * 100
    label = int(future_max_return_pct >= target_return_pct and future_max_drawdown_pct >= -max_drawdown_pct)
    return {
        "future_max_return_pct": round(future_max_return_pct, 2),
        "future_max_drawdown_pct": round(future_max_drawdown_pct, 2),
        "label": label,
    }


def build_vshape_confirm_mask(df: pd.DataFrame) -> pd.Series:
    """买点3专用：只使用当下及历史信息构造 V 形确认点。"""
    day_pct_change = (df["close"] / df["close"].shift(1) - 1) * 100
    drop_from_high_20 = (df["close"] / df["high"].rolling(20).max() - 1) * 100
    local_low_recent = (
        ((df["is_local_low_10"] == 1) | (df["is_local_low_20"] == 1))
        .rolling(3, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
        == 1
    )
    not_break_new_low = df["low"] >= df["low"].shift(1).rolling(3, min_periods=1).min().fillna(df["low"]) * 0.995
    stop_fall = day_pct_change > -2
    rebound = (df["is_red"] == 1) | (df["close"] > df["close"].shift(1))
    prev_high_3 = df["high"].shift(1).rolling(3, min_periods=1).max()
    breakout = df["close"] >= prev_high_3.fillna(df["close"])
    ma_turn = df["ma5_slope_3"] > 0
    support_signal = (
        (df["lower_shadow_pct"] >= 15)
        | (df["vol_ratio_5"] >= 1.1)
        | (df["turnover_ratio_5"] >= 1.05)
    )

    score = (
        local_low_recent.astype(int)
        + not_break_new_low.astype(int)
        + stop_fall.astype(int)
        + rebound.astype(int)
        + breakout.astype(int)
        + ma_turn.astype(int)
        + support_signal.astype(int)
    )

    raw_mask = (
        (drop_from_high_20 <= -15)
        & local_low_recent
        & stop_fall
        & rebound
        & ((breakout) | (ma_turn))
        & (score >= 5)
    )
    return raw_mask.fillna(False)


def compress_event_mask(mask: pd.Series, cooldown_days: int = 7) -> pd.Series:
    """同一波反转只保留最早确认点，避免重复打样。"""
    result = pd.Series(False, index=mask.index)
    last_keep = -10**9
    for idx, flag in enumerate(mask.fillna(False).astype(bool).tolist()):
        if not flag:
            continue
        if idx - last_keep <= cooldown_days:
            continue
        result.iloc[idx] = True
        last_keep = idx
    return result


def build_row(
    data: pd.DataFrame,
    idx: int,
    symbol: str,
    feature_columns: list[str],
    label_info: dict,
    candidate_bottom: int,
    sample_source: str,
) -> dict:
    row = {
        "symbol": str(symbol).zfill(6),
        "date": pd.Timestamp(data.loc[idx, "date"]).strftime("%Y-%m-%d"),
        "candidate_bottom": int(candidate_bottom),
        "sample_source": sample_source,
    }
    for col in feature_columns:
        value = data.loc[idx, col] if col in data.columns else None
        if pd.isna(value):
            value = None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            value = round(float(value), 6)
        row[col] = value
    row.update(label_info)
    return row


def generate_bp3_samples_for_symbol(
    data: pd.DataFrame,
    symbol: str,
    feature_columns: list[str],
    future_return_days: int,
    future_drawdown_days: int,
    target_return_pct: float,
    max_drawdown_pct: float,
    random_negative_ratio: float = 1.0,
) -> pd.DataFrame:
    """买点3：不过候选筛选，改为“确认型正样本 + 困难负样本 + 随机背景负样本”。"""
    confirm_mask = compress_event_mask(build_vshape_confirm_mask(data), cooldown_days=7)
    rows: list[dict] = []

    positive_indices: list[int] = []
    hard_negative_indices: list[int] = []
    background_negative_candidates: list[int] = []

    for idx in range(len(data)):
        label_info = compute_future_label(
            data,
            idx,
            future_return_days,
            future_drawdown_days,
            target_return_pct,
            max_drawdown_pct,
        )
        if label_info["label"] is None:
            continue

        is_confirm = bool(confirm_mask.iloc[idx])
        if is_confirm and int(label_info["label"]) == 1:
            positive_indices.append(idx)
        elif is_confirm and int(label_info["label"]) == 0:
            hard_negative_indices.append(idx)
        elif (not is_confirm) and int(label_info["label"]) == 0:
            background_negative_candidates.append(idx)

    rng_seed = int(str(symbol).zfill(6)) if str(symbol).isdigit() else sum(ord(ch) for ch in str(symbol))
    rng = np.random.default_rng(rng_seed)

    target_random_negatives = int(max(1, round(max(len(positive_indices), len(hard_negative_indices), 1) * random_negative_ratio)))
    target_random_negatives = min(target_random_negatives, len(background_negative_candidates))
    random_negative_indices = []
    if target_random_negatives > 0:
        random_negative_indices = sorted(
            rng.choice(background_negative_candidates, size=target_random_negatives, replace=False).tolist()
        )

    for idx in positive_indices:
        label_info = compute_future_label(
            data,
            idx,
            future_return_days,
            future_drawdown_days,
            target_return_pct,
            max_drawdown_pct,
        )
        rows.append(build_row(data, idx, symbol, feature_columns, label_info, 1, "bp3_positive_confirm"))

    for idx in hard_negative_indices:
        label_info = compute_future_label(
            data,
            idx,
            future_return_days,
            future_drawdown_days,
            target_return_pct,
            max_drawdown_pct,
        )
        rows.append(build_row(data, idx, symbol, feature_columns, label_info, 1, "bp3_hard_negative"))

    for idx in random_negative_indices:
        label_info = compute_future_label(
            data,
            idx,
            future_return_days,
            future_drawdown_days,
            target_return_pct,
            max_drawdown_pct,
        )
        rows.append(build_row(data, idx, symbol, feature_columns, label_info, 0, "bp3_random_negative"))

    return pd.DataFrame(rows)


def generate_samples_for_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    future_return_days: int,
    future_drawdown_days: int,
    target_return_pct: float,
    max_drawdown_pct: float,
    only_candidates: bool = True,
    feature_config_path: str = "configs/features.yaml",
    buy_point_rule_name: str = "bottom_reversal_v1",
    buy_point_name: str | None = None,
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

    if buy_point_name == "buy_point_3":
        return generate_bp3_samples_for_symbol(
            data=data,
            symbol=symbol,
            feature_columns=feature_columns,
            future_return_days=future_return_days,
            future_drawdown_days=future_drawdown_days,
            target_return_pct=target_return_pct,
            max_drawdown_pct=max_drawdown_pct,
        )

    candidate_mask = build_candidate_mask_by_rule(data, buy_point_rule_name)

    rows = []
    for idx in range(len(data)):
        if only_candidates and not bool(candidate_mask.iloc[idx]):
            continue

        label_info = compute_future_label(
            data, idx, future_return_days, future_drawdown_days, target_return_pct, max_drawdown_pct
        )
        if label_info["label"] is None:
            continue

        rows.append(
            build_row(
                data,
                idx,
                symbol,
                feature_columns,
                label_info,
                int(candidate_mask.iloc[idx]),
                "candidate_sample",
            )
        )

    return pd.DataFrame(rows)


from pandas.errors import EmptyDataError


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path, dtype={"symbol": str})
    except (EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def normalize_symbol_col(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return df


def merge_and_save_results(existing_path: str, new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    existing = normalize_symbol_col(load_csv_safe(existing_path))
    new_df = pd.concat([f for f in new_frames if not f.empty], ignore_index=True) if new_frames else pd.DataFrame()
    new_df = normalize_symbol_col(new_df)

    if not new_df.empty and "label" in new_df.columns:
        new_df = new_df.dropna(subset=["label"]).copy()
        new_df["label"] = new_df["label"].astype(int)

    if not existing.empty and not new_df.empty:
        result = pd.concat([existing, new_df], ignore_index=True)
        result = result.drop_duplicates(subset=["symbol", "date"], keep="last")
    elif not existing.empty:
        result = existing
    else:
        result = new_df

    os.makedirs(os.path.dirname(existing_path) or ".", exist_ok=True)
    result.to_csv(existing_path, index=False, encoding="utf-8-sig")
    return result


def merge_and_save_errors(error_path: str, new_errors: list[dict]) -> pd.DataFrame:
    existing = normalize_symbol_col(load_csv_safe(error_path))
    new_df = normalize_symbol_col(pd.DataFrame(new_errors))
    if not existing.empty and not new_df.empty:
        result = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["symbol", "error"], keep="last")
    elif not existing.empty:
        result = existing
    else:
        result = new_df
    os.makedirs(os.path.dirname(error_path) or ".", exist_ok=True)
    result.to_csv(error_path, index=False, encoding="utf-8-sig")
    return result


def load_processed_symbols(output_path: str, error_log: str) -> set[str]:
    processed = set()
    df1 = normalize_symbol_col(load_csv_safe(output_path))
    df2 = normalize_symbol_col(load_csv_safe(error_log))
    if not df1.empty and "symbol" in df1.columns:
        processed.update(df1["symbol"].tolist())
    if not df2.empty and "symbol" in df2.columns:
        processed.update(df2["symbol"].tolist())
    return processed


def append_error(errors: list[dict], symbol: str, msg: str):
    errors.append({"symbol": str(symbol).zfill(6), "error": msg})


def main() -> None:
    parser = argparse.ArgumentParser(description="生成买点模型训练样本（单进程分段安全跑版）")
    parser.add_argument("--mode", choices=["single", "market"], default="single")
    parser.add_argument("--symbol", type=str, default="600584")
    parser.add_argument("--start", type=str, default="20210101")
    parser.add_argument("--end", type=str, default="20260701")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--future-return-days", type=int, default=20)
    parser.add_argument("--future-drawdown-days", type=int, default=10)
    parser.add_argument("--target-return-pct", type=float, default=15.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--all-days", action="store_true")
    parser.add_argument("--output", type=str, default="output/buy_point_training_samples.csv")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--error-log", type=str, default="output/buy_point_training_errors.csv")
    parser.add_argument("--warn-fail-streak", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--buy-point", type=str, default=None, help="买点定义名称，对应 configs/buy_points.yaml")
    parser.add_argument("--buy-point-config", type=str, default="configs/buy_points.yaml")
    args = parser.parse_args()

    buy_point_config = load_buy_point_config(args.buy_point_config)
    buy_point_def = get_buy_point_definition(buy_point_config, args.buy_point)

    rule_name = buy_point_def["candidate_rule"]
    label_rule = buy_point_def["label_rule"]

    future_return_days = label_rule["future_return_days"]
    future_drawdown_days = label_rule["future_drawdown_days"]
    target_return_pct = label_rule["target_return_pct"]
    max_drawdown_pct = label_rule["max_drawdown_pct"]

    print(f"当前买点定义: {buy_point_def['name']}")
    print(f"候选规则: {rule_name}")
    print(f"标签规则: {label_rule}")
    if buy_point_def["name"] == "buy_point_3":
        print("样本策略: 买点3启用‘确认型正样本 + 困难负样本 + 随机背景负样本’，预测/训练均不过候选筛选")

    only_candidates = not args.all_days
    if buy_point_def["name"] == "buy_point_3":
        only_candidates = False
    batch_frames: list[pd.DataFrame] = []
    batch_errors: list[dict] = []
    fail_streak = 0

    if args.mode == "single":
        try:
            df = generate_samples_for_symbol(
                str(args.symbol).zfill(6),
                args.start,
                args.end,
                future_return_days,
                future_drawdown_days,
                target_return_pct,
                max_drawdown_pct,
                only_candidates,
                args.feature_config,
                rule_name,
                buy_point_def["name"],
            )
            if not df.empty:
                batch_frames.append(df)
        except Exception as e:
            append_error(batch_errors, args.symbol, str(e))
    else:
        symbols = get_all_a_stock_symbols()
        if args.limit and args.limit > 0:
            symbols = symbols[:args.limit]

        total_all = len(symbols)
        start_idx = max(1, args.start_index)
        end_idx = args.end_index if args.end_index and args.end_index > 0 else total_all
        end_idx = min(end_idx, total_all)
        symbols = [str(s).zfill(6) for s in symbols[start_idx - 1:end_idx]]

        processed_symbols = load_processed_symbols(args.output, args.error_log) if args.resume else set()
        if args.resume:
            before = len(symbols)
            symbols = [s for s in symbols if s not in processed_symbols]
            print(f"resume 已启用：区间内原股票数 {before}，已跳过 {before - len(symbols)}，待处理 {len(symbols)}", flush=True)

        total = len(symbols)
        processed = 0

        for idx, symbol in enumerate(symbols, start=1):
            print(f"[{idx}/{total}] 处理 {symbol}", flush=True)
            try:
                df = generate_samples_for_symbol(
                    symbol,
                    args.start,
                    args.end,
                    future_return_days,
                    future_drawdown_days,
                    target_return_pct,
                    max_drawdown_pct,
                    only_candidates,
                    args.feature_config,
                    rule_name,
                    buy_point_def["name"],
                )
                if not df.empty:
                    batch_frames.append(df)
                fail_streak = 0
            except Exception as e:
                msg = str(e)
                print(f"  -> 跳过 {symbol}: {msg}", flush=True)
                append_error(batch_errors, symbol, msg)
                fail_streak += 1
                if args.warn_fail_streak > 0 and fail_streak >= args.warn_fail_streak:
                    print(f"  !! 警告：已连续失败 {fail_streak} 只，请检查网络或数据源状态", flush=True)

            processed += 1
            if args.save_every > 0 and processed % args.save_every == 0:
                current = merge_and_save_results(args.output, batch_frames)
                all_err = merge_and_save_errors(args.error_log, batch_errors)
                batch_frames = []
                batch_errors = []
                print(f"  -> 已自动保存，当前样本数: {len(current)}，错误数: {len(all_err)}", flush=True)

            time.sleep(args.delay)

    result = merge_and_save_results(args.output, batch_frames)
    all_err = merge_and_save_errors(args.error_log, batch_errors)

    print("=" * 80, flush=True)
    print(f"样本数: {len(result)}", flush=True)
    if not result.empty and "label" in result.columns:
        print(f"正样本数: {int((result['label'] == 1).sum())}", flush=True)
        print(f"负样本数: {int((result['label'] == 0).sum())}", flush=True)
        print(f"正样本占比: {(result['label'] == 1).mean() * 100:.2f}%", flush=True)
    if not result.empty and "sample_source" in result.columns:
        print("样本来源分布:", flush=True)
        print(result["sample_source"].value_counts(dropna=False).to_string(), flush=True)
    print(f"输出文件: {args.output}", flush=True)
    print(f"错误日志: {args.error_log}", flush=True)
    print(f"错误股票数: {len(all_err)}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
