import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import akshare as ak
import mplfinance as mpf
import pandas as pd

from data.akshare_data import get_stock_data


@dataclass
class TrendParams:
    lookback_return_days: int = 60
    min_return_pct: float = 15.0
    short_ma: int = 20
    long_ma: int = 60
    slope_days: int = 5
    breakout_lookback: int = 20
    max_deviation_pct: float = 35.0
    min_segment_days: int = 8
    use_higher_low: bool = False
    score_threshold: int = 4
    smooth_window: int = 7
    smooth_min_count: int = 4
    start_lookback_days: int = 15
    end_drawdown_pct: float = 12.0
    hard_break_below_ma_pct: float = 3.0
    hard_break_confirm_days: int = 2
    pivot_window: int = 5
    pivot_start_expand_days: int = 20
    pivot_end_expand_days: int = 8
    min_segment_return_pct: float = 5.0



def get_all_a_stock_symbols() -> list[str]:
    """
    symbols: list[str] = []

    try:
        df_sh = ak.stock_info_sh_name_code()
        if df_sh is not None and not df_sh.empty and "证券代码" in df_sh.columns:
            symbols.extend(
                df_sh["证券代码"].astype(str).str.strip().str.zfill(6).dropna().tolist()
            )
    except Exception as e:
        print(f"获取沪市股票列表失败: {e}")

    try:
        df_sz = ak.stock_info_sz_name_code()
        if df_sz is not None and not df_sz.empty:
            if "A股代码" in df_sz.columns:
                code_col = "A股代码"
            elif "证券代码" in df_sz.columns:
                code_col = "证券代码"
            else:
                code_col = df_sz.columns[1]
            symbols.extend(
                df_sz[code_col].astype(str).str.strip().str.zfill(6).dropna().tolist()
            )
    except Exception as e:
        print(f"获取深市股票列表失败: {e}")

    return pd.Series(symbols).drop_duplicates().tolist()
    """
    try:
        df = ak.stock_zh_a_spot()
        if df is not None and not df.empty:
            if "代码" in df.columns:
                symbols = df["代码"].astype(str).str.zfill(6).tolist()
                print(f"成功获取全A股列表(新浪接口): {len(symbols)} 只")
                return symbols
            elif "symbol" in df.columns:
                # 新浪的 symbol 可能带前缀，如 sh600000
                symbols = df["symbol"].astype(str).str.replace(r"[a-zA-Z]+", "", regex=True).str.zfill(6).tolist()
                print(f"成功获取全A股列表(新浪接口): {len(symbols)} 只")
                return symbols
    except Exception as e:
        print("新浪接口获取失败")
        

def prepare_trend_features(df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    data = df.copy().sort_values("date").reset_index(drop=True)

    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data[f"ma{params.short_ma}"] = data["close"].rolling(params.short_ma).mean()
    data[f"ma{params.long_ma}"] = data["close"].rolling(params.long_ma).mean()

    data["ret_pct"] = data["close"].pct_change(params.lookback_return_days) * 100
    data["ret20_pct"] = data["close"].pct_change(20) * 100
    data["ret40_pct"] = data["close"].pct_change(40) * 100

    data["short_ma_slope_pct"] = data[f"ma{params.short_ma}"].pct_change(params.slope_days) * 100
    data["long_ma_slope_pct"] = data[f"ma{params.long_ma}"].pct_change(params.slope_days) * 100

    data["prev_breakout_high"] = data["high"].shift(1).rolling(params.breakout_lookback).max()
    data["price_to_short_ma_pct"] = (data["close"] / data[f"ma{params.short_ma}"] - 1) * 100
    data["price_to_ma10_pct"] = (data["close"] / data["ma10"] - 1) * 100

    data["rolling_20_high"] = data["high"].rolling(20).max()
    data["rolling_20_low"] = data["low"].rolling(20).min()
    data["higher_high"] = data["rolling_20_high"] > data["rolling_20_high"].shift(10)
    data["higher_low"] = data["rolling_20_low"] > data["rolling_20_low"].shift(10)

    conditions = {
        "close_gt_ma20": data["close"] > data[f"ma{params.short_ma}"],
        "ma20_up": data["short_ma_slope_pct"] > 0,
        "close_gt_ma60": data["close"] > data[f"ma{params.long_ma}"],
        "ma20_gt_ma60": data[f"ma{params.short_ma}"] > data[f"ma{params.long_ma}"],
        "ret20_strong": data["ret20_pct"] > max(6.0, params.min_return_pct * 0.45),
        "ret40_strong": data["ret40_pct"] > max(10.0, params.min_return_pct * 0.75),
        "near_recent_high": data["close"] >= data["rolling_20_high"] * 0.97,
    }

    score_df = pd.DataFrame({k: v.fillna(False).astype(int) for k, v in conditions.items()})
    data["trend_score"] = score_df.sum(axis=1)
    data["raw_uptrend"] = (data["trend_score"] >= params.score_threshold).fillna(False)
    data["is_uptrend"] = (
        data["raw_uptrend"].rolling(params.smooth_window, min_periods=1).sum() >= params.smooth_min_count
    ).fillna(False)

    if params.use_higher_low:
        data["is_uptrend"] = data["is_uptrend"] & data["higher_low"].fillna(False)

    return data


def _find_preliminary_segments(mask: pd.Series, min_segment_days: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_idx = None
    for i, is_up in enumerate(mask.fillna(False)):
        if is_up and start_idx is None:
            start_idx = i
        elif (not is_up) and start_idx is not None:
            end_idx = i - 1
            if end_idx - start_idx + 1 >= min_segment_days:
                segments.append((start_idx, end_idx))
            start_idx = None
    if start_idx is not None:
        end_idx = len(mask) - 1
        if end_idx - start_idx + 1 >= min_segment_days:
            segments.append((start_idx, end_idx))
    return segments


def _split_segment_by_drawdown(data: pd.DataFrame, start_idx: int, end_idx: int, params: TrendParams) -> list[tuple[int, int]]:
    if start_idx >= end_idx:
        return [(start_idx, end_idx)]

    result: list[tuple[int, int]] = []
    current_start = start_idx
    peak_idx = start_idx
    peak_close = float(data.loc[start_idx, "close"])
    below_ma_count = 0

    for i in range(start_idx + 1, end_idx + 1):
        close = float(data.loc[i, "close"])
        ma10 = float(data.loc[i, "ma10"]) if pd.notna(data.loc[i, "ma10"]) else close

        if close >= peak_close:
            peak_close = close
            peak_idx = i
            below_ma_count = 0
            continue

        drawdown_pct = (close / peak_close - 1) * 100
        below_ma = close < ma10 * (1 - params.hard_break_below_ma_pct / 100)
        below_ma_count = below_ma_count + 1 if below_ma else 0

        if drawdown_pct <= -params.end_drawdown_pct or below_ma_count >= params.hard_break_confirm_days:
            if peak_idx - current_start + 1 >= params.min_segment_days:
                result.append((current_start, peak_idx))
            current_start = i
            peak_idx = i
            peak_close = close
            below_ma_count = 0

    if end_idx - current_start + 1 >= params.min_segment_days:
        result.append((current_start, end_idx))

    return result


def _find_pivot_low(data: pd.DataFrame, left: int, right: int, pivot_window: int) -> int:
    left = max(0, left)
    right = min(len(data) - 1, right)
    if left >= right:
        return left

    candidate_indices: list[int] = []
    for i in range(left, right + 1):
        w_left = max(left, i - pivot_window)
        w_right = min(right, i + pivot_window)
        low = float(data.loc[i, "low"])
        if low <= float(data.loc[w_left:w_right, "low"].min()):
            candidate_indices.append(i)

    if candidate_indices:
        best_idx = min(candidate_indices, key=lambda idx: (float(data.loc[idx, "low"]), idx))
        return int(best_idx)
    return int(data.loc[left:right, "low"].idxmin())


def _find_pivot_high(data: pd.DataFrame, left: int, right: int, pivot_window: int) -> int:
    left = max(0, left)
    right = min(len(data) - 1, right)
    if left >= right:
        return left

    candidate_indices: list[int] = []
    for i in range(left, right + 1):
        w_left = max(left, i - pivot_window)
        w_right = min(right, i + pivot_window)
        high = float(data.loc[i, "high"])
        if high >= float(data.loc[w_left:w_right, "high"].max()):
            candidate_indices.append(i)

    if not candidate_indices:
        return int(data.loc[left:right, "high"].idxmax())

    max_high = max(float(data.loc[idx, "high"]) for idx in candidate_indices)

    def has_reversal(idx: int) -> bool:
        future_right = min(len(data) - 1, idx + pivot_window)
        future = data.loc[idx + 1:future_right]
        if future.empty:
            return False
        peak_high = float(data.loc[idx, "high"])
        peak_close = float(data.loc[idx, "close"])
        min_future_low = float(future["low"].min())
        min_future_close = float(future["close"].min())
        dd_low = (min_future_low / peak_high - 1) * 100 if peak_high else 0
        dd_close = (min_future_close / peak_close - 1) * 100 if peak_close else 0
        down_days = int((future["close"].diff() < 0).sum())
        return dd_low <= -5 or dd_close <= -4 or down_days >= 2

    strong_candidates = [
        idx for idx in candidate_indices
        if float(data.loc[idx, "high"]) >= max_high * 0.985 and has_reversal(idx)
    ]
    if strong_candidates:
        return int(min(strong_candidates))

    near_top_candidates = [idx for idx in candidate_indices if float(data.loc[idx, "high"]) >= max_high * 0.985]
    if near_top_candidates:
        return int(min(near_top_candidates))

    return int(max(candidate_indices, key=lambda idx: float(data.loc[idx, "high"])))


def _refine_segment_with_pivots(data: pd.DataFrame, start_idx: int, end_idx: int, params: TrendParams) -> tuple[int, int]:
    start_left = max(0, start_idx - params.pivot_start_expand_days)
    start_right = min(len(data) - 1, start_idx + params.pivot_window)
    refined_start = _find_pivot_low(data, start_left, start_right, params.pivot_window)

    end_left = max(refined_start, end_idx - params.pivot_window)
    end_right = min(len(data) - 1, end_idx + params.pivot_end_expand_days)
    refined_end = _find_pivot_high(data, end_left, end_right, params.pivot_window)

    if refined_end <= refined_start:
        refined_end = max(refined_start, end_idx)
    return refined_start, refined_end


def identify_uptrend_segments(df: pd.DataFrame, params: TrendParams | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    params = params or TrendParams()
    data = prepare_trend_features(df, params)

    segments: list[dict[str, Any]] = []
    preliminary_segments = _find_preliminary_segments(data["is_uptrend"], params.min_segment_days)

    for raw_start, raw_end in preliminary_segments:
        split_segments = _split_segment_by_drawdown(data, raw_start, raw_end, params)
        for split_start, split_end in split_segments:
            refined_start, refined_end = _refine_segment_with_pivots(data, split_start, split_end, params)
            if refined_end - refined_start + 1 >= params.min_segment_days:
                seg = _build_segment(data, refined_start, refined_end, raw_start, raw_end)
                if seg["period_return_pct"] >= params.min_segment_return_pct:
                    segments.append(seg)

    # 清理重叠段：默认保留先出现且更完整的波段，避免一个峰顶附近生成多段重叠结果
    cleaned: list[dict[str, Any]] = []
    for seg in sorted(segments, key=lambda x: (x["start_idx"], x["end_idx"])):
        if not cleaned:
            cleaned.append(seg)
            continue
        last = cleaned[-1]
        if seg["start_idx"] <= last["end_idx"]:
            continue
        cleaned.append(seg)
    segments = cleaned

    data["uptrend_start"] = False
    data["uptrend_end"] = False
    for seg in segments:
        data.loc[seg["start_idx"], "uptrend_start"] = True
        data.loc[seg["end_idx"], "uptrend_end"] = True

    latest_is_uptrend = False
    if not data.empty and segments:
        latest_idx = len(data) - 1
        latest_is_uptrend = any(seg["start_idx"] <= latest_idx <= seg["end_idx"] for seg in segments)
    data.attrs["latest_is_uptrend"] = latest_is_uptrend

    return data, segments


def _build_segment(
    data: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    raw_start_idx: int | None = None,
    raw_end_idx: int | None = None,
) -> dict[str, Any]:
    start_row = data.loc[start_idx]
    end_row = data.loc[end_idx]
    period_return = (end_row["close"] / start_row["close"] - 1) * 100
    return {
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "raw_start_idx": int(raw_start_idx) if raw_start_idx is not None else int(start_idx),
        "raw_end_idx": int(raw_end_idx) if raw_end_idx is not None else int(end_idx),
        "start_date": pd.Timestamp(start_row["date"]).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end_row["date"]).strftime("%Y-%m-%d"),
        "start_close": round(float(start_row["close"]), 3),
        "end_close": round(float(end_row["close"]), 3),
        "days": int(end_idx - start_idx + 1),
        "period_return_pct": round(float(period_return), 2),
        "peak_close": round(float(data.loc[start_idx:end_idx, "close"].max()), 3),
        "start_low": round(float(start_row["low"]), 3),
        "end_high": round(float(end_row["high"]), 3),
    }


def analyze_single_stock(symbol: str, start_date: str, end_date: str, params: TrendParams | None = None, output_dir: str = "output/trend_plots", plot: bool = True) -> dict[str, Any]:
    params = params or TrendParams()
    raw_df = get_stock_data(symbol, start_date, end_date)
    data, segments = identify_uptrend_segments(raw_df, params)

    latest_is_uptrend = bool(data.attrs.get("latest_is_uptrend", False))
    latest_segment = None
    if segments and latest_is_uptrend:
        latest_idx = len(data) - 1
        for seg in reversed(segments):
            if seg["start_idx"] <= latest_idx <= seg["end_idx"]:
                latest_segment = seg
                break

    plot_path = None
    if plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{symbol}_{start_date}_{end_date}_trend.png")
        plot_kline_with_trend(data, symbol, start_date, end_date, plot_path)

    return {
        "symbol": symbol,
        "latest_is_uptrend": latest_is_uptrend,
        "latest_segment": latest_segment,
        "segment_count": len(segments),
        "segments": segments,
        "plot_path": plot_path,
        "data": data,
    }


def screen_market_uptrend(start_date: str, end_date: str, params: TrendParams | None = None, limit: int | None = None, delay: float = 0.2, latest_only: bool = True, output_csv: str = "output/uptrend_screen_result.csv") -> pd.DataFrame:
    params = params or TrendParams()
    symbols = get_all_a_stock_symbols()
    if limit:
        symbols = symbols[:limit]

    rows: list[dict[str, Any]] = []
    total = len(symbols)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    for idx, symbol in enumerate(symbols, start=1):
        try:
            print(f"[{idx}/{total}] 分析 {symbol}")
            raw_df = get_stock_data(symbol, start_date, end_date)
            if raw_df is None or raw_df.empty or len(raw_df) < params.long_ma + 30:
                raise ValueError("历史数据不足")

            data, segments = identify_uptrend_segments(raw_df, params)
            latest_is_uptrend = bool(data.attrs.get("latest_is_uptrend", False))

            if latest_only and not latest_is_uptrend:
                time.sleep(delay)
                continue
            if not segments:
                time.sleep(delay)
                continue

            latest_segment = segments[-1]
            rows.append({
                "symbol": symbol,
                "latest_is_uptrend": latest_is_uptrend,
                "latest_segment_start": latest_segment["start_date"],
                "latest_segment_end": latest_segment["end_date"],
                "latest_segment_days": latest_segment["days"],
                "latest_segment_return_pct": latest_segment["period_return_pct"],
                "close": round(float(data["close"].iloc[-1]), 3),
                f"ma{params.short_ma}": round(float(data[f"ma{params.short_ma}"].iloc[-1]), 3),
                f"ma{params.long_ma}": round(float(data[f"ma{params.long_ma}"].iloc[-1]), 3),
                "trend_score": int(data["trend_score"].iloc[-1]),
            })
        except Exception as e:
            print(f"  -> 跳过 {symbol}: {e}")
        finally:
            time.sleep(delay)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["latest_is_uptrend", "latest_segment_return_pct", "latest_segment_days"], ascending=[False, False, False]).reset_index(drop=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result_df


def plot_kline_with_trend(data: pd.DataFrame, symbol: str, start_date: str, end_date: str, output_path: str, mav: tuple[int, int, int] = (5, 20, 60)) -> None:
    plot_df = data.copy().rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df = plot_df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

    original_index = data.set_index(pd.to_datetime(data["date"]))
    start_marker = pd.Series(index=plot_df.index, dtype=float)
    end_marker = pd.Series(index=plot_df.index, dtype=float)

    start_dates = data.loc[data["uptrend_start"], "date"]
    end_dates = data.loc[data["uptrend_end"], "date"]

    for dt in start_dates:
        start_marker.loc[pd.Timestamp(dt)] = original_index.loc[pd.Timestamp(dt), "low"] * 0.98
    for dt in end_dates:
        end_marker.loc[pd.Timestamp(dt)] = original_index.loc[pd.Timestamp(dt), "high"] * 1.02

    addplots = []
    if start_marker.notna().any():
        addplots.append(mpf.make_addplot(start_marker, type="scatter", markersize=100, marker="^", color="g"))
    if end_marker.notna().any():
        addplots.append(mpf.make_addplot(end_marker, type="scatter", markersize=100, marker="v", color="r"))

    events = pd.concat([start_dates, end_dates]).sort_values() if (not start_dates.empty or not end_dates.empty) else pd.Series(dtype='datetime64[ns]')
    vlines = dict(vlines=[pd.Timestamp(d) for d in events.tolist()], linewidths=0.7, colors=["green" if d in set(start_dates.tolist()) else "red" for d in events.tolist()], alpha=0.45) if not events.empty else None

    kwargs = dict(type="candle", volume=True, mav=mav, style="charles", title=f"{symbol} Trend Stage {start_date} ~ {end_date}", ylabel="Price", ylabel_lower="Volume", savefig=output_path, figsize=(16, 9), warn_too_much_data=10000)
    if addplots:
        kwargs["addplot"] = addplots
    if vlines:
        kwargs["vlines"] = vlines
    mpf.plot(plot_df, **kwargs)


def print_single_result(result: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"股票: {result['symbol']}")
    print(f"当前是否处于上升阶段: {'是' if result['latest_is_uptrend'] else '否'}")
    print(f"识别到的上升阶段数量: {result['segment_count']}")
    print("-" * 80)
    if result["segments"]:
        for idx, seg in enumerate(result["segments"], start=1):
            print(f"阶段{idx}: {seg['start_date']} -> {seg['end_date']} | 持续{seg['days']}天 | 区间涨幅{seg['period_return_pct']}%")
    else:
        print("未识别到满足条件的上升阶段")
    if result["plot_path"]:
        print(f"K线图已保存: {result['plot_path']}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="股票上升阶段识别与筛选")
    parser.add_argument("--mode", choices=["single", "market"], default="single")
    parser.add_argument("--symbol", type=str, default="600584")
    parser.add_argument("--start", type=str, default="20240101")
    parser.add_argument("--end", type=str, default="20260701")
    parser.add_argument("--limit", type=int, default=0, help="market 模式下限制股票数量，0 表示全部")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--lookback-return-days", type=int, default=60)
    parser.add_argument("--min-return-pct", type=float, default=15.0)
    parser.add_argument("--short-ma", type=int, default=20)
    parser.add_argument("--long-ma", type=int, default=60)
    parser.add_argument("--slope-days", type=int, default=5)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--max-deviation-pct", type=float, default=35.0)
    parser.add_argument("--min-segment-days", type=int, default=8)
    parser.add_argument("--score-threshold", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--smooth-min-count", type=int, default=4)
    parser.add_argument("--start-lookback-days", type=int, default=15)
    parser.add_argument("--end-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--hard-break-below-ma-pct", type=float, default=3.0)
    parser.add_argument("--hard-break-confirm-days", type=int, default=2)
    parser.add_argument("--pivot-window", type=int, default=5)
    parser.add_argument("--pivot-start-expand-days", type=int, default=20)
    parser.add_argument("--pivot-end-expand-days", type=int, default=8)
    parser.add_argument("--min-segment-return-pct", type=float, default=10)
    parser.add_argument("--use-higher-low", action="store_true")
    parser.add_argument("--latest-only", action="store_true", help="market 模式仅保留当前仍处于上升阶段的股票")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = TrendParams(
        lookback_return_days=args.lookback_return_days,
        min_return_pct=args.min_return_pct,
        short_ma=args.short_ma,
        long_ma=args.long_ma,
        slope_days=args.slope_days,
        breakout_lookback=args.breakout_lookback,
        max_deviation_pct=args.max_deviation_pct,
        min_segment_days=args.min_segment_days,
        use_higher_low=args.use_higher_low,
        score_threshold=args.score_threshold,
        smooth_window=args.smooth_window,
        smooth_min_count=args.smooth_min_count,
        start_lookback_days=args.start_lookback_days,
        end_drawdown_pct=args.end_drawdown_pct,
        hard_break_below_ma_pct=args.hard_break_below_ma_pct,
        hard_break_confirm_days=args.hard_break_confirm_days,
        pivot_window=args.pivot_window,
        pivot_start_expand_days=args.pivot_start_expand_days,
        pivot_end_expand_days=args.pivot_end_expand_days,
        min_segment_return_pct=args.min_segment_return_pct,
    )

    if args.mode == "single":
        result = analyze_single_stock(symbol=args.symbol, start_date=args.start, end_date=args.end, params=params, plot=not args.no_plot)
        print_single_result(result)
    else:
        result_df = screen_market_uptrend(start_date=args.start, end_date=args.end, params=params, limit=(args.limit or None), delay=args.delay, latest_only=args.latest_only)
        print(result_df.head(20).to_string(index=False) if not result_df.empty else "无结果")
        print("结果已保存到 output/uptrend_screen_result.csv")


if __name__ == "__main__":
    main()
