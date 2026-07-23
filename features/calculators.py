import pandas as pd

from features.registry import FEATURE_REGISTRY
import numpy as np

def find_candidate_bottoms(df: pd.DataFrame, pivot_window: int = 5) -> list[int]:
    candidates: list[int] = []
    if df is None or df.empty:
        return candidates

    for i in range(pivot_window, len(df) - pivot_window):
        low = float(df.loc[i, "low"])
        left = df.loc[i - pivot_window:i - 1, "low"]
        right = df.loc[i + 1:i + pivot_window, "low"]
        if low <= float(left.min()) and low <= float(right.min()):
            candidates.append(i)
    return candidates


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_ma_features(data: pd.DataFrame) -> pd.DataFrame:
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()
    return data


def calc_return_features(data: pd.DataFrame) -> pd.DataFrame:
    for n in [3, 5, 10, 20, 40, 60]:
        data[f"ret_{n}"] = data["close"].pct_change(n) * 100
    return data


def calc_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    for n in [3, 5, 10, 20, 40, 60]:
        data[f"vol_mean_{n}"] = data["volume"].rolling(n).mean()
    data["vol_ratio_5"] = data["volume"] / data["vol_mean_5"]
    data["vol_ratio_20"] = data["volume"] / data["vol_mean_20"]
    return data


def calc_price_position_features(data: pd.DataFrame) -> pd.DataFrame:
    for n in [10, 20, 60]:
        data[f"price_pos_to_low_{n}"] = (data["close"] / data["low"].rolling(n).min() - 1) * 100
        data[f"price_pos_to_high_{n}"] = (data["close"] / data["high"].rolling(n).max() - 1) * 100
    return data


def calc_amplitude_features(data: pd.DataFrame) -> pd.DataFrame:
    for n in [5, 10, 20]:
        data[f"amplitude_mean_{n}"] = ((data["high"] / data["low"] - 1) * 100).rolling(n).mean()
    return data


def calc_atr_feature(data: pd.DataFrame) -> pd.DataFrame:
    data["atr_14"] = calc_atr(data, 14)
    return data


def calc_candle_features(data: pd.DataFrame) -> pd.DataFrame:
    body = (data["close"] - data["open"]).abs()
    candle_range = (data["high"] - data["low"]).replace(0, pd.NA)
    upper = data["high"] - data[["open", "close"]].max(axis=1)
    lower = data[["open", "close"]].min(axis=1) - data["low"]

    data["body_pct"] = (body / candle_range) * 100
    data["upper_shadow_pct"] = (upper / candle_range) * 100
    data["lower_shadow_pct"] = (lower / candle_range) * 100
    data["is_red"] = (data["close"] > data["open"]).astype(int)
    return data


def calc_short_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    data["down_days_5"] = (data["close"].diff() < 0).rolling(5).sum()
    data["up_days_5"] = (data["close"].diff() > 0).rolling(5).sum()
    return data

def calc_pivot_confirm_features(data: pd.DataFrame) -> pd.DataFrame:
    prev_high_3 = data["high"].shift(1).rolling(3).max()
    data["breakout_strength"] = (data["close"] / prev_high_3 - 1) * 100

    data["vol_break_ratio"] = data["volume"] / data["volume"].shift(1).rolling(5).mean()

    vol_mean_5 = data["volume"].rolling(5).mean()
    vol_mean_20 = data["volume"].rolling(20).mean()
    data["vol_contraction_then_expand"] = (vol_mean_5 / vol_mean_20 - 1) * 100

    data["prev_drop_5"] = data["close"].pct_change(5) * 100

    # 连阳天数
    streak = 0
    streaks = []
    for i in range(len(data)):
        if data.loc[i, "close"] > data.loc[i, "open"]:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    data["consecutive_red"] = streaks

    return data



def calc_ma_relation_features(data: pd.DataFrame) -> pd.DataFrame:
    if "ma5" not in data.columns or "ma10" not in data.columns or "ma20" not in data.columns or "ma60" not in data.columns:
        data = calc_ma_features(data)

    data["close_below_ma20_pct"] = (data["close"] / data["ma20"] - 1) * 100
    data["close_below_ma60_pct"] = (data["close"] / data["ma60"] - 1) * 100
    data["ma5_slope_3"] = data["ma5"].pct_change(3) * 100
    data["ma10_slope_3"] = data["ma10"].pct_change(3) * 100
    data["ma20_slope_5"] = data["ma20"].pct_change(5) * 100
    data["ma60_slope_5"] = data["ma60"].pct_change(5) * 100

    data["ma5_gt_ma10"] = (data["ma5"] > data["ma10"]).astype(int)
    data["ma10_gt_ma20"] = (data["ma10"] > data["ma20"]).astype(int)
    data["ma20_gt_ma60"] = (data["ma20"] > data["ma60"]).astype(int)
    return data


def calc_local_low_features(data: pd.DataFrame) -> pd.DataFrame:
    data["is_local_low_5"] = 0
    data.loc[find_candidate_bottoms(data, 5), "is_local_low_5"] = 1

    data["is_local_low_3"] = 0
    data.loc[find_candidate_bottoms(data, 3), "is_local_low_3"] = 1
    return data


def enrich_features(data: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    result = data.copy().sort_values("date").reset_index(drop=True)

    if feature_names is None:
        # 全量兼容模式
        funcs = list(dict.fromkeys(FEATURE_REGISTRY.values()))
    else:
        func_names = []
        for feature in feature_names:
            func_name = FEATURE_REGISTRY.get(feature)
            if func_name:
                func_names.append(func_name)
        funcs = list(dict.fromkeys(func_names))

    for func_name in funcs:
        result = CALC_FUNC_MAP[func_name](result)
    
    result = result.replace([np.inf, -np.inf], np.nan)

    return result

def calc_turnover_features(data: pd.DataFrame) -> pd.DataFrame:
    """换手率相关特征"""
    if "turnover" not in data.columns:
        data["turnover"] = 0.0

    to = data["turnover"]

    # 1. 当日换手率（原值）
    data["turnover_today"] = to

    # 2. 换手率倍数（今日 / 前5日均量）
    data["turnover_ratio_5"] = to / to.shift(1).rolling(5).mean()

    # 3. 换手率倍数（今日 / 前20日均量）
    data["turnover_ratio_20"] = to / to.shift(1).rolling(20).mean()

    # 4. 换手率收缩度（近5日均 / 近20日均）
    data["turnover_contraction"] = to.rolling(5).mean() / to.rolling(20).mean()

    # 5. 换手率分位数（今日换手率在近60日中的分位）
    data["turnover_quantile_60"] = to.rolling(60).rank(pct=True)

    # 6. 换手率跳升（今日换手率 / 近20日最低换手率）
    data["turnover_jump"] = to / to.rolling(20).min()

    return data

def calc_limit_up_features(data: pd.DataFrame) -> pd.DataFrame:
    """涨停相关特征（动态计算，无未来函数）"""
    # 计算涨停价（前一日收盘价 * 1.1，简化处理，忽略ST和科创板20%）
    prev_close = data["close"].shift(1)
    
    # 涨幅
    pct_change = (data["close"] / prev_close - 1) * 100
    
    # 是否涨停（涨幅大于9.8%）
    is_limit = pct_change >= 9.8
    
    # 1. 当日是否涨停
    data["is_limit_up"] = is_limit.astype(int)
    
    # 2. 近20日涨停次数
    data["limit_up_count_20"] = is_limit.rolling(20).sum()
    
    # 3. 连板数
    streak = 0
    streaks = []
    for i in range(len(data)):
        if is_limit.iloc[i]:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    data["consecutive_limit_up"] = streaks
    
    return data

def calc_local_low_features(data: pd.DataFrame) -> pd.DataFrame:
    for pw in [3, 5, 10, 20]:
        col = f"is_local_low_{pw}"
        data[col] = 0
        data.loc[find_candidate_bottoms(data, pw), col] = 1
    return data


CALC_FUNC_MAP = {
    "calc_ma_features": calc_ma_features,
    "calc_return_features": calc_return_features,
    "calc_volume_features": calc_volume_features,
    "calc_price_position_features": calc_price_position_features,
    "calc_amplitude_features": calc_amplitude_features,
    "calc_atr_feature": calc_atr_feature,
    "calc_candle_features": calc_candle_features,
    "calc_short_trend_features": calc_short_trend_features,
    "calc_ma_relation_features": calc_ma_relation_features,
    "calc_local_low_features": calc_local_low_features,
    "calc_pivot_confirm_features": calc_pivot_confirm_features,
    "calc_turnover_features": calc_turnover_features,  
    "calc_limit_up_features": calc_limit_up_features,
}
