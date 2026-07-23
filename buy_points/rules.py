import pandas as pd


def bottom_reversal_v1(df: pd.DataFrame) -> pd.Series:
    mask = (
        (df["is_local_low_5"] == 1)
        | ((df["is_local_low_3"] == 1) & (df["lower_shadow_pct"] >= 20))
        | ((df["trend_score"] >= 2) & (df["close_below_ma20_pct"] <= 2) & (df["ret_10"] <= 3))
    )
    return mask.fillna(False)


def pivot_confirm_v1(df: pd.DataFrame) -> pd.Series:
    prev_high_3 = df["high"].shift(1).rolling(3).max()
    mask = (
        (
            (df["is_local_low_5"] == 1)
            | (df["is_local_low_3"] == 1)
        )
        & (df["close"] >= prev_high_3.fillna(df["close"]))
        & (df["is_red"] == 1)
    ) | (
        (df["trend_score"] >= 2)
        & (df["close_below_ma20_pct"] <= 2)
        & (df["ma5_slope_3"] > 0)
        & (df["is_red"] == 1)
    )
    return mask.fillna(False)


def v_shape_candidate_v1(df: pd.DataFrame) -> pd.Series:
    """V形低点候选规则（只看过去和现在，不看未来）"""
    drop_from_high_20 = (df["close"] / df["high"].rolling(20).max() - 1) * 100
    down_days_20 = (df["close"].diff() < 0).rolling(20).sum()
    below_ma20 = df["close_below_ma20_pct"] < -8
    no_rebound_20 = df["is_red"].rolling(20).apply(
        lambda x: max(sum(1 for i in range(len(x)-1) if x[i] == 1 and x[i+1] == 1), 0), raw=True
    ) == 0

    weak_count = (
        (drop_from_high_20 < -12).astype(int)
        + (down_days_20 >= 10).astype(int)
        + below_ma20.astype(int)
        + no_rebound_20.astype(int)
    )
    is_crash = (drop_from_high_20 < -12) & (weak_count >= 2)

    is_local_low = (df["is_local_low_10"] == 1) | (df["is_local_low_20"] == 1)
    long_lower = df["lower_shadow_pct"] >= 15
    day_pct_change = (df["close"] / df["close"].shift(1) - 1) * 100
    narrow_drop_or_red = (day_pct_change > -2) | (df["is_red"] == 1)

    vol_break_ratio_20 = df["volume"] / df["volume"].shift(1).rolling(20).mean()
    vol_spike = vol_break_ratio_20 > 1.1

    sharp_count = (
        is_local_low.astype(int)
        + long_lower.astype(int)
        + narrow_drop_or_red.astype(int)
        + vol_spike.astype(int)
    )
    is_sharp = is_local_low & (sharp_count >= 2)

    mask = is_crash & is_sharp
    return mask.fillna(False)


def platform_breakout_v1(df: pd.DataFrame) -> pd.Series:
    """平台突破"""
    prev_high_20 = df["high"].shift(1).rolling(20).max()
    prev_low_20 = df["low"].shift(1).rolling(20).min()
    platform_range_pct = (prev_high_20 / prev_low_20 - 1) * 100

    mask = (
        (platform_range_pct <= 18)
        & (df["close"] >= prev_high_20.fillna(df["close"]) * 0.995)
        & (df["breakout_strength"] >= 1.0)
        & (df["vol_ratio_5"] >= 1.2)
        & (df["is_red"] == 1)
        & (df["ma20_slope_5"] >= -1)
    )
    return mask.fillna(False)


def pullback_confirm_v1(df: pd.DataFrame) -> pd.Series:
    """回踩确认"""
    prev_high_20 = df["high"].shift(5).rolling(20).max()
    recent_strong = (df["price_pos_to_low_20"] >= 12) | (df["ret_20"] >= 8)
    near_ma20 = df["close_below_ma20_pct"].between(-3, 3)
    stop_drop = ((df["low"] / df["low"].shift(1).rolling(5).min()) - 1) * 100 >= -1.5

    mask = (
        recent_strong
        & near_ma20
        & stop_drop
        & (df["ma20_slope_5"] > 0)
        & ((df["is_red"] == 1) | (df["lower_shadow_pct"] >= 20))
        & (df["close"] >= prev_high_20.fillna(df["close"]) * 0.90)
    )
    return mask.fillna(False)


def trend_continuation_v1(df: pd.DataFrame) -> pd.Series:
    """趋势中继"""
    pullback_ok = df["close_below_ma20_pct"].between(-8, 2)
    trend_ok = (
        (df["ma20_slope_5"] > 0)
        & (df["ma60_slope_5"] >= 0)
        & (df["ma20_gt_ma60"] == 1)
    )
    restart_signal = (
        ((df["is_red"] == 1) & (df["ma5_slope_3"] > 0))
        | (df["breakout_strength"] > 0.5)
    )

    mask = (
        trend_ok
        & pullback_ok
        & (df["ret_60"] >= 10)
        & (df["price_pos_to_high_20"] >= -10)
        & restart_signal
    )
    return mask.fillna(False)


def strong_pullback_rebound_v1(df: pd.DataFrame) -> pd.Series:
    """强势股首次大回撤企稳"""
    was_strong = (df["ret_60"] >= 25) | (df["price_pos_to_low_60"] >= 30)
    deep_pullback = df["price_pos_to_high_20"].between(-20, -8)
    stop_fall = (
        (df["is_local_low_5"] == 1)
        | (df["lower_shadow_pct"] >= 18)
        | (df["is_red"] == 1)
    )

    mask = (
        was_strong
        & deep_pullback
        & stop_fall
        & (df["close_below_ma20_pct"] >= -10)
        & (df["turnover_ratio_5"] >= 0.8)
    )
    return mask.fillna(False)


def volume_turning_point_v1(df: pd.DataFrame) -> pd.Series:
    """放量拐点启动"""
    low_zone = df["price_pos_to_high_20"] <= -5
    structure_ok = (
        (df["is_local_low_3"] == 1)
        | (df["is_local_low_5"] == 1)
        | (df["ma5_slope_3"] > 0)
    )
    volume_ok = (df["vol_ratio_5"] >= 1.5) | (df["turnover_ratio_5"] >= 1.3)
    candle_ok = (df["is_red"] == 1) & (df["body_pct"] >= 35)

    mask = (
        low_zone
        & structure_ok
        & volume_ok
        & candle_ok
        & (df["breakout_strength"] >= 0)
    )
    return mask.fillna(False)


def build_candidate_mask_by_rule(df: pd.DataFrame, rule_name: str) -> pd.Series:
    if rule_name not in BUY_POINT_RULES:
        raise ValueError(f"未注册的买点规则: {rule_name}")
    return BUY_POINT_RULES[rule_name](df)


BUY_POINT_RULES = {
    "bottom_reversal_v1": bottom_reversal_v1,
    "pivot_confirm_v1": pivot_confirm_v1,
    "v_shape_candidate_v1": v_shape_candidate_v1,
    "platform_breakout_v1": platform_breakout_v1,
    "pullback_confirm_v1": pullback_confirm_v1,
    "trend_continuation_v1": trend_continuation_v1,
    "strong_pullback_rebound_v1": strong_pullback_rebound_v1,
    "volume_turning_point_v1": volume_turning_point_v1,
}