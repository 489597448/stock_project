import os
import time
from pathlib import Path

import pandas as pd

from data.akshare_data import get_stock_data
from data.trend_stage import TrendParams, get_all_a_stock_symbols, prepare_trend_features
from features.calculators import enrich_features
from utils.config_loader import load_feature_config


DEFAULT_DATA_ROOT = Path("local_data")
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_FEATURE_DIR = DEFAULT_DATA_ROOT / "featured"
DEFAULT_SYMBOLS_FILE = DEFAULT_DATA_ROOT / "symbols.csv"
DEFAULT_META_FILE = DEFAULT_DATA_ROOT / "meta.csv"
DEFAULT_PARAMS = TrendParams()


RAW_COLUMNS = [
    "date", "open", "high", "low", "close", "volume",
    "amount", "amplitude", "pct_change", "change", "turnover",
]


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.zfill(6)


def ensure_dirs(data_root: str | Path = DEFAULT_DATA_ROOT) -> tuple[Path, Path]:
    root = Path(data_root)
    raw_dir = root / "raw"
    feature_dir = root / "featured"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, feature_dir


def raw_path(symbol: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    raw_dir, _ = ensure_dirs(data_root)
    return raw_dir / f"{normalize_symbol(symbol)}.csv"


def feature_path(symbol: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    _, feature_dir = ensure_dirs(data_root)
    return feature_dir / f"{normalize_symbol(symbol)}.csv"


def symbols_file(data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / "symbols.csv"


def meta_file(data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / "meta.csv"


def get_all_feature_columns(config_path: str = "configs/features.yaml") -> list[str]:
    config = load_feature_config(config_path)
    cols = config.get("all_features", [])
    if not cols:
        raise ValueError("features.yaml 中未配置 all_features")
    return cols


def load_raw_data(symbol: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    path = raw_path(symbol, data_root)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def load_feature_data(symbol: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    path = feature_path(symbol, data_root)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def save_raw_data(symbol: str, df: pd.DataFrame, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    path = raw_path(symbol, data_root)
    to_save = df.copy().sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    to_save.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_feature_data(symbol: str, df: pd.DataFrame, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    path = feature_path(symbol, data_root)
    to_save = df.copy().sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    to_save.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def merge_raw_data(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        result = new_df.copy()
    elif new_df is None or new_df.empty:
        result = old_df.copy()
    else:
        result = pd.concat([old_df, new_df], ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return result


def rebuild_feature_data(raw_df: pd.DataFrame, feature_config_path: str = "configs/features.yaml") -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    feature_cols = get_all_feature_columns(feature_config_path)
    trend_df = prepare_trend_features(raw_df.copy(), DEFAULT_PARAMS)
    data = enrich_features(trend_df, feature_cols)
    return data.sort_values("date").reset_index(drop=True)


def update_meta(symbol: str, raw_df: pd.DataFrame, feature_df: pd.DataFrame, data_root: str | Path = DEFAULT_DATA_ROOT) -> None:
    path = meta_file(data_root)
    symbol = normalize_symbol(symbol)
    row = {
        "symbol": symbol,
        "raw_rows": int(len(raw_df)) if raw_df is not None else 0,
        "feature_rows": int(len(feature_df)) if feature_df is not None else 0,
        "raw_latest_date": pd.Timestamp(raw_df["date"].max()).strftime("%Y-%m-%d") if raw_df is not None and not raw_df.empty else None,
        "feature_latest_date": pd.Timestamp(feature_df["date"].max()).strftime("%Y-%m-%d") if feature_df is not None and not feature_df.empty else None,
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if path.exists() and path.stat().st_size > 0:
        meta_df = pd.read_csv(path, dtype={"symbol": str})
        meta_df["symbol"] = meta_df["symbol"].astype(str).str.zfill(6)
        meta_df = meta_df[meta_df["symbol"] != symbol]
        meta_df = pd.concat([meta_df, pd.DataFrame([row])], ignore_index=True)
    else:
        meta_df = pd.DataFrame([row])
    meta_df = meta_df.sort_values("symbol").reset_index(drop=True)
    meta_df.to_csv(path, index=False, encoding="utf-8-sig")


def refresh_symbol_list(data_root: str | Path = DEFAULT_DATA_ROOT) -> list[str]:
    syms = [normalize_symbol(s) for s in get_all_a_stock_symbols()]
    df = pd.DataFrame({"symbol": sorted(pd.Series(syms).drop_duplicates().tolist())})
    df.to_csv(symbols_file(data_root), index=False, encoding="utf-8-sig")
    return df["symbol"].tolist()


def get_symbol_list(data_root: str | Path = DEFAULT_DATA_ROOT, refresh: bool = False) -> list[str]:
    path = symbols_file(data_root)
    if refresh or (not path.exists()) or path.stat().st_size == 0:
        return refresh_symbol_list(data_root)
    df = pd.read_csv(path, dtype={"symbol": str})
    return df["symbol"].astype(str).str.zfill(6).drop_duplicates().tolist()


def calc_fetch_start_date(existing_raw: pd.DataFrame, requested_start: str | None = None, overlap_days: int = 5) -> str:
    if existing_raw is None or existing_raw.empty:
        return requested_start or "20100101"
    last_date = pd.Timestamp(existing_raw["date"].max())
    fetch_start = last_date - pd.Timedelta(days=overlap_days)
    if requested_start:
        fetch_start = max(fetch_start, pd.Timestamp(requested_start))
    return fetch_start.strftime("%Y%m%d")


def sync_one_symbol(
    symbol: str,
    start_date: str | None,
    end_date: str,
    feature_config_path: str = "configs/features.yaml",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force_full: bool = False,
) -> dict:
    symbol = normalize_symbol(symbol)
    existing_raw = load_raw_data(symbol, data_root)
    fetch_start = start_date if force_full else calc_fetch_start_date(existing_raw, start_date)
    new_df = get_stock_data(symbol, fetch_start, end_date)
    merged_raw = merge_raw_data(pd.DataFrame(columns=RAW_COLUMNS) if force_full else existing_raw, new_df)
    feature_df = rebuild_feature_data(merged_raw, feature_config_path)
    save_raw_data(symbol, merged_raw, data_root)
    save_feature_data(symbol, feature_df, data_root)
    update_meta(symbol, merged_raw, feature_df, data_root)
    return {
        "symbol": symbol,
        "raw_rows": int(len(merged_raw)),
        "feature_rows": int(len(feature_df)),
        "latest_date": pd.Timestamp(feature_df["date"].max()).strftime("%Y-%m-%d") if not feature_df.empty else None,
        "fetched_from": fetch_start,
        "fetched_to": end_date,
    }


def sync_market_data(
    start_date: str,
    end_date: str,
    feature_config_path: str = "configs/features.yaml",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    limit: int = 0,
    delay: float = 0.2,
    refresh_symbols: bool = False,
    force_full: bool = False,
) -> pd.DataFrame:
    symbols = get_symbol_list(data_root, refresh=refresh_symbols)
    if limit and limit > 0:
        symbols = symbols[:limit]

    rows = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{total}] 同步 {symbol}", flush=True)
        try:
            row = sync_one_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                feature_config_path=feature_config_path,
                data_root=data_root,
                force_full=force_full,
            )
            rows.append(row)
        except Exception as e:
            print(f"  -> 跳过 {symbol}: {e}", flush=True)
        time.sleep(delay)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(Path(data_root) / "sync_result.csv", index=False, encoding="utf-8-sig")
    return result


def ensure_symbol_feature_data(
    symbol: str,
    end_date: str,
    start_date: str | None = None,
    feature_config_path: str = "configs/features.yaml",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    auto_update: bool = False,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    feature_df = load_feature_data(symbol, data_root)
    target_date = pd.Timestamp(end_date)
    if feature_df.empty:
        if not auto_update:
            return pd.DataFrame()
        sync_one_symbol(symbol, start_date, end_date, feature_config_path, data_root)
        feature_df = load_feature_data(symbol, data_root)
    if feature_df.empty:
        return pd.DataFrame()

    latest_date = pd.Timestamp(feature_df["date"].max())
    if latest_date < target_date and auto_update:
        sync_one_symbol(symbol, start_date, end_date, feature_config_path, data_root)
        feature_df = load_feature_data(symbol, data_root)
    return feature_df[feature_df["date"] <= target_date].copy().reset_index(drop=True)
