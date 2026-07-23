import signal
import akshare as ak
import pandas as pd


class DataFetchTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise DataFetchTimeout("数据获取超时")


def fetch_with_timeout(func, timeout_seconds=30, *args, **kwargs):
    """给任意函数加超时"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


def add_market_prefix(symbol: str) -> str:
    symbol = str(symbol).replace("sh", "").replace("sz", "").replace("bj", "").strip()

    if symbol.startswith(("600", "601", "603", "605", "688")):
        return "sh" + symbol
    elif symbol.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz" + symbol
    elif symbol.startswith(("430", "440", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "920")):
        return "bj" + symbol
    else:
        raise ValueError(f"无法识别交易所前缀: {symbol}")


def get_stock_data(symbol: str, start_date: str, end_date: str, timeout_seconds: int = 30) -> pd.DataFrame:
    """
    获取 A 股前复权日线数据（带超时保护）
    """
    symbol = add_market_prefix(symbol)
    
    try:
        df = fetch_with_timeout(
            ak.stock_zh_a_daily,
            timeout_seconds=timeout_seconds,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except DataFetchTimeout:
        print(f"  -> 数据获取超时({timeout_seconds}s)，跳过: {symbol}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  -> 数据获取失败: {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    #print(df.head())

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    })

    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [c for c in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "change", "turnover"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


#print(get_stock_data("600584", "20260710", "20260710").head())
      
