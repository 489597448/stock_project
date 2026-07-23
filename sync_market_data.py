import argparse
import os
import pandas as pd

from data.local_data_store import sync_market_data, sync_one_symbol


def main() -> None:
    parser = argparse.ArgumentParser(description="同步本地股票原始数据与特征缓存")
    parser.add_argument("--mode", choices=["market", "single"], default="market")
    parser.add_argument("--symbol", type=str, default="600584")
    parser.add_argument("--start", type=str, default="20180101")
    parser.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--data-root", type=str, default="local_data")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--refresh-symbols", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    if args.mode == "single":
        result = sync_one_symbol(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            feature_config_path=args.feature_config,
            data_root=args.data_root,
            force_full=args.force_full,
        )
        print(result)
    else:
        df = sync_market_data(
            start_date=args.start,
            end_date=args.end,
            feature_config_path=args.feature_config,
            data_root=args.data_root,
            limit=args.limit,
            delay=args.delay,
            refresh_symbols=args.refresh_symbols,
            force_full=args.force_full,
        )
        print(f"同步完成，股票数: {len(df)}")
        if not df.empty:
            print(df.head(20).to_string(index=False))
        print(f"输出目录: {args.data_root}")


if __name__ == "__main__":
    main()
