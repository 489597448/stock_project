import argparse
import os
import re
import smtplib
from datetime import datetime, timedelta
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from data.local_data_store import ensure_symbol_feature_data, normalize_symbol


TO_EMAIL = "489597448@qq.com"
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465
FROM_EMAIL = "m19821879218@163.com"
SMTP_AUTH_CODE = "UBgrW8SCQ9WVqm2E"


PRED_FILE_PATTERN = re.compile(r"buy_point_(.+?)_live_predictions\.csv$")


def get_last_sunday(base_date: pd.Timestamp | None = None) -> pd.Timestamp:
    base = pd.Timestamp(base_date or pd.Timestamp.today()).normalize()
    return base - pd.Timedelta(days=(base.weekday() + 1) % 7)


def get_previous_week_range(report_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    week_start = report_date - pd.Timedelta(days=6)
    prev_week_start = week_start - pd.Timedelta(days=7)
    prev_week_end = week_start - pd.Timedelta(days=1)
    return prev_week_start.normalize(), prev_week_end.normalize()


def iter_prediction_files(history_dir: Path, start_date: pd.Timestamp, end_date: pd.Timestamp):
    cur = start_date
    while cur <= end_date:
        day_dir = history_dir / cur.strftime("%Y%m%d")
        if day_dir.exists():
            for file in sorted(day_dir.glob("buy_point_*_live_predictions.csv")):
                yield cur, file
        cur += pd.Timedelta(days=1)


def load_prediction_history(history_dir: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    base = Path(history_dir)
    for pred_day, file_path in iter_prediction_files(base, start_date, end_date):
        m = PRED_FILE_PATTERN.search(file_path.name)
        buy_point_name = m.group(1) if m else "unknown"
        try:
            df = pd.read_csv(file_path, dtype={"symbol": str})
        except Exception:
            continue
        if df.empty:
            continue
        df["symbol"] = df["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).map(normalize_symbol)
        if "date" in df.columns:
            df["signal_date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["signal_date"] = pred_day
        df["file_date"] = pred_day
        df["buy_point"] = buy_point_name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=["signal_date", "symbol"]).copy()
    result["pred_score"] = pd.to_numeric(result.get("pred_score"), errors="coerce")
    result["close"] = pd.to_numeric(result.get("close"), errors="coerce")
    result = result.sort_values(["buy_point", "symbol", "signal_date", "pred_score"], ascending=[True, True, True, False])
    return result.reset_index(drop=True)


def calc_signal_max_return(row: pd.Series, report_end: pd.Timestamp, data_root: str, feature_config: str) -> dict:
    symbol = normalize_symbol(row["symbol"])
    signal_date = pd.Timestamp(row["signal_date"]).normalize()
    df = ensure_symbol_feature_data(
        symbol=symbol,
        start_date="20210101",
        end_date=report_end.strftime("%Y%m%d"),
        feature_config_path=feature_config,
        data_root=data_root,
        auto_update=False,
    )
    if df.empty:
        return {"eval_status": "no_feature_data", "max_return_pct": None, "eval_days": 0, "max_return_date": None}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    entry_rows = df[df["date"] == signal_date]
    if entry_rows.empty:
        return {"eval_status": "missing_signal_date", "max_return_pct": None, "eval_days": 0, "max_return_date": None}

    entry_close = pd.to_numeric(entry_rows.iloc[-1].get("close"), errors="coerce")
    if pd.isna(entry_close) or float(entry_close) <= 0:
        return {"eval_status": "bad_entry_close", "max_return_pct": None, "eval_days": 0, "max_return_date": None}

    future = df[(df["date"] >= signal_date) & (df["date"] <= report_end)].copy()
    if future.empty:
        return {"eval_status": "no_eval_window", "max_return_pct": None, "eval_days": 0, "max_return_date": None}

    future["high"] = pd.to_numeric(future["high"], errors="coerce")
    future = future.dropna(subset=["high"])
    if future.empty:
        return {"eval_status": "no_high_data", "max_return_pct": None, "eval_days": 0, "max_return_date": None}

    idx = future["high"].idxmax()
    max_high = float(future.loc[idx, "high"])
    max_return_pct = (max_high / float(entry_close) - 1) * 100
    max_return_date = pd.Timestamp(future.loc[idx, "date"]).strftime("%Y-%m-%d")
    return {
        "eval_status": "ok",
        "max_return_pct": round(max_return_pct, 4),
        "eval_days": int(len(future)),
        "max_return_date": max_return_date,
    }


def summarize_by_threshold(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for bp_name, g in df.groupby("buy_point"):
        for th in thresholds:
            picked = g[g["pred_score"] >= th].copy()
            rows.append({
                "buy_point": bp_name,
                "threshold": th,
                "signals": int(len(picked)),
                "avg_max_return_pct": round(float(picked["max_return_pct"].mean()), 4) if not picked.empty else None,
                "median_max_return_pct": round(float(picked["max_return_pct"].median()), 4) if not picked.empty else None,
                "win_rate_pct": round(float((picked["max_return_pct"] > 0).mean() * 100), 2) if not picked.empty else None,
                "gt_3_pct": round(float((picked["max_return_pct"] >= 3).mean() * 100), 2) if not picked.empty else None,
                "gt_5_pct": round(float((picked["max_return_pct"] >= 5).mean() * 100), 2) if not picked.empty else None,
                "best_max_return_pct": round(float(picked["max_return_pct"].max()), 4) if not picked.empty else None,
            })
    return pd.DataFrame(rows)


def format_mail_content(raw_df: pd.DataFrame, summary_df: pd.DataFrame, pred_start: pd.Timestamp, pred_end: pd.Timestamp, report_end: pd.Timestamp) -> str:
    lines = []
    lines.append(f"周评测区间：预测日期 {pred_start.strftime('%Y-%m-%d')} ~ {pred_end.strftime('%Y-%m-%d')}，收益统计截止 {report_end.strftime('%Y-%m-%d')}")
    lines.append("")

    if raw_df.empty:
        lines.append("没有找到可评测的历史预测文件。")
        return "\n".join(lines)

    lines.append("一、原始信号统计")
    lines.append("=" * 80)
    stat_df = raw_df.groupby("buy_point").agg(
        signals=("symbol", "count"),
        valid_eval=("eval_status", lambda x: int((x == "ok").sum())),
        avg_max_return_pct=("max_return_pct", "mean"),
        best_max_return_pct=("max_return_pct", "max"),
    ).reset_index()
    if not stat_df.empty:
        for _, row in stat_df.iterrows():
            lines.append(
                f"{row['buy_point']}: 信号{int(row['signals'])}个, 有效评测{int(row['valid_eval'])}个, "
                f"平均最大收益{(row['avg_max_return_pct'] if pd.notna(row['avg_max_return_pct']) else 0):.2f}%, "
                f"最佳{(row['best_max_return_pct'] if pd.notna(row['best_max_return_pct']) else 0):.2f}%"
            )
    lines.append("")

    lines.append("二、按阈值汇总")
    lines.append("=" * 80)
    if summary_df.empty:
        lines.append("无有效阈值汇总结果")
    else:
        for bp_name, g in summary_df.groupby("buy_point"):
            lines.append(f"【{bp_name}】")
            lines.append(f"{'阈值':<8}{'信号数':<8}{'平均最大收益':<14}{'中位最大收益':<14}{'>0占比':<10}{'>=3%':<10}{'>=5%':<10}{'最佳收益':<10}")
            for _, row in g.sort_values("threshold").iterrows():
                lines.append(
                    f"{row['threshold']:<8.2f}{int(row['signals']):<8}"
                    f"{((row['avg_max_return_pct'] if pd.notna(row['avg_max_return_pct']) else 0)):<14.2f}"
                    f"{((row['median_max_return_pct'] if pd.notna(row['median_max_return_pct']) else 0)):<14.2f}"
                    f"{((row['win_rate_pct'] if pd.notna(row['win_rate_pct']) else 0)):<10.2f}"
                    f"{((row['gt_3_pct'] if pd.notna(row['gt_3_pct']) else 0)):<10.2f}"
                    f"{((row['gt_5_pct'] if pd.notna(row['gt_5_pct']) else 0)):<10.2f}"
                    f"{((row['best_max_return_pct'] if pd.notna(row['best_max_return_pct']) else 0)):<10.2f}"
                )
            lines.append("")

    lines.append("三、Top信号（按最大收益）")
    lines.append("=" * 80)
    top_df = raw_df[raw_df["eval_status"] == "ok"].sort_values("max_return_pct", ascending=False).head(20)
    if top_df.empty:
        lines.append("无有效Top信号")
    else:
        lines.append(f"{'买点':<14}{'代码':<10}{'信号日':<12}{'分数':<10}{'最大收益':<12}{'发生日期':<12}")
        for _, row in top_df.iterrows():
            lines.append(
                f"{row['buy_point']:<14}{row['symbol']:<10}{pd.Timestamp(row['signal_date']).strftime('%Y-%m-%d'):<12}"
                f"{float(row['pred_score']):<10.4f}{float(row['max_return_pct']):<12.2f}{str(row['max_return_date']):<12}"
            )
    return "\n".join(lines)


def send_mail(subject: str, content: str) -> None:
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(FROM_EMAIL, SMTP_AUTH_CODE)
        server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="按周评测历史预测结果并发送邮件")
    parser.add_argument("--history-dir", type=str, default="output/history/daily_predictions")
    parser.add_argument("--data-root", type=str, default="local_data")
    parser.add_argument("--feature-config", type=str, default="configs/features.yaml")
    parser.add_argument("--report-date", type=str, default="")
    parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output-dir", type=str, default="output/weekly_eval")
    parser.add_argument("--no-mail", action="store_true")
    args = parser.parse_args()

    report_date = get_last_sunday(pd.Timestamp(args.report_date)) if args.report_date else get_last_sunday()
    pred_start, pred_end = get_previous_week_range(report_date)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    raw_df = load_prediction_history(args.history_dir, pred_start, pred_end)
    if not raw_df.empty:
        eval_rows = []
        for _, row in raw_df.iterrows():
            info = calc_signal_max_return(row, report_date, args.data_root, args.feature_config)
            eval_rows.append(info)
        eval_df = pd.concat([raw_df.reset_index(drop=True), pd.DataFrame(eval_rows)], axis=1)
        valid_df = eval_df[eval_df["eval_status"] == "ok"].copy()
    else:
        eval_df = pd.DataFrame()
        valid_df = pd.DataFrame()

    summary_df = summarize_by_threshold(valid_df, thresholds) if not valid_df.empty else pd.DataFrame()

    os.makedirs(args.output_dir, exist_ok=True)
    eval_path = Path(args.output_dir) / f"weekly_eval_signals_{report_date.strftime('%Y%m%d')}.csv"
    summary_path = Path(args.output_dir) / f"weekly_eval_summary_{report_date.strftime('%Y%m%d')}.csv"
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    content = format_mail_content(eval_df, summary_df, pred_start, pred_end, report_date)
    print(content)
    print(f"明细输出: {eval_path}")
    print(f"汇总输出: {summary_path}")

    if not args.no_mail:
        subject = f"周度买点评测 {pred_start.strftime('%Y-%m-%d')}~{pred_end.strftime('%Y-%m-%d')} 截止{report_date.strftime('%Y-%m-%d')}"
        send_mail(subject, content)
        print("邮件发送成功")


if __name__ == "__main__":
    main()
