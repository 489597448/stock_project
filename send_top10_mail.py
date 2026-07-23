import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
import pandas as pd
from pandas.errors import EmptyDataError

# 仅买点2的预测结果文件
CSV_BP2 = "output/buy_point_buy_point_2_live_predictions.csv"

TO_EMAIL = "489597448@qq.com"

# ====== 邮箱配置 ======
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465
FROM_EMAIL = "m19821879218@163.com"
SMTP_AUTH_CODE = "UBgrW8SCQ9WVqm2E"
# ======================


def load_predictions(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, dtype={"symbol": str})
        df["symbol"] = df["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        return df
    except EmptyDataError:
        return pd.DataFrame()


def format_section(title, df):
    lines = [title]
    lines.append("=" * 40)
    if df.empty:
        lines.append("无符合条件的预测结果")
    else:
        lines.append(f"{'排名':<6}{'股票代码':<12}{'预测分数':<12}")
        lines.append("-" * 40)
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            score = row.get("pred_score", 0)
            symbol = row.get("symbol", "未知")
            lines.append(f"{idx:<6}{symbol:<12}{score:<12.6f}")
    lines.append("")
    return "\n".join(lines)


def main():
    df_bp2 = load_predictions(CSV_BP2)
    today_str = datetime.now().strftime("%Y-%m-%d")

    content_parts = [f"今日（{today_str}）买点2预测结果：", ""]
    top_bp2 = df_bp2.sort_values("pred_score", ascending=False).head(10) if not df_bp2.empty else pd.DataFrame()
    content_parts.append(format_section("【买点2】Top10 预测", top_bp2))
    content = "\n".join(content_parts)

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = Header(f"买点2预测Top10 {today_str}", "utf-8")
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(FROM_EMAIL, SMTP_AUTH_CODE)
            server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")


if __name__ == "__main__":
    main()
