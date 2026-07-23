import pandas as pd
import numpy as np

def main():
    # 读取预测结果
    df = pd.read_csv("output/buy_point_2_v2_lgbm_predictions.csv", dtype={"symbol": str})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    
    # 只看测试集
    test_df = df[df["dataset"] == "test"].copy()
    
    # 筛选：阈值 >= 0.75 且 label == 0 (假阳性)
    false_positives = test_df[(test_df["pred_score"] >= 0.75) & (test_df["label"] == 0)].copy()
    
    print("=" * 80)
    print(f"测试集中阈值 >= 0.75 的样本总数: {len(test_df[test_df['pred_score'] >= 0.75])}")
    print(f"其中真阳例 (label=1) 数量: {len(test_df[(test_df['pred_score'] >= 0.75) & (test_df['label'] == 1)])}")
    print(f"其中假阳性 (label=0) 数量: {len(false_positives)}")
    print("=" * 80)
    
    if false_positives.empty:
        print("没有假阳性样本")
        return

    # 分析这些假阳性的后续走势
    print("\n假阳性样本后续走势统计:")
    print("-" * 80)
    
    ret_col = "future_max_return_pct"
    dd_col = "future_max_drawdown_pct"
    
    stats = false_positives[[ret_col, dd_col]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(stats.to_string())
    
    print("\n假阳性样本分类:")
    print("-" * 80)
    
    # 分类1：还是涨了，但没到18%的门槛
    mild_gain = false_positives[(false_positives[ret_col] > 0) & (false_positives[ret_col] < 18)]
    print(f"涨幅 0~18% (没达标但没亏): {len(mild_gain)} 只 ({len(mild_gain)/len(false_positives)*100:.1f}%)")
    
    # 分类2：几乎没涨或小亏
    flat = false_positives[(false_positives[ret_col] <= 0) & (false_positives[dd_col] >= -6)]
    print(f"涨幅 <=0 且回撤 >-6% (横盘小亏): {len(flat)} 只 ({len(flat)/len(false_positives)*100:.1f}%)")
    
    # 分类3：直接大跌 (回撤超过 -6%)
    big_drop = false_positives[false_positives[dd_col] < -6]
    print(f"回撤 <-6% (直接大跌): {len(big_drop)} 只 ({len(big_drop)/len(false_positives)*100:.1f}%)")
    
    # 打印部分具体案例
    print("\n假阳性样本案例 (按预测分数排序):")
    print("-" * 80)
    cols_to_show = ["symbol", "date", "pred_score", "label", ret_col, dd_col, "close", "breakout_strength", "turnover_today"]
    cols_to_show = [c for c in cols_to_show if c in false_positives.columns]
    print(false_positives.sort_values("pred_score", ascending=False)[cols_to_show].head(20).to_string(index=False))

if __name__ == "__main__":
    main()