#!/bin/bash

# ============================================
# 每日自动同步数据 -> 预测 -> 归档预测结果 -> 发送邮件（仅买点2）
# ============================================

PROJECT_DIR="/data2/xiaohui36/project"
cd "$PROJECT_DIR" || exit 1

source /data0/miniconda3/etc/profile.d/conda.sh
conda activate xiaohui36

TODAY=$(date +%Y%m%d)
PREDICT_LOG="output/predict_${TODAY}.log"

MODEL_BP2="output/buy_point_2_v2_lgbm_model.joblib"
MODELS="${MODEL_BP2}"
ARCHIVE_DIR="output/history/daily_predictions/${TODAY}"

echo "========================================" | tee "$PREDICT_LOG"
echo "开始执行: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$PREDICT_LOG"
echo "日期: $TODAY" | tee -a "$PREDICT_LOG"
echo "步骤1: 同步本地数据与特征缓存" | tee -a "$PREDICT_LOG"
echo "========================================" | tee -a "$PREDICT_LOG"

python -u sync_market_data.py \
  --mode market \
  --start 20210101 \
  --end "$TODAY" \
  --data-root local_data \
  2>&1 | tee -a "$PREDICT_LOG"

echo "" | tee -a "$PREDICT_LOG"
echo ">>> 步骤2: 基于本地最新缓存预测（仅买点2，且先过候选规则） <<<" | tee -a "$PREDICT_LOG"
python -u predict_multi_models.py \
  --mode market \
  --model-paths "$MODELS" \
  --start 20210101 \
  --end "$TODAY" \
  --limit 0 \
  --top-n 10 \
  --min-score 0.0 \
  2>&1 | tee -a "$PREDICT_LOG"

echo "" | tee -a "$PREDICT_LOG"
echo ">>> 步骤3: 归档今日预测结果 <<<" | tee -a "$PREDICT_LOG"
mkdir -p "$ARCHIVE_DIR"
find output -maxdepth 1 -type f -name 'buy_point_*_live_predictions.csv' -exec cp {} "$ARCHIVE_DIR"/ \;

echo "" | tee -a "$PREDICT_LOG"
echo ">>> 步骤4: 发送邮件 <<<" | tee -a "$PREDICT_LOG"
python -u send_top10_mail.py 2>&1 | tee -a "$PREDICT_LOG"

echo "========================================" | tee -a "$PREDICT_LOG"
echo "全部完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$PREDICT_LOG"
echo "========================================" | tee -a "$PREDICT_LOG"
