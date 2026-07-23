#!/bin/bash

# ============================================
# 每周日晚：同步数据 -> 评测上周预测 -> 发送周报邮件
# ============================================

PROJECT_DIR="/data2/xiaohui36/project"
cd "$PROJECT_DIR" || exit 1

source /data0/miniconda3/etc/profile.d/conda.sh
conda activate xiaohui36

TODAY=$(date +%Y%m%d)
LOG_FILE="output/weekly_eval_${TODAY}.log"

echo "========================================" | tee "$LOG_FILE"
echo "开始执行周评测: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

echo ">>> 步骤1: 同步本地数据与特征缓存 <<<" | tee -a "$LOG_FILE"
python -u sync_market_data.py \
  --mode market \
  --start 20210101 \
  --end "$TODAY" \
  --data-root local_data \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> 步骤2: 评测上周预测并发送邮件 <<<" | tee -a "$LOG_FILE"
python -u evaluate_weekly_predictions.py \
  --history-dir output/history/daily_predictions \
  --data-root local_data \
  --output-dir output/weekly_eval \
  2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "周评测完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
