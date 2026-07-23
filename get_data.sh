rm -f output/buy_point_3_training_samples.csv
rm -f output/buy_point_3_training_errors.csv

python -u generate_buy_point_samples.py \
  --mode market \
  --start 20210101 \
  --end 20260701 \
  --limit 0 \
  --buy-point buy_point_3 \
  --delay 0.2 \
  --save-every 100 \
  --output output/buy_point_3_training_samples.csv \
  --error-log output/buy_point_3_training_errors.csv
