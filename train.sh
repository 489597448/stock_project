python train_buy_point_model.py \
  --input output/buy_point_3_training_samples.csv \
  --buy-point buy_point_3 \
  --model-config configs/model.yaml \
  --model-output output/buy_point_3_lgbm_model.joblib \
  --pred-output output/buy_point_3_lgbm_predictions.csv \
  --metrics-output output/buy_point_3_lgbm_metrics.csv \
  --feature-importance-output output/buy_point_3_lgbm_feature_importance.csv
