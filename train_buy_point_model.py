import argparse
import os
from typing import Iterable
from utils.config_loader import load_feature_config, get_feature_set, load_model_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
from lightgbm import LGBMClassifier
from utils.config_loader import load_feature_config, get_feature_set, load_model_config
from buy_points.base import load_buy_point_config, get_buy_point_definition
import lightgbm as lgb

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


EXCLUDE_COLUMNS = {
    "symbol",
    "date",
    "label",
    "future_max_return_pct",
    "future_max_drawdown_pct",
}



def lgb_precision_eval(y_true, y_pred):
    """LightGBM 自定义评估函数：precision"""
    y_pred_bin = (y_pred >= 0.5).astype(int)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    return "precision", prec, True  # True 表示越大越好

def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False, dtype={"symbol": str})
    
    df = df.replace([np.inf, -np.inf], pd.NA)
    
    if df.empty:
        raise ValueError("训练数据为空")
        
    required = {"date", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    
    for col in df.columns:
        if col not in ["symbol", "date", "label"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # 核心修改：丢掉 label 为空的行
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    
    return df




def pick_feature_columns(
    df: pd.DataFrame,
    config_path: str = "configs/features.yaml",
    feature_set_name: str | None = None,
    custom_features: list[str] | None = None,
) -> tuple[list[str], str]:
    if custom_features:
        cols = [c.strip() for c in custom_features if c.strip() in df.columns]
        if not cols:
            raise ValueError("指定的特征列不存在")
        return cols, "manual_features"

    config = load_feature_config(config_path)
    target_name = feature_set_name or config.get("default_feature_set")
    feature_cols = get_feature_set(config, target_name)

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("配置中的特征列在训练数据中不存在")

    return cols, target_name



def time_split(df: pd.DataFrame, train_end: str, valid_end: str | None = None):
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end) if valid_end else None

    train_df = df[df["date"] <= train_end_ts].copy()
    if valid_end_ts is None:
        test_df = df[df["date"] > train_end_ts].copy()
        valid_df = pd.DataFrame(columns=df.columns)
    else:
        valid_df = df[(df["date"] > train_end_ts) & (df["date"] <= valid_end_ts)].copy()
        test_df = df[df["date"] > valid_end_ts].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请调整 train_end")
    if test_df.empty and valid_df.empty:
        raise ValueError("验证集/测试集为空，请调整时间切分")

    return train_df, valid_df, test_df

def build_model_by_config(model_type: str, params: dict) -> Pipeline:
    model_type = model_type.lower()

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 8),
            class_weight=params.get("class_weight", "balanced_subsample"),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )

    elif model_type == "lightgbm":
        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            class_weight=params.get("class_weight", "balanced"),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
            verbosity=-1,            # 开启日志
        )

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def evaluate_binary_model(model: Pipeline, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5, dataset_name: str = "test") -> dict:
    if df.empty:
        return {"dataset": dataset_name, "rows": 0}

    X = df[feature_cols]
    y = df["label"].astype(int)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "positive_ratio": round(float(y.mean() * 100), 2),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if y.nunique() > 1 else None,
        "pred_positive_ratio": round(float(pred.mean() * 100), 2),
    }
    return metrics

def evaluate_precision_by_thresholds(model: Pipeline, df: pd.DataFrame, feature_cols: list[str], thresholds: list[float] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    X = df[feature_cols]
    y = df["label"].astype(int)
    proba = model.predict_proba(X)[:, 1]

    rows = []
    for th in thresholds:
        pred = (proba >= th).astype(int)
        picked = df[proba >= th]
        prec = precision_score(y, pred, zero_division=0)
        rows.append({
            "threshold": th,
            "picked_count": int(pred.sum()),
            "precision": round(float(prec), 4),
            "hit_rate_pct": round(float(picked["label"].mean() * 100), 2) if len(picked) > 0 else 0,
            "avg_future_return_pct": round(float(picked["future_max_return_pct"].mean()), 2) if len(picked) > 0 and "future_max_return_pct" in picked.columns else None,
            "avg_future_drawdown_pct": round(float(picked["future_max_drawdown_pct"].mean()), 2) if len(picked) > 0 and "future_max_drawdown_pct" in picked.columns else None,
        })

    return pd.DataFrame(rows)



def prediction_report(model: Pipeline, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> pd.DataFrame:
    result = df.copy()
    result["pred_score"] = model.predict_proba(result[feature_cols])[:, 1]
    result["pred_label"] = (result["pred_score"] >= threshold).astype(int)
    return result


def trading_summary(pred_df: pd.DataFrame, top_quantile: float = 0.2) -> dict:
    if pred_df.empty:
        return {}
    pred_df = pred_df.copy().sort_values("pred_score", ascending=False)
    top_n = max(1, int(len(pred_df) * top_quantile))
    top_df = pred_df.head(top_n)
    return {
        "top_count": int(len(top_df)),
        "top_hit_rate_pct": round(float(top_df["label"].mean() * 100), 2) if "label" in top_df.columns else None,
        "top_avg_future_return_pct": round(float(top_df["future_max_return_pct"].mean()), 2) if "future_max_return_pct" in top_df.columns else None,
        "top_avg_future_drawdown_pct": round(float(top_df["future_max_drawdown_pct"].mean()), 2) if "future_max_drawdown_pct" in top_df.columns else None,
    }


def save_feature_importance(model: Pipeline, feature_cols: list[str], output_path: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    importances = estimator.feature_importances_

    if len(importances) != len(feature_cols):
        feature_cols = feature_cols[:len(importances)]

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi.to_csv(output_path, index=False, encoding="utf-8-sig")
    return fi


def main() -> None:
    parser = argparse.ArgumentParser(description="训练买点识别模型")

    parser.add_argument("--config", type=str, default="configs/features.yaml")
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--buy-point", type=str, default=None)
    parser.add_argument("--buy-point-config", type=str, default="configs/buy_points.yaml")

    # 允许命令行覆盖配置

    parser.add_argument("--input", type=str, default="output/buy_point_training_samples.csv")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--valid-end", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--model-output", type=str, default="output/buy_point_model.joblib")
    parser.add_argument("--pred-output", type=str, default="output/buy_point_predictions.csv")
    parser.add_argument("--metrics-output", type=str, default="output/buy_point_metrics.csv")
    parser.add_argument("--feature-importance-output", type=str, default="output/buy_point_feature_importance.csv")
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    model_type = model_config.get("model", {}).get("type", "random_forest")
    model_params = model_config.get("model", {}).get("params", {})
    split_config = model_config.get("split", {})
    predict_config = model_config.get("predict", {})

    threshold_used = args.threshold if args.threshold is not None else predict_config.get("score_threshold", 0.5)

    train_end = args.train_end or split_config.get("train_end")
    valid_end = args.valid_end or split_config.get("valid_end")

    buy_point_config = load_buy_point_config(args.buy_point_config)
    buy_point_def = get_buy_point_definition(buy_point_config, args.buy_point)
    buy_point_name_used = buy_point_def["name"]



    os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)

    df = load_dataset(args.input)
    #feature_cols = pick_feature_columns(df)
    custom_features = args.features.split(",") if args.features else None

    feature_cols, feature_set_name_used = pick_feature_columns(
    df,
    config_path=args.config,
    feature_set_name=args.feature_set,
    custom_features=custom_features,
    )
    train_df, valid_df, test_df = time_split(df, train_end=train_end, valid_end=valid_end)

    #model = build_model(n_estimators=args.n_estimators, max_depth=args.max_depth)
    effective_model_params = dict(model_params)

    if args.n_estimators is not None:
        effective_model_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        effective_model_params["max_depth"] = args.max_depth

    model = build_model_by_config(model_type, effective_model_params)
    #model.fit(train_df[feature_cols], train_df["label"])
    if model_type == "lightgbm":
        model.fit(
            train_df[feature_cols],
            train_df["label"],
            model__eval_set=[(valid_df[feature_cols], valid_df["label"])],
            model__eval_metric=lgb_precision_eval,
            model__callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=10),
            ],
        )
    else:
        model.fit(train_df[feature_cols], train_df["label"])

    metrics_rows = [evaluate_binary_model(model, train_df, feature_cols, args.threshold, "train")]
    if not valid_df.empty:
        metrics_rows.append(evaluate_binary_model(model, valid_df, feature_cols, args.threshold, "valid"))
    if not test_df.empty:
        metrics_rows.append(evaluate_binary_model(model, test_df, feature_cols, args.threshold, "test"))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.metrics_output, index=False, encoding="utf-8-sig")

    pred_parts = []
    if not valid_df.empty:
        valid_pred = prediction_report(model, valid_df, feature_cols, args.threshold)
        valid_pred["dataset"] = "valid"
        pred_parts.append(valid_pred)
    if not test_df.empty:
        test_pred = prediction_report(model, test_df, feature_cols, args.threshold)
        test_pred["dataset"] = "test"
        pred_parts.append(test_pred)
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    pred_df.to_csv(args.pred_output, index=False, encoding="utf-8-sig")

    fi_df = save_feature_importance(model, feature_cols, args.feature_importance_output)


    payload = {
    "model": model,
    "feature_cols": feature_cols,
    "feature_set_name": feature_set_name_used,
    "feature_config_path": args.config,
    "buy_point_name": buy_point_name_used,
    "buy_point_config_path": args.buy_point_config,
    "train_end": train_end,
    "valid_end": valid_end,
    "threshold": threshold_used,
    "model_type": model_type,
    "model_config_path": args.model_config,
    "model_params": effective_model_params,
    }
    joblib.dump(payload, args.model_output)

    print("=" * 80)
    print("训练完成")
    print(f"使用特征数: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    print("-" * 80)
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(valid_df)}")
    print(f"测试集: {len(test_df)}")
    print("\n评估结果:")
    print(metrics_df.to_string(index=False))
    print("\nTop 10 特征重要性:")
    print(fi_df.head(10).to_string(index=False))
    if not pred_df.empty:
        print("\n验证/测试交易摘要:")
        for ds_name in pred_df["dataset"].drop_duplicates().tolist():
            summary = trading_summary(pred_df[pred_df["dataset"] == ds_name])
            print(ds_name, summary)
    print("\n输出文件:")
    print(f"模型: {args.model_output}")
    print(f"预测: {args.pred_output}")
    print(f"指标: {args.metrics_output}")
    print(f"特征重要性: {args.feature_importance_output}")
    print("=" * 80)

    if not test_df.empty:
        print("\n测试集多阈值 Precision 对比:")
        print("-" * 80)
        th_df = evaluate_precision_by_thresholds(model, test_df, feature_cols)
        print(th_df.to_string(index=False))
        th_df.to_csv(args.metrics_output.replace(".csv", "_thresholds.csv"), index=False, encoding="utf-8-sig")
        print(f"\n阈值分析已保存: {args.metrics_output.replace('.csv', '_thresholds.csv')}")


if __name__ == "__main__":
    main()
