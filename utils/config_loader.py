from pathlib import Path
import yaml


def load_yaml_config(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_feature_config(config_path: str = "configs/features.yaml") -> dict:
    return load_yaml_config(config_path)


def load_model_config(config_path: str = "configs/model.yaml") -> dict:
    return load_yaml_config(config_path)


def get_feature_set(config: dict, feature_set_name: str | None = None) -> list[str]:
    feature_sets = config.get("train_feature_sets", {})
    default_set = config.get("default_feature_set")

    target_name = feature_set_name or default_set
    if not target_name:
        raise ValueError("未指定 feature_set，且配置文件中没有 default_feature_set")

    if target_name not in feature_sets:
        raise ValueError(f"feature_set 不存在: {target_name}")

    return feature_sets[target_name]
