from pathlib import Path
from utils.config_loader import load_yaml_config


def load_buy_point_config(config_path: str = "configs/buy_points.yaml") -> dict:
    return load_yaml_config(config_path)


def get_buy_point_definition(config: dict, buy_point_name: str | None = None) -> dict:
    buy_points = config.get("buy_points", {})
    default_name = config.get("default_buy_point")

    target_name = buy_point_name or default_name
    if not target_name:
        raise ValueError("未指定 buy_point，且配置文件中没有 default_buy_point")

    if target_name not in buy_points:
        raise ValueError(f"buy_point 不存在: {target_name}")

    result = buy_points[target_name].copy()
    result["name"] = target_name
    return result
