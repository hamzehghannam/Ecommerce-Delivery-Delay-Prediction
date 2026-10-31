
from pathlib import Path
import ast

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "11_model_features.py"

FORBIDDEN_COLUMNS = {
    "order_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_id",
    "customer_id",
    "customer_unique_id",
    "is_late",
    "actual_delivery_days",
    "delivery_delay_days",
    "shipping_limit_date",
}

def test_model_features_exclude_leakage():
    source = MODEL_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    feature_list = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "model_features"
                ):
                    feature_list = ast.literal_eval(node.value)

    assert feature_list is not None
    assert len(feature_list) > 0
    assert len(feature_list) == len(set(feature_list))

    forbidden_found = (
        set(feature_list) & FORBIDDEN_COLUMNS
    )

    assert not forbidden_found, (
        f"Forbidden model features: {forbidden_found}"
    )
