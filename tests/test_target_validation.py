
import pandas as pd
import pytest


def calculate_is_late(actual_delivery, estimated_delivery):
    actual = pd.to_datetime(actual_delivery, errors="coerce")
    estimated = pd.to_datetime(estimated_delivery, errors="coerce")

    if pd.isna(actual) or pd.isna(estimated):
        return None

    return int(actual.normalize() > estimated.normalize())


def test_delivery_on_expected_day():
    result = calculate_is_late(
        "2018-05-10 18:30:00",
        "2018-05-10 00:00:00"
    )

    assert result == 0


def test_delivery_after_expected_day():
    result = calculate_is_late(
        "2018-05-11 09:00:00",
        "2018-05-10 00:00:00"
    )

    assert result == 1


def test_missing_actual_delivery():
    result = calculate_is_late(
        None,
        "2018-05-10"
    )

    assert result is None


def test_delivery_before_expected_day():
    result = calculate_is_late(
        "2018-05-08",
        "2018-05-10"
    )

    assert result == 0


def test_missing_estimated_delivery():
    result = calculate_is_late(
        "2018-05-10",
        None
    )

    assert result is None
