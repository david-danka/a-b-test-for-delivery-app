import pandas as pd

from abtest.data import (
    CONTROL,
    EXPERIMENTAL,
    add_log_order_value,
    load_aa_test,
    load_ab_test,
    remove_outliers,
    split_groups,
)


def test_aa_test_has_two_samples():
    aa_test = load_aa_test()

    assert list(aa_test.columns) == ["Sample 1", "Sample 2"]


def test_ab_test_columns_and_group_labels():
    ab_test = load_ab_test()

    assert list(ab_test.columns) == [
        "session_id", "group", "date", "session_duration", "order_value",
    ]
    assert set(ab_test["group"]) == {CONTROL, EXPERIMENTAL}


def test_remove_outliers_drops_rows_high_in_any_column():
    df = pd.DataFrame({"a": range(1, 101), "b": range(100, 0, -1)})

    cleaned = remove_outliers(df, ["a", "b"], 99)

    # a=100 is above the cutoff in "a"; a=1 (so b=100) is above it in "b".
    assert len(cleaned) == 98
    assert cleaned["a"].max() == 99
    assert cleaned["b"].max() == 99


def test_remove_outliers_does_not_modify_input():
    df = pd.DataFrame({"a": range(1, 101)})

    remove_outliers(df, ["a"], 99)

    assert len(df) == 100


def test_split_groups_returns_control_then_experimental():
    df = pd.DataFrame({"group": [CONTROL, EXPERIMENTAL, CONTROL], "x": [1, 10, 3]})

    control, experimental = split_groups(df, "x")

    assert list(control) == [1, 3]
    assert list(experimental) == [10]


def test_add_log_order_value_is_natural_log():
    df = pd.DataFrame({"order_value": [1.0, 10.0]})

    result = add_log_order_value(df)

    assert result["log_order_value"].iloc[0] == 0.0
    assert abs(result["log_order_value"].iloc[1] - 2.302585) < 1e-6
    assert "log_order_value" not in df.columns
