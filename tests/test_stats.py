"""Tests for abtest.stats.

The "original script" tests pin the numbers the pre-refactor main.py printed
for the bundled datasets, so any change to the analysis shows up here.
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from abtest import stats
from abtest.data import (
    OUTLIER_COLUMNS,
    OUTLIER_PERCENTILE,
    load_aa_test,
    load_ab_test,
    remove_outliers,
)

ALPHA = 0.05


def make_result(p_value: float) -> stats.HypothesisTestResult:
    return stats.HypothesisTestResult(
        test_name="T-test",
        statistic_label="t",
        statistic=1.0,
        p_value=p_value,
        alpha=ALPHA,
        null_hypothesis="Means are equal",
    )


def make_comparison(t_test_p_value: float) -> stats.MeanComparison:
    return stats.MeanComparison(levene=make_result(0.5), ttest=make_result(t_test_p_value))


@pytest.fixture(scope="module")
def ab_test():
    return load_ab_test()


@pytest.fixture(scope="module")
def cleaned(ab_test):
    return remove_outliers(ab_test, OUTLIER_COLUMNS, OUTLIER_PERCENTILE)


# --- behavior that does not depend on the datasets ---------------------------


def test_reject_null_when_p_value_equals_alpha():
    assert make_result(ALPHA).reject_null
    assert not make_result(ALPHA + 0.001).reject_null


def test_compare_means_uses_welch_when_variances_differ():
    rng = np.random.default_rng(0)
    narrow = rng.normal(0, 1, 60)
    wide = rng.normal(1, 6, 60)

    result = stats.compare_means(narrow, wide, ALPHA)

    assert result.levene.reject_null
    expected = scipy_stats.ttest_ind(narrow, wide, equal_var=False)
    assert result.ttest.statistic == pytest.approx(expected.statistic)
    assert result.ttest.p_value == pytest.approx(expected.pvalue)


def test_compare_means_uses_student_when_variances_match():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 60)
    shifted = sample + 1  # identical variance

    result = stats.compare_means(sample, shifted, ALPHA)

    assert not result.levene.reject_null
    expected = scipy_stats.ttest_ind(sample, shifted, equal_var=True)
    assert result.ttest.statistic == pytest.approx(expected.statistic)


def test_sample_size_verdict():
    assert stats.sample_size_verdict(make_comparison(0.01)) == (
        "Means are not equal -> larger sample size needed."
    )
    assert stats.sample_size_verdict(make_comparison(0.5)) == (
        "Means are equal -> sample size is sufficient."
    )


@pytest.mark.parametrize(
    ("t_test_p_value", "control_aov", "experimental_aov", "expected"),
    [
        (0.5, 30.0, 40.0, "No statistically significant difference in AOV."),
        (
            0.01, 30.0, 40.0,
            "Statistically significant increase in AOV observed in experimental group.",
        ),
        (
            0.01, 40.0, 30.0,
            "Statistically significant decrease in AOV observed in experimental group.",
        ),
    ],
)
def test_aov_interpretation(t_test_p_value, control_aov, experimental_aov, expected):
    aov = stats.AovComparison(make_comparison(t_test_p_value), control_aov, experimental_aov)

    assert aov.interpretation == expected


def test_estimate_sample_size_rounds_up_to_multiple():
    exact = stats.estimate_sample_size(0.2, 0.8, ALPHA, round_to=1)
    rounded = stats.estimate_sample_size(0.2, 0.8, ALPHA, round_to=100)

    assert rounded % 100 == 0
    assert exact <= rounded < exact + 100


# --- numbers the original script printed for the bundled data ----------------


def test_sample_size_verification_matches_original_script():
    comparison = stats.verify_sample_size(load_aa_test(), ALPHA)

    assert comparison.levene.statistic == pytest.approx(0.0, abs=5e-4)
    assert not comparison.levene.reject_null
    assert comparison.ttest.statistic == pytest.approx(-3.432, abs=5e-4)
    assert comparison.ttest.reject_null


def test_sample_size_estimate_matches_original_script():
    assert stats.estimate_sample_size(0.2, 0.8, ALPHA) == 400


def test_group_sizes_match_original_script(ab_test):
    assert stats.count_groups(ab_test) == stats.GroupSizes(control=400, experimental=400)


def test_order_value_summary_matches_original_script(cleaned):
    summary = stats.summarize_order_value(cleaned)

    assert summary.mean == pytest.approx(33.75, abs=5e-3)
    assert summary.std == pytest.approx(32.19, abs=5e-3)
    assert summary.max == pytest.approx(202.98, abs=5e-3)


def test_distribution_test_matches_original_script(cleaned):
    result = stats.compare_distributions(cleaned, ALPHA)

    assert result.statistic == pytest.approx(60612.0, abs=5e-4)
    assert result.reject_null


def test_aov_comparison_matches_original_script(cleaned):
    aov = stats.compare_aov(cleaned, ALPHA)

    assert aov.comparison.levene.statistic == pytest.approx(30.174, abs=5e-4)
    assert aov.comparison.levene.reject_null
    assert aov.comparison.ttest.statistic == pytest.approx(-5.859, abs=5e-4)
    assert aov.comparison.ttest.reject_null
    assert aov.control_aov == pytest.approx(30.58, abs=5e-3)
    assert aov.experimental_aov == pytest.approx(36.90, abs=5e-3)
    assert aov.interpretation == (
        "Statistically significant increase in AOV observed in experimental group."
    )
