"""Statistical analysis of the A/A and A/B tests.

Every function takes data in and returns a plain result object. Nothing here
prints, plots or waits for input, so the same results can feed a CLI, a
Streamlit app or a test.
"""

import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power

from abtest.data import CONTROL, EXPERIMENTAL, add_log_order_value, split_groups

ALPHA = 0.05


@dataclass(frozen=True)
class HypothesisTestResult:
    """Outcome of one hypothesis test."""

    test_name: str
    statistic_label: str
    statistic: float
    p_value: float
    alpha: float
    null_hypothesis: str

    @property
    def reject_null(self) -> bool:
        return self.p_value <= self.alpha


@dataclass(frozen=True)
class MeanComparison:
    """Levene's test, then a t-test that assumes equal variances only if Levene's does."""

    levene: HypothesisTestResult
    ttest: HypothesisTestResult

    @property
    def means_differ(self) -> bool:
        return self.ttest.reject_null


@dataclass(frozen=True)
class GroupSizes:
    control: int
    experimental: int


@dataclass(frozen=True)
class OrderValueSummary:
    mean: float
    std: float
    max: float


@dataclass(frozen=True)
class AovComparison:
    """Log-scale t-test of order values plus the raw average order value per group."""

    comparison: MeanComparison
    control_aov: float
    experimental_aov: float

    @property
    def interpretation(self) -> str:
        if not self.comparison.means_differ:
            return "No statistically significant difference in AOV."
        direction = "decrease" if self.control_aov > self.experimental_aov else "increase"
        return (
            f"Statistically significant {direction} in AOV "
            "observed in experimental group."
        )


def compare_means(
    sample_a: pd.Series,
    sample_b: pd.Series,
    alpha: float,
) -> MeanComparison:
    """Compare two samples' means, choosing the t-test variant from Levene's test."""
    levene = stats.levene(sample_a, sample_b)
    equal_variances = levene.pvalue > alpha
    t_test = stats.ttest_ind(sample_a, sample_b, equal_var=equal_variances)

    return MeanComparison(
        levene=HypothesisTestResult(
            test_name="Levene's test",
            statistic_label="W",
            statistic=float(levene.statistic),
            p_value=float(levene.pvalue),
            alpha=alpha,
            null_hypothesis="Variances are equal",
        ),
        ttest=HypothesisTestResult(
            test_name="T-test",
            statistic_label="t",
            statistic=float(t_test.statistic),
            p_value=float(t_test.pvalue),
            alpha=alpha,
            null_hypothesis="Means are equal",
        ),
    )


def verify_sample_size(aa_test: pd.DataFrame, alpha: float) -> MeanComparison:
    """Compare the means of the two A/A test samples."""
    return compare_means(aa_test["Sample 1"], aa_test["Sample 2"], alpha)


def sample_size_verdict(comparison: MeanComparison) -> str:
    if comparison.means_differ:
        return "Means are not equal -> larger sample size needed."
    return "Means are equal -> sample size is sufficient."


def estimate_sample_size(
    effect_size: float,
    power: float,
    alpha: float,
    round_to: int = 100,
) -> int:
    """Estimate the required size of each group, rounded up to a multiple of round_to."""
    sample_size = tt_ind_solve_power(
        effect_size=effect_size,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=1.0,
    )
    return math.ceil(sample_size / round_to) * round_to


def count_groups(ab_test: pd.DataFrame) -> GroupSizes:
    return GroupSizes(
        control=int((ab_test["group"] == CONTROL).sum()),
        experimental=int((ab_test["group"] == EXPERIMENTAL).sum()),
    )


def summarize_order_value(ab_test: pd.DataFrame) -> OrderValueSummary:
    order_value = ab_test["order_value"]
    return OrderValueSummary(
        mean=float(order_value.mean()),
        std=float(order_value.std(ddof=0)),
        max=float(order_value.max()),
    )


def compare_distributions(ab_test: pd.DataFrame, alpha: float) -> HypothesisTestResult:
    """Mann-Whitney U test of order values, used because they are not normally distributed."""
    control, experimental = split_groups(ab_test, "order_value")
    result = stats.mannwhitneyu(control, experimental)

    return HypothesisTestResult(
        test_name="Mann-Whitney U test",
        statistic_label="U1",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        null_hypothesis="Distributions are same",
    )


def compare_aov(ab_test: pd.DataFrame, alpha: float) -> AovComparison:
    """Parametric test of order values after a log transform to approximate normality."""
    log_control, log_experimental = split_groups(
        add_log_order_value(ab_test), "log_order_value"
    )
    control, experimental = split_groups(ab_test, "order_value")

    return AovComparison(
        comparison=compare_means(log_control, log_experimental, alpha),
        control_aov=float(control.mean()),
        experimental_aov=float(experimental.mean()),
    )
