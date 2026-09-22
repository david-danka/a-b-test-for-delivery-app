"""
A/B Test Analysis for Delivery App (command-line runner)

Runs the full A/B testing workflow and prints the results:
- Sample validation (A/A test)
- Sample size estimation (power analysis)
- Exploratory data analysis (EDA)
- Non-parametric hypothesis testing
- Parametric validation via log transformation

Charts open in your browser between stages. Use --no-plots for a text-only run
without charts or pauses.
"""

import argparse

import plotly.graph_objects as go

from abtest import plots, stats
from abtest.data import (
    OUTLIER_COLUMNS,
    OUTLIER_PERCENTILE,
    load_aa_test,
    load_ab_test,
    remove_outliers,
)


def pause(interactive: bool) -> None:
    if interactive:
        input("\nPress Enter to continue\n")


def show(fig: go.Figure, interactive: bool) -> None:
    if interactive:
        fig.show()


def print_test_result(result: stats.HypothesisTestResult) -> None:
    print(result.test_name)
    print(
        f"{result.statistic_label} = {result.statistic:.3f}, "
        f"p-value {'<=' if result.reject_null else '>'} {result.alpha}"
    )
    print(f"Reject null hypothesis: {'yes' if result.reject_null else 'no'}")
    print(f"{result.null_hypothesis}: {'no' if result.reject_null else 'yes'}")


def print_mean_comparison(comparison: stats.MeanComparison) -> None:
    print_test_result(comparison.levene)
    print()
    print_test_result(comparison.ttest)
    print()


def main(interactive: bool = True) -> None:
    aa_test = load_aa_test()
    ab_test = load_ab_test()
    cleaned = remove_outliers(ab_test, OUTLIER_COLUMNS, OUTLIER_PERCENTILE)

    print("=== Stage 1: Sample Size Verification ===")
    comparison = stats.verify_sample_size(aa_test, stats.ALPHA)
    print_mean_comparison(comparison)
    print(stats.sample_size_verdict(comparison))
    pause(interactive)

    print("\n=== Stage 2: Sample Size Estimation ===")
    sample_size = stats.estimate_sample_size(
        effect_size=0.2, power=0.8, alpha=stats.ALPHA
    )
    group_sizes = stats.count_groups(ab_test)
    print(f"Sample size: {sample_size}")
    print()
    print(f"Control group: {group_sizes.control}")
    print(f"Experimental group: {group_sizes.experimental}")
    pause(interactive)

    print("\n=== Stage 3: Exploratory Data Analysis ===")
    show(plots.sessions_by_day(ab_test), interactive)
    show(
        plots.histogram_by_group(
            ab_test, "order_value", "Distribution of order value by group"
        ),
        interactive,
    )
    show(
        plots.histogram_by_group(
            ab_test, "session_duration", "Distribution of session duration by group"
        ),
        interactive,
    )
    summary = stats.summarize_order_value(cleaned)
    print(f"Mean: {summary.mean:.2f}")
    print(f"Standard deviation: {summary.std:.2f}")
    print(f"Max: {summary.max}")
    pause(interactive)

    print("\n=== Stage 4: Non-parametric Test ===")
    print_test_result(stats.compare_distributions(cleaned, stats.ALPHA))
    pause(interactive)

    print("\n=== Stage 5: Parametric Test ===")
    show(plots.log_order_value_histogram(cleaned), interactive)
    aov = stats.compare_aov(cleaned, stats.ALPHA)
    print_mean_comparison(aov.comparison)
    print("Interpretation:")
    if aov.comparison.means_differ:
        print(f"Control AOV: {aov.control_aov:.2f}")
        print(f"Experimental AOV: {aov.experimental_aov:.2f}")
    print(aov.interpretation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A/B test analysis.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip the charts and the pauses between stages (text output only)",
    )
    main(interactive=not parser.parse_args().no_plots)
