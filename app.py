"""
Streamlit app: A/B Test Analysis for Delivery App

An interactive walkthrough of the five-stage analysis in main.py, reusing
abtest/ for every statistic and chart. Data is fixed to aa_test.csv and
ab_test.csv, bundled with the app -- there is no upload or data-entry step.
"""

import streamlit as st

from abtest import plots, stats
from abtest.data import (
    OUTLIER_COLUMNS,
    OUTLIER_PERCENTILE,
    load_aa_test,
    load_ab_test,
    remove_outliers,
)

st.set_page_config(
    page_title="A/B Test: Delivery App",
    page_icon="\U0001F69A",  # delivery truck
    layout="wide",
)


@st.cache_data
def get_data():
    aa_test = load_aa_test()
    ab_test = load_ab_test()
    cleaned = remove_outliers(ab_test, OUTLIER_COLUMNS, OUTLIER_PERCENTILE)
    return aa_test, ab_test, cleaned


def current_theme() -> str:
    """Streamlit's active theme, defaulting to light if it can't be read."""
    return st.context.theme.type or "light"


def format_p(p_value: float) -> str:
    return f"{p_value:.2e}" if p_value < 0.0001 else f"{p_value:.4f}"


def render_test(result: stats.HypothesisTestResult) -> None:
    """A three-metric row for one hypothesis test, plus its H0 as a caption."""
    stat_col, p_col, verdict_col = st.columns(3)
    stat_col.metric(result.statistic_label, f"{result.statistic:.3f}")
    p_col.metric("p-value", format_p(result.p_value))
    verdict_col.metric(
        "Verdict",
        "Reject H₀" if result.reject_null else "Fail to reject H₀",
    )
    outcome = "rejected" if result.reject_null else "not rejected"
    st.caption(f"H₀: {result.null_hypothesis}. {outcome.capitalize()} at α = {result.alpha}.")


def render_mean_comparison(comparison: stats.MeanComparison) -> None:
    equal_variances = not comparison.levene.reject_null
    variant = "Student's" if equal_variances else "Welch's"
    st.markdown(f"**Levene's test** (equal variances?)")
    render_test(comparison.levene)
    st.markdown(f"**T-test** ({variant} variant, chosen from the Levene result above)")
    render_test(comparison.ttest)


def main() -> None:
    aa_test, ab_test, cleaned = get_data()
    theme = current_theme()

    st.title("A/B Test Analysis for Delivery App")
    st.markdown(
        "A new web interface was tested against the existing one, aiming to "
        "increase average order value (AOV). This walkthrough follows the "
        "same five-stage pipeline as `main.py`, from sample-size validation "
        "through to the final parametric test."
    )

    tabs = st.tabs(
        [
            "1. Sample Size Verification",
            "2. Sample Size Estimation",
            "3. Exploratory Data Analysis",
            "4. Non-parametric Test",
            "5. Parametric Test",
        ]
    )

    with tabs[0]:
        st.header("Sample Size Verification (A/A Test)")
        st.markdown(
            "Two samples were drawn **before** the new interface shipped. If "
            "their means differ, the sample size isn't yet large enough for "
            "a reliable A/B test."
        )
        comparison = stats.verify_sample_size(aa_test, stats.ALPHA)
        render_mean_comparison(comparison)
        verdict = stats.sample_size_verdict(comparison)
        (st.warning if comparison.means_differ else st.success)(verdict)

    with tabs[1]:
        st.header("Sample Size Estimation")
        st.markdown(
            "Power analysis for a two-sample t-test, using a small effect "
            "size (Cohen's d = 0.2), 80% power and α = 0.05."
        )
        required = stats.estimate_sample_size(effect_size=0.2, power=0.8, alpha=stats.ALPHA)
        group_sizes = stats.count_groups(ab_test)
        req_col, control_col, exp_col = st.columns(3)
        req_col.metric("Required per group", required)
        control_col.metric(
            "Control group (actual)",
            group_sizes.control,
            delta=group_sizes.control - required,
        )
        exp_col.metric(
            "Experimental group (actual)",
            group_sizes.experimental,
            delta=group_sizes.experimental - required,
        )

    with tabs[2]:
        st.header("Exploratory Data Analysis")
        st.plotly_chart(plots.sessions_by_day(ab_test, theme=theme), width="stretch")
        st.plotly_chart(
            plots.histogram_by_group(
                ab_test, "order_value", "Distribution of order value by group", theme=theme
            ),
            width="stretch",
        )
        st.plotly_chart(
            plots.histogram_by_group(
                ab_test, "session_duration", "Distribution of session duration by group", theme=theme
            ),
            width="stretch",
        )
        st.subheader("Order value, with the top 1% trimmed")
        summary = stats.summarize_order_value(cleaned)
        mean_col, std_col, max_col = st.columns(3)
        mean_col.metric("Mean", f"{summary.mean:.2f}")
        std_col.metric("Standard deviation", f"{summary.std:.2f}")
        max_col.metric("Max", f"{summary.max:.2f}")

    with tabs[3]:
        st.header("Non-parametric Test")
        st.markdown(
            "Order values are not normally distributed, so group "
            "distributions are compared with a **Mann-Whitney U test** "
            "rather than a t-test. Run on the outlier-trimmed data."
        )
        render_test(stats.compare_distributions(cleaned, stats.ALPHA))

    with tabs[4]:
        st.header("Parametric Test")
        st.markdown(
            "To validate the non-parametric result, order values are "
            "**log-transformed** to approximate a normal distribution, then "
            "compared with a t-test. This tests whether *geometric* means "
            "differ; the raw average order value (AOV) is reported "
            "separately below."
        )
        st.plotly_chart(plots.log_order_value_histogram(cleaned, theme=theme), width="stretch")
        aov = stats.compare_aov(cleaned, stats.ALPHA)
        render_mean_comparison(aov.comparison)

        st.subheader("Average order value (AOV)")
        if aov.comparison.means_differ:
            control_col, exp_col = st.columns(2)
            control_col.metric("Control AOV", f"{aov.control_aov:.2f}")
            exp_col.metric(
                "Experimental AOV",
                f"{aov.experimental_aov:.2f}",
                delta=f"{aov.experimental_aov - aov.control_aov:+.2f}",
            )
            st.success(aov.interpretation)
        else:
            st.info(aov.interpretation)


if __name__ == "__main__":
    main()
