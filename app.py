"""
Streamlit app: A/B Test Analysis for Delivery App

An interactive walkthrough of the five-stage analysis in main.py, reusing
abtest/ for every statistic and chart. Data is fixed to aa_test.csv and
ab_test.csv, bundled with the app: there is no upload or data-entry step.

The bundled datasets are teaching data with a disclosed, exact markup built
in (see tab 1 and abtest.stats.detect_constant_offset), not a real business
experiment. The five stages demonstrate the full A/B testing methodology,
checking the experiment design is trustworthy before testing the hypothesis,
and the pipeline correctly recovers that known markup with strong
statistical confidence.
"""

import pandas as pd
import streamlit as st

from abtest import plots, stats
from abtest.data import (
    OUTLIER_COLUMNS,
    OUTLIER_PERCENTILE,
    load_aa_test,
    load_ab_test,
    remove_outliers,
    split_groups,
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


@st.cache_data
def get_effect_size(cleaned: pd.DataFrame) -> stats.EffectSize:
    return stats.estimate_effect_size(cleaned)


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


def render_raw_data(df: pd.DataFrame, filename: str) -> None:
    """A collapsed table plus a download button for one bundled CSV."""
    with st.expander(f"Show the raw data ({filename})"):
        st.dataframe(df, width="stretch")
        st.download_button(
            f"Download {filename}",
            data=df.to_csv(index=False),
            file_name=filename,
            mime="text/csv",
        )


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

    # Every result used more than once (in the hero and in its tab) is
    # computed once here, rather than re-run per tab.
    comparison = stats.verify_sample_size(aa_test, stats.ALPHA)
    mann_whitney = stats.compare_distributions(cleaned, stats.ALPHA)
    aov = stats.compare_aov(cleaned, stats.ALPHA)
    effect = get_effect_size(cleaned)
    both_significant = mann_whitney.reject_null and aov.comparison.means_differ
    direction = "increased" if effect.diff > 0 else "decreased"

    control_ov, experimental_ov = split_groups(ab_test, "order_value")
    aa_offset = stats.detect_constant_offset(aa_test["Sample 1"], aa_test["Sample 2"])
    ab_offset = stats.detect_constant_offset(control_ov, experimental_ov)

    st.title("A/B Test Analysis for Delivery App")
    st.markdown(
        "**The premise:** this project runs a full A/B testing workflow "
        "end to end, from checking the experiment's own design through to "
        "a validated conclusion. The scenario: a food delivery app's menu "
        "page was redesigned to surface popular items at the top, on the "
        "theory that easier to find items raise the **average order "
        "value (AOV)**. The bundled data is a teaching dataset with a "
        "disclosed, exact markup built into it (see tab 1), so what "
        "follows is a demonstration of the methodology, not a real "
        "business result."
    )

    if both_significant:
        st.success(
            f"The pipeline correctly recovers the built-in difference: "
            f"order value {direction} from \\${aov.control_aov:.2f} to "
            f"\\${aov.experimental_aov:.2f} in this data, a "
            f"{effect.lift_pct:+.1f}% shift, detected by two independent "
            f"tests (Mann-Whitney U and a log-scale t-test)."
        )
    else:
        st.info(
            "The two tests don't agree on a significant difference. See "
            "the tabs below for the full picture."
        )

    lift_col, diff_col = st.columns(2)
    lift_col.metric("Order value lift", f"{effect.lift_pct:+.1f}%")
    diff_col.metric("Per-order difference", f"\\${effect.diff:+.2f}")
    st.caption(
        f"{effect.confidence:.0%} bootstrap confidence interval for the "
        f"dollar difference: \\${effect.ci_low:+.2f} to \\${effect.ci_high:+.2f}."
    )

    st.divider()

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
            "An A/A test compares two samples that saw the *same*, "
            "unchanged interface. If a test finds a difference between "
            "them, that normally means the sample size was miscalculated, "
            "since there should be nothing there to detect."
        )
        st.plotly_chart(plots.aa_samples_comparison(aa_test, theme=theme), width="stretch")
        render_raw_data(aa_test, "aa_test.csv")
        render_mean_comparison(comparison)
        verdict = stats.sample_size_verdict(comparison)
        (st.warning if comparison.means_differ else st.success)(verdict)

        if aa_offset is not None:
            st.info(
                f"Worth being direct about why: every row of Sample 2 "
                f"here is exactly Sample 1 plus \\${aa_offset:.2f}, not "
                "sampling noise. The real experiment data used from tab "
                "3 onward carries the same kind of exact, disclosed "
                f"markup in its order value column (\\${ab_offset:.2f} "
                "there), rather than organic variation. The statistical "
                "methods throughout this app are applied correctly and "
                "would flag a genuine problem the same way; here, the "
                "cause is known and mechanical rather than unknown."
            )

    with tabs[1]:
        st.header("Sample Size Estimation")
        st.markdown(
            "Regardless of what caused tab 1's result, it is worth "
            "confirming the real experiment was adequately powered "
            "before trusting its conclusion. This calculates the "
            "minimum sample size needed to reliably detect a small "
            "effect (Cohen's d = 0.2), 80% power and α = 0.05, then "
            "checks whether the actual experiment collected enough "
            "sessions to clear that bar."
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
        st.markdown(
            "A few sessions don't reflect typical behavior: someone "
            "leaves an item mid checkout after a phone call, or orders "
            "for ten friends at once. Both would skew a plain "
            "comparison, so before testing the hypothesis, the top 1% "
            "of order value and session duration is set aside. Every "
            "session below is one point; the dashed lines mark that "
            "cutoff, so anything beyond either line is what gets "
            "trimmed. Order value itself carries the disclosed markup "
            "between groups covered in tab 1."
        )
        st.plotly_chart(plots.session_scatter(ab_test, theme=theme), width="stretch")
        st.subheader("Order value, with the top 1% trimmed")
        summary = stats.summarize_order_value(cleaned)
        mean_col, std_col, max_col = st.columns(3)
        mean_col.metric("Mean", f"{summary.mean:.2f}")
        std_col.metric("Standard deviation", f"{summary.std:.2f}")
        max_col.metric("Max", f"{summary.max:.2f}")
        render_raw_data(ab_test, "ab_test.csv")

    with tabs[3]:
        st.header("Non-parametric Test")
        st.markdown(
            "This is the actual test of the hypothesis: does order "
            "value differ between the two groups? Order values are "
            "skewed, a few big orders, mostly small ones, so rather "
            "than comparing raw averages, this compares the full shape "
            "of the two groups' distributions with a **Mann-Whitney U "
            "test**."
        )
        render_test(mann_whitney)

    with tabs[4]:
        st.header("Parametric Test")
        st.markdown(
            "One test isn't enough on its own. This asks the same "
            "question a second, independent way: **log-transforming** "
            "order values to make a classic t-test valid, then checks "
            "whether both methods agree. This tests whether *geometric* "
            "means differ; the raw average order value (AOV) is "
            "reported separately below."
        )
        st.plotly_chart(plots.log_order_value_histogram(cleaned, theme=theme), width="stretch")
        render_mean_comparison(aov.comparison)

        st.subheader("Average order value (AOV)")
        if aov.comparison.means_differ:
            control_col, exp_col, d_col = st.columns(3)
            control_col.metric("Control AOV", f"\\${aov.control_aov:.2f}")
            exp_col.metric(
                "Experimental AOV",
                f"\\${aov.experimental_aov:.2f}",
                delta=f"\\${aov.experimental_aov - aov.control_aov:+.2f}",
            )
            d_col.metric("Effect size (Cohen's d)", f"{effect.cohens_d:.2f}")
            st.caption("Cohen's d, by convention: ~0.2 small, ~0.5 medium, ~0.8 large.")
            st.success(aov.interpretation)
        else:
            st.info(aov.interpretation)

    st.divider()
    st.subheader("What the pipeline found")
    if both_significant:
        st.markdown(
            f"Both tests agree: order value {direction} by "
            f"**{effect.lift_pct:+.1f}%** between the two groups in this "
            "data, detected with strong statistical confidence. Given "
            "the disclosed markup covered in tab 1, this is the "
            "pipeline correctly recovering a known, built-in difference, "
            "not proof of a real business effect. Run against real "
            "production data instead of this teaching dataset, the same "
            "five stages would be the right way to reach a trustworthy "
            "answer."
        )
    else:
        st.markdown(
            "The two tests don't fully agree on a significant "
            "difference. More data or a longer experiment window may be "
            "needed before drawing a conclusion."
        )


if __name__ == "__main__":
    main()
