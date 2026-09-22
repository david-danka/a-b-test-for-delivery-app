# A/B Test Analysis for Delivery App

**Live demo: https://a-b-test-for-delivery-app.streamlit.app/**

## Overview

This project runs a complete A/B testing workflow end to end: checking an experiment's own design before trusting its conclusion, estimating the sample size a real test would need, exploring and cleaning the data, then testing the hypothesis two independent statistical ways.

The scenario: a food delivery app's menu page was redesigned to surface popular items at the top, on the theory that easier to find items raise the average order value (AOV). The bundled data is a teaching dataset (from the Hyperskill Data Analyst track), not a real business experiment. See "A note on the data" below before treating any result as a real-world finding.

## A note on the data

Both bundled datasets have a disclosed, exact constant markup built into them rather than organic sampling variation. Comparing sorted values shows `aa_test.csv`'s Sample 2 is Sample 1 plus exactly $4.00 on every row, and `ab_test.csv`'s Experimental order value is Control's order value plus exactly $6.00 across the whole distribution. Session duration does not show this pattern.

This means the statistical methods used throughout this project (Levene's test, the t-tests, Mann-Whitney U, the log transform, the bootstrap confidence interval, Cohen's d) are demonstrated correctly and do correctly detect and characterize a real, exact difference in this data. What they cannot support is a causal claim about real user behavior, since the difference was written into the dataset rather than observed. Treat the results below as a demonstration of a correctly executed A/B testing pipeline, not proof that redesigning a menu page increases spending.

This check is implemented in `abtest.stats.detect_constant_offset` and covered by tests in `tests/test_stats.py`.

## Analysis Stages

The analysis follows a complete experimental pipeline. Numbers below are the actual results for the bundled data.

1. **Sample Size Verification (A/A Test)**
   * Two samples drawn from the same, unchanged interface, to check the experiment's own design before trusting anything downstream.
   * Levene's test: W = 0.000, p > 0.05 (variances equal). T-test: t = -3.432, p <= 0.05 (means not equal).
   * A significant result here normally means the sample size was miscalculated. In this data it instead traces to the disclosed markup above.

2. **Sample Size Estimation**
   * Power analysis for a two-sample t-test (effect size 0.2, power 0.8, alpha 0.05): 400 sessions required per group.
   * The real experiment collected 400 Control and 400 Experimental sessions, meeting that bar.

3. **Exploratory Data Analysis**
   * Session counts by day, and the distributions of order value and session duration, plotted per group.
   * The top 1 percent of order value and session duration is trimmed as outliers before testing.
   * Trimmed order value: mean $33.75, standard deviation $32.19, max $202.98.

4. **Non-parametric Test**
   * Order values are skewed, so group distributions are compared with a Mann-Whitney U test rather than a t-test.
   * U1 = 60612.000, p <= 0.05 (about 3.1e-07): the distributions differ.

5. **Parametric Validation**
   * Order values are log-transformed to approximate normality, then compared with a second, independent t-test.
   * Levene's test: W = 30.174, p <= 0.05 (unequal variances, Welch's t-test used). T-test: t = -5.859, p <= 0.05 (means differ).
   * Control AOV $30.58, Experimental AOV $36.90: a difference of $6.31, a 20.6% lift, 95% bootstrap confidence interval $1.71 to $10.72, Cohen's d 0.42.
   * Both tests agree: the pipeline correctly recovers the built-in markup with strong statistical confidence.

## Repository Structure

* `abtest/`, the analysis package
  * `data.py`, loading and cleaning the datasets
  * `stats.py`, every statistical test, returning plain result objects
  * `plots.py`, every chart, returning Plotly figures
* `app.py`, the Streamlit app (the live demo above)
* `main.py`, a command line runner over the same `abtest` package
* `data/`, the bundled datasets, `aa_test.csv` and `ab_test.csv`
* `tests/`, the pytest suite

## Technologies Used

* Python
* pandas
* NumPy
* SciPy
* statsmodels
* Plotly
* Streamlit

## How to Run

Create a virtual environment and install the dependencies (developed with Python 3.13):

```bash
python -m venv .venv
# then activate it
pip install -r requirements.txt
```

Run the interactive app:

```bash
streamlit run app.py
```

Or run the command line version:

```bash
python main.py
```

Charts open in your browser and the script pauses between stages; use `python main.py --no-plots` for a text-only run. Both entry points read `aa_test.csv` and `ab_test.csv` from the `data/` folder automatically.

## Testing

Install the development dependencies and run the suite:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the statistics, the constant-offset check, and the chart structure, and pin the exact numbers above so a future change to the analysis will be caught.

## Notes

This project was originally completed as part of the Hyperskill Data Analyst track. It was later refactored into a tested package backing both a command line script and a Streamlit app, with Plotly replacing the original matplotlib charts.
