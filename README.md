# A/B Test Analysis for Delivery App

## Overview

This project simulates a real-world A/B testing workflow for a food delivery application.

The goal is to increase average order value (AOV) by deploying a new web interface based on user feedback.  
The analysis evaluates whether the new interface leads to a statistically significant change in AOV using hypothesis testing.

## Project Structure

The analysis follows a complete experimental pipeline:

1. **Sample Size Verification (A/A Test)**

   * Context: Two samples were drawn before deploying the new interface  
   * Goal: Verify that the sample size is sufficient for A/B testing  
   * Methods:
     * Levene’s test for variance equality  
     * Independent t-test for mean comparison  
   * Result: The test indicates unequal means, suggesting insufficient sample size  

2. **Sample Size Estimation**

   * Goal: Estimate the required sample size for reliable testing  
   * Method: Power analysis  
   * Result: Required sample size is calculated and compared with available data  

3. **Exploratory Data Analysis (EDA)**

   * Goal: Understand user behavior and data distribution  
   * Methods:
     * Session trends over time (seasonality check)  
     * Distribution analysis  
     * Outlier detection and removal (99th percentile)  

4. **Non-parametric Hypothesis Testing**

   * Goal: Compare AOV distributions between control and experimental groups  
   * Method: Mann-Whitney U test (no normality assumption)  
   * Result: The test suggests a difference in distributions between groups  

5. **Parametric Validation**

   * Goal: Compare average order value between groups using a parametric approach  
   * Method:
     * Log transformation of skewed data  
     * Levene’s test for variance equality  
     * Independent t-test on transformed data  
   * Results:
     * Evidence suggests a difference in AOV between groups  
     * Higher AOV observed in the experimental group  
   * Interpretation: The experimental interface is associated with higher AOV, suggesting a positive impact on user spending  

   To validate the findings, a parametric test is performed after transforming the data to better approximate normality.

## Technologies Used

* Python  
* pandas  
* NumPy  
* SciPy  
* statsmodels  
* Plotly  

## How to Run

Create a virtual environment and install the dependencies (developed with Python 3.13): `python -m venv .venv`, activate it, then `pip install -r requirements.txt`. Then run the script:

```bash
python main.py
```

Ensure that both datasets (`aa_test.csv` and `ab_test.csv`) are located in the same directory as the script.

All five stages will execute sequentially and display results. Charts open in your browser and the script pauses between stages; use `python main.py --no-plots` for a text-only run.

## Notes

This project was originally completed as part of the Hyperskill Data Analyst track.

The solution was later refactored to improve code structure, readability, and clarity of statistical interpretation.
