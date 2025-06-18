# End-to-End Insurance Risk Analytics and Predictive Modeling

## Project Overview

This project implements a comprehensive insurance risk analytics and predictive modeling system. The project is structured in multiple tasks, each building upon the previous work to create a complete end-to-end solution.

## Task 3: Statistical Hypothesis Testing for Risk Segmentation

### Objective
Statistically validate or reject key hypotheses about risk drivers to form the basis of our new segmentation strategy through A/B hypothesis testing.

### Hypotheses Tested
1. **H₀**: There are no risk differences across provinces
2. **H₀**: There are no risk differences between zip codes
3. **H₀**: There are no significant margin (profit) differences between zip codes
4. **H₀**: There are no significant risk differences between Women and Men

### Key Metrics
- **Risk Quantification**:
  - **Claim Frequency**: Proportion of policies with at least one claim
  - **Claim Severity**: Average amount of a claim, given a claim occurred
- **Margin**: TotalPremium - TotalClaims

### Statistical Methods Used
- **Chi-Square Test**: For categorical variables (claim frequency differences)
- **Two-Sample T-Test**: For comparing means between two groups
- **Mann-Whitney U Test**: Non-parametric alternative for two-group comparisons
- **One-Way ANOVA**: For comparing means across multiple groups
- **Kruskal-Wallis Test**: Non-parametric alternative for multi-group comparisons

### Results Summary
Based on our analysis with sample data:

#### Rejected Hypotheses (Significant Differences Found):
1. **Province Risk Differences**: Significant differences found in claim frequency, claim severity, and loss ratio across provinces (p < 0.001)
2. **Gender Risk Differences**: Significant differences found in margin between genders (p < 0.001)

#### Failed to Reject (No Significant Differences):
1. **Zip Code Risk Differences**: No significant differences in risk metrics between zip codes
2. **Zip Code Margin Differences**: No significant differences in profit margins between zip codes

### Business Recommendations
1. **Implement province-specific pricing strategies** due to significant risk variations across provinces
2. **Consider gender-based risk factors** in underwriting (subject to regulatory compliance and fairness considerations)

## Project Structure


A starter repo to get everyone up and running with Python, Git, and GitHub Actions CI so you can focus on the challenge!

##  Quickstart

1. ## Fork and  clone

   ```bash
   git clone https://github.com/YeabsiraNigusse/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling
   cd End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling
   ```

2. ## Create and activate virtual environment

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate      # macOS/Linux
    # .venv\Scripts\activate       # Windows PowerShell
    ```

3. ## Install dependencies

    ```bash
    pip install -r requirements.txt
    ```
4. ## Branching and Workflow

- Branch off main for any work:
    ```bash
    git add .
    git commit -m "feat: describe your change"
    ```
5. ## Push and open a PR against main

    ```bash
    git push -u origin feature/your-feature
    ```

