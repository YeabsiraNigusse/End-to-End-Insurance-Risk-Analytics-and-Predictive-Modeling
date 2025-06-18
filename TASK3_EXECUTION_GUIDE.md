# Task 3: Statistical Hypothesis Testing - Execution Guide

## Overview
This guide provides step-by-step instructions for executing Task 3: Statistical Hypothesis Testing for Insurance Risk Analytics.

## Prerequisites

### Required Python Packages
```bash
pip install pandas numpy scipy matplotlib seaborn
```

### Data Requirements
- Insurance data file: `data/MachineLearningRating_v3.txt` (pipe-separated)
- If real data is not available, the system will automatically generate sample data

## Execution Methods

### Method 1: Command Line Execution (Recommended)

```bash
# Navigate to the src directory
cd src

# Run the complete analysis
python run_hypothesis_tests.py
```

**Output:**
- Console output with analysis progress and summary
- `results/hypothesis_testing_results.csv` - Detailed statistical results
- `results/hypothesis_testing_summary.txt` - Comprehensive summary report

### Method 2: Jupyter Notebook (Interactive Analysis)

```bash
# Start Jupyter notebook
jupyter notebook

# Open the analysis notebook
# Navigate to: notebook/Task3_Hypothesis_Testing.ipynb
```

**Features:**
- Interactive data exploration
- Step-by-step analysis with visualizations
- Detailed explanations of each hypothesis test
- Customizable parameters and sample sizes

### Method 3: Python Module Import

```python
from src.hypothesis_testing import InsuranceHypothesisTester
import pandas as pd

# Load your data
df = pd.read_csv("data/MachineLearningRating_v3.txt", sep="|")

# Initialize and run analysis
tester = InsuranceHypothesisTester(df=df, alpha=0.05)
results = tester.run_all_hypothesis_tests()

# Access results
print(results['summary'])
```

## Testing the Implementation

### Run Unit Tests
```bash
cd tests
python test_hypothesis_testing.py
```

**Expected Output:**
- 15 tests covering all major functionality
- All tests should pass
- Coverage includes data preprocessing, statistical tests, and hypothesis testing

### Test Individual Components

```python
# Test data preprocessing
from src.data_preprocessing import InsuranceDataProcessor
processor = InsuranceDataProcessor(df=your_data)
cleaned_data = processor.prepare_hypothesis_data()

# Test statistical utilities
from src.statistical_utils import StatisticalTester
tester = StatisticalTester(alpha=0.05)
result = tester.chi_square_test(group_col, outcome_col)
```

## Understanding the Results

### Hypothesis Testing Results

The analysis tests four key hypotheses:

1. **H₁: Province Risk Differences**
   - Tests: Chi-square (claim frequency), Kruskal-Wallis (severity, loss ratio)
   - Interpretation: Significant differences suggest province-specific pricing

2. **H₂: Zip Code Risk Differences**
   - Tests: Chi-square (claim frequency), Kruskal-Wallis (severity, loss ratio)
   - Interpretation: No significant differences suggest zip code pricing unnecessary

3. **H₃: Zip Code Margin Differences**
   - Tests: Kruskal-Wallis (margin)
   - Interpretation: Margin consistency across zip codes

4. **H₄: Gender Risk Differences**
   - Tests: Chi-square (claim frequency), Mann-Whitney U (severity, loss ratio, margin)
   - Interpretation: Gender-based risk factors (subject to regulatory compliance)

### Key Metrics Explained

- **Claim Frequency**: Proportion of policies with claims (binary: 0 or 1)
- **Claim Severity**: Average claim amount for policies with claims
- **Margin**: TotalPremium - TotalClaims (profitability measure)
- **Loss Ratio**: TotalClaims / TotalPremium (risk measure)

### Statistical Significance

- **p-value < 0.05**: Reject null hypothesis (significant difference found)
- **p-value ≥ 0.05**: Fail to reject null hypothesis (no significant difference)
- **Effect Size**: Magnitude of the difference (practical significance)

## File Structure

```
├── src/
│   ├── data_preprocessing.py      # Data cleaning and risk metrics
│   ├── statistical_utils.py      # Statistical test implementations
│   ├── hypothesis_testing.py     # Main hypothesis testing logic
│   └── run_hypothesis_tests.py   # Standalone execution script
├── notebook/
│   └── Task3_Hypothesis_Testing.ipynb  # Interactive analysis
├── tests/
│   └── test_hypothesis_testing.py      # Unit tests
├── results/
│   ├── hypothesis_testing_results.csv  # Detailed results
│   ├── hypothesis_testing_summary.txt  # Summary report
│   └── business_interpretation.md      # Business recommendations
└── TASK3_EXECUTION_GUIDE.md           # This guide
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install required packages
   ```bash
   pip install pandas numpy scipy matplotlib seaborn
   ```

2. **Data file not found**: The system will create sample data automatically
   - Place real data at `data/MachineLearningRating_v3.txt`
   - Ensure pipe-separated format

3. **Memory issues with large datasets**: 
   - The system automatically samples large datasets (>50,000 records)
   - Adjust sample size in `run_hypothesis_tests.py` if needed

4. **Import errors**: Ensure you're running from the correct directory
   ```bash
   # For standalone script
   cd src && python run_hypothesis_tests.py
   
   # For tests
   cd tests && python test_hypothesis_testing.py
   ```

### Performance Optimization

- **Large datasets**: Automatic sampling to 50,000 records
- **Statistical tests**: Automatic selection of parametric vs non-parametric tests
- **Memory management**: Efficient data processing with pandas

## Business Application

### Immediate Actions
1. Review statistical results in `results/hypothesis_testing_summary.txt`
2. Read business recommendations in `results/business_interpretation.md`
3. Validate findings with domain experts

### Implementation Considerations
1. **Regulatory Compliance**: Ensure gender-based pricing complies with local laws
2. **System Integration**: Plan integration with existing pricing systems
3. **Monitoring**: Establish ongoing validation of risk factors

## Next Steps

1. **Validation**: Test with additional datasets and time periods
2. **Refinement**: Adjust statistical methods based on business requirements
3. **Integration**: Incorporate findings into pricing models
4. **Monitoring**: Establish regular hypothesis testing cycles

## Support

For technical issues or questions:
1. Check the unit tests for expected behavior
2. Review the business interpretation document for context
3. Examine the detailed statistical results for specific findings

---

**Note**: This analysis provides statistical evidence for business decisions but should be combined with domain expertise, regulatory requirements, and business strategy considerations.
