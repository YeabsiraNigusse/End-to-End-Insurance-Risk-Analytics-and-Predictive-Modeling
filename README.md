# End-to-End Insurance Risk Analytics and Predictive Modeling

## 🎯 Project Overview

This project develops a comprehensive insurance risk analytics and predictive modeling system to optimize insurance pricing, improve risk assessment, and enhance profitability through data-driven insights.

### 🏆 Current Phase: Task 1.2 - EDA & Statistical Analysis

**Objective:** Develop foundational understanding of insurance data, assess quality, and uncover initial patterns in risk and profitability.

## 📊 Key Business Questions Addressed

1. **Loss Ratio Analysis**: What is the overall Loss Ratio (TotalClaims / TotalPremium) for the portfolio?
2. **Risk Segmentation**: How does Loss Ratio vary by Province, VehicleType, and Gender?
3. **Data Quality**: What are the distributions of key financial variables and outliers?
4. **Temporal Trends**: Are there temporal patterns in claim frequency or severity?
5. **Vehicle Risk**: Which vehicle makes/models are associated with highest/lowest claims?

## 🛠️ Technical Stack

- **Python 3.8+**
- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly, bokeh
- **Statistical Analysis**: statsmodels
- **Jupyter Notebooks** for interactive analysis
- **DVC** for data version control

## 📁 Project Structure

```
├── data/                          # Data files (managed by DVC)
│   └── MachineLearningRating_v3.txt.dvc
├── notebook/                      # Jupyter notebooks
│   └── EDA.ipynb                 # Comprehensive EDA analysis
├── src/                          # Source code
│   ├── __init__.py
│   └── eda_analysis.py           # EDA analysis class
├── tests/                        # Unit tests
│   └── __init__.py
├── plots/                        # Generated visualizations
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # License file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YeabsiraNigusse/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling.git
   cd End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up data (if available)**
   ```bash
   dvc pull  # If DVC is configured
   ```

## 📈 Analysis Features

### 🔍 Comprehensive EDA Class (`InsuranceEDA`)

The `InsuranceEDA` class provides:

- **Data Loading & Preparation**: Automatic data type detection and conversion
- **Quality Assessment**: Missing value analysis and data validation
- **Descriptive Statistics**: Comprehensive statistical summaries
- **Loss Ratio Analysis**: Core insurance profitability metrics
- **Outlier Detection**: Multiple methods (IQR, Z-score)
- **Temporal Analysis**: Time-series trend identification
- **Statistical Testing**: Hypothesis testing for group differences
- **Advanced Visualizations**: Three creative, insightful plots

### 📊 Key Visualizations

1. **Interactive Loss Ratio Dashboard**: Province vs Vehicle Type heatmap
2. **Risk Profile Matrix**: Premium vs Claims segmentation
3. **Temporal Claims Analysis**: Multi-panel time series analysis

## 🎯 Key Performance Indicators (KPIs)

- **Overall Loss Ratio**: Portfolio profitability metric
- **Segmented Loss Ratios**: By Province, Vehicle Type, Gender
- **Claim Frequency**: Number of claims per period
- **Claim Severity**: Average claim amount
- **Risk Concentration**: Geographic and demographic risk distribution

## 📋 Usage Examples

### Quick Start with Jupyter Notebook

```bash
jupyter notebook notebook/EDA.ipynb
```

### Programmatic Analysis

```python
from src.eda_analysis import InsuranceEDA

# Initialize EDA
eda = InsuranceEDA(data_path="data/MachineLearningRating_v3.txt")

# Run complete analysis
results = eda.run_complete_analysis()

# Access specific analyses
loss_analysis = eda.loss_ratio_analysis()
outliers = eda.outlier_detection()
insights = eda.generate_insights()
```

## 📊 Sample Insights Generated

- ✅ Portfolio profitability assessment
- 🏢 Geographic risk concentration identification
- 🚗 Vehicle type risk ranking
- 📈 Temporal trend analysis
- 🎯 Actionable recommendations for pricing and underwriting

## 🔬 Statistical Methods Applied

- **Descriptive Statistics**: Mean, median, variance, skewness, kurtosis
- **Hypothesis Testing**: T-tests, ANOVA for group comparisons
- **Outlier Detection**: IQR method, Z-score analysis
- **Correlation Analysis**: Pearson correlation matrices
- **Distribution Analysis**: Statistical distribution fitting

## 📈 Visualization Capabilities

- **Univariate Analysis**: Histograms, box plots, bar charts
- **Bivariate Analysis**: Scatter plots, correlation heatmaps
- **Advanced Plots**: Interactive dashboards, risk matrices
- **Temporal Analysis**: Time series plots, trend analysis

## 🎯 Business Value Delivered

1. **Risk Assessment**: Identify high-risk segments and geographies
2. **Pricing Optimization**: Data-driven insights for premium setting
3. **Profitability Analysis**: Loss ratio monitoring and improvement
4. **Operational Efficiency**: Automated analysis and reporting
5. **Strategic Planning**: Evidence-based decision making

## 🚀 Next Steps (Upcoming Tasks)

- [ ] **Feature Engineering**: Advanced feature creation and selection
- [ ] **Predictive Modeling**: Machine learning model development
- [ ] **Model Validation**: Cross-validation and performance metrics
- [ ] **Deployment**: Production-ready model deployment
- [ ] **Monitoring**: Model performance tracking and maintenance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Lead Data Scientist**: Insurance Analytics Team
- **Project Type**: End-to-End Insurance Risk Analytics
- **Industry**: Insurance & Financial Services

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the development team.

---

**⭐ Star this repository if you find it helpful!**


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

