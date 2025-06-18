"""
Comprehensive Exploratory Data Analysis for Insurance Risk Analytics
Author: Insurance Analytics Team
Date: 2025-06-18

This module provides comprehensive EDA functionality for insurance data analysis,
focusing on risk assessment and profitability patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')

class InsuranceEDA:
    """
    Comprehensive EDA class for insurance risk analytics
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the EDA class
        
        Args:
            data_path: Path to the insurance dataset
        """
        self.data_path = data_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.date_cols = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data preparation
        
        Returns:
            Loaded DataFrame
        """
        try:
            if self.data_path and os.path.exists(self.data_path):
                print(f"Loading data from {self.data_path}")
                self.df = pd.read_csv(self.data_path, sep="|", low_memory=False)
                print(f"Data loaded successfully. Shape: {self.df.shape}")
            else:
                print("Data file not found. Creating sample data for demonstration...")
                self.df = self._create_sample_data()
                
            self._identify_column_types()
            self._initial_data_preparation()
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample data for demonstration...")
            self.df = self._create_sample_data()
            self._identify_column_types()
            self._initial_data_preparation()
            return self.df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample insurance data for demonstration purposes
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Generate sample data
        provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape', 'Free State']
        vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Truck', 'Motorcycle']
        genders = ['Male', 'Female', 'Not Specified']
        makes = ['Toyota', 'BMW', 'Mercedes', 'Volkswagen', 'Ford', 'Nissan']
        
        data = {
            'PolicyID': range(1, n_samples + 1),
            'TransactionMonth': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'Province': np.random.choice(provinces, n_samples),
            'VehicleType': np.random.choice(vehicle_types, n_samples),
            'Gender': np.random.choice(genders, n_samples, p=[0.45, 0.45, 0.1]),
            'Make': np.random.choice(makes, n_samples),
            'TotalPremium': np.random.lognormal(3, 1, n_samples),
            'TotalClaims': np.random.lognormal(2, 1.5, n_samples) * np.random.binomial(1, 0.3, n_samples),
            'CustomValueEstimate': np.random.lognormal(4, 0.8, n_samples),
            'PostalCode': np.random.randint(1000, 9999, n_samples),
            'SumInsured': np.random.lognormal(4.5, 0.7, n_samples),
            'RegistrationYear': np.random.randint(2010, 2024, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic relationships
        df.loc[df['VehicleType'] == 'SUV', 'TotalPremium'] *= 1.3
        df.loc[df['VehicleType'] == 'Motorcycle', 'TotalPremium'] *= 0.7
        df.loc[df['Province'] == 'Gauteng', 'TotalClaims'] *= 1.2
        
        return df
    
    def _identify_column_types(self):
        """Identify and categorize column types"""
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to convert date columns
        for col in self.df.columns:
            if 'date' in col.lower() or 'month' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    if col not in self.date_cols:
                        self.date_cols.append(col)
                        if col in self.categorical_cols:
                            self.categorical_cols.remove(col)
                except:
                    pass
    
    def _initial_data_preparation(self):
        """Perform initial data preparation"""
        # Convert numeric columns
        for col in ['TotalClaims', 'TotalPremium']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Calculate Loss Ratio
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = np.where(
                self.df['TotalPremium'] != 0,
                self.df['TotalClaims'] / self.df['TotalPremium'],
                0
            )
            self.numerical_cols.append('LossRatio')
    
    def data_overview(self) -> Dict:
        """
        Provide comprehensive data overview
        
        Returns:
            Dictionary with data overview statistics
        """
        overview = {
            'shape': self.df.shape,
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'numerical_columns': len(self.numerical_cols),
            'categorical_columns': len(self.categorical_cols),
            'date_columns': len(self.date_cols),
            'missing_values_total': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        print("=== DATA OVERVIEW ===")
        for key, value in overview.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return overview
    
    def missing_value_analysis(self) -> pd.DataFrame:
        """
        Comprehensive missing value analysis
        
        Returns:
            DataFrame with missing value statistics
        """
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data_Type': self.df.dtypes
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("\n=== MISSING VALUE ANALYSIS ===")
        print(missing_stats[missing_stats['Missing_Count'] > 0])
        
        return missing_stats
    
    def descriptive_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive descriptive statistics
        
        Returns:
            Dictionary with numerical and categorical statistics
        """
        stats_dict = {}
        
        # Numerical statistics
        if self.numerical_cols:
            numerical_stats = self.df[self.numerical_cols].describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                col: {
                    'variance': self.df[col].var(),
                    'std_dev': self.df[col].std(),
                    'skewness': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis(),
                    'cv': self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else 0
                } for col in self.numerical_cols
            }).T
            
            stats_dict['numerical'] = pd.concat([numerical_stats.T, additional_stats], axis=1)
        
        # Categorical statistics
        if self.categorical_cols:
            cat_stats = pd.DataFrame({
                col: {
                    'unique_count': self.df[col].nunique(),
                    'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                    'most_frequent_count': self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0,
                    'most_frequent_percentage': (self.df[col].value_counts().iloc[0] / len(self.df)) * 100 if len(self.df[col].value_counts()) > 0 else 0
                } for col in self.categorical_cols
            }).T
            
            stats_dict['categorical'] = cat_stats
        
        print("\n=== DESCRIPTIVE STATISTICS ===")
        if 'numerical' in stats_dict:
            print("\nNumerical Variables:")
            print(stats_dict['numerical'].round(4))
        
        if 'categorical' in stats_dict:
            print("\nCategorical Variables:")
            print(stats_dict['categorical'])
        
        return stats_dict

    def loss_ratio_analysis(self) -> Dict:
        """
        Comprehensive Loss Ratio Analysis

        Returns:
            Dictionary with loss ratio insights
        """
        if 'LossRatio' not in self.df.columns:
            print("Loss Ratio not available in dataset")
            return {}

        analysis = {}

        # Overall Loss Ratio
        overall_loss_ratio = self.df['LossRatio'].mean()
        analysis['overall_loss_ratio'] = overall_loss_ratio

        print(f"\n=== LOSS RATIO ANALYSIS ===")
        print(f"Overall Portfolio Loss Ratio: {overall_loss_ratio:.4f}")

        # Loss Ratio by Province
        if 'Province' in self.df.columns:
            province_loss = self.df.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            province_loss['LossRatio'] = province_loss['TotalClaims'] / province_loss['TotalPremium']
            province_loss = province_loss.sort_values('LossRatio', ascending=False)
            analysis['province_loss_ratio'] = province_loss

            print(f"\nLoss Ratio by Province:")
            print(province_loss['LossRatio'].round(4))

        # Loss Ratio by Vehicle Type
        if 'VehicleType' in self.df.columns:
            vehicle_loss = self.df.groupby('VehicleType').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            vehicle_loss['LossRatio'] = vehicle_loss['TotalClaims'] / vehicle_loss['TotalPremium']
            vehicle_loss = vehicle_loss.sort_values('LossRatio', ascending=False)
            analysis['vehicle_loss_ratio'] = vehicle_loss

            print(f"\nLoss Ratio by Vehicle Type:")
            print(vehicle_loss['LossRatio'].round(4))

        # Loss Ratio by Gender
        if 'Gender' in self.df.columns:
            gender_loss = self.df.groupby('Gender').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            gender_loss['LossRatio'] = gender_loss['TotalClaims'] / gender_loss['TotalPremium']
            gender_loss = gender_loss.sort_values('LossRatio', ascending=False)
            analysis['gender_loss_ratio'] = gender_loss

            print(f"\nLoss Ratio by Gender:")
            print(gender_loss['LossRatio'].round(4))

        return analysis

    def outlier_detection(self) -> Dict:
        """
        Detect outliers using multiple methods

        Returns:
            Dictionary with outlier information
        """
        outlier_info = {}

        print(f"\n=== OUTLIER DETECTION ===")

        for col in ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']:
            if col in self.df.columns:
                # IQR Method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_iqr = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

                # Z-Score Method
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers_zscore = self.df[z_scores > 3]

                outlier_info[col] = {
                    'iqr_outliers': len(outliers_iqr),
                    'iqr_percentage': (len(outliers_iqr) / len(self.df)) * 100,
                    'zscore_outliers': len(outliers_zscore),
                    'zscore_percentage': (len(outliers_zscore) / len(self.df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

                print(f"\n{col}:")
                print(f"  IQR Outliers: {len(outliers_iqr)} ({(len(outliers_iqr) / len(self.df)) * 100:.2f}%)")
                print(f"  Z-Score Outliers: {len(outliers_zscore)} ({(len(outliers_zscore) / len(self.df)) * 100:.2f}%)")

        return outlier_info

    def temporal_analysis(self) -> Dict:
        """
        Analyze temporal trends in the data

        Returns:
            Dictionary with temporal insights
        """
        temporal_info = {}

        if not self.date_cols:
            print("No date columns found for temporal analysis")
            return temporal_info

        print(f"\n=== TEMPORAL ANALYSIS ===")

        date_col = self.date_cols[0]  # Use first date column

        # Monthly aggregation
        self.df['YearMonth'] = self.df[date_col].dt.to_period('M')

        monthly_stats = self.df.groupby('YearMonth').agg({
            'TotalClaims': ['sum', 'mean', 'count'],
            'TotalPremium': ['sum', 'mean']
        }).round(2)

        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
        temporal_info['monthly_stats'] = monthly_stats

        # Calculate claim frequency and severity
        monthly_stats['ClaimFrequency'] = monthly_stats['TotalClaims_count'] / len(self.df)
        monthly_stats['ClaimSeverity'] = monthly_stats['TotalClaims_sum'] / monthly_stats['TotalClaims_count']

        print("Monthly Trends (First 5 months):")
        print(monthly_stats.head())

        return temporal_info

    def create_univariate_plots(self, save_plots: bool = True):
        """
        Create comprehensive univariate analysis plots

        Args:
            save_plots: Whether to save plots to files
        """
        print(f"\n=== CREATING UNIVARIATE PLOTS ===")

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')

        # Numerical variables distribution
        if self.numerical_cols:
            n_cols = min(3, len(self.numerical_cols))
            n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(self.numerical_cols[:9]):  # Limit to 9 plots
                if i < len(axes):
                    # Histogram with KDE
                    self.df[col].hist(bins=30, alpha=0.7, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')

            # Remove empty subplots
            for i in range(len(self.numerical_cols), len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            if save_plots:
                plt.savefig('plots/univariate_numerical.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Categorical variables
        if self.categorical_cols:
            for col in self.categorical_cols[:4]:  # Limit to 4 categorical plots
                plt.figure(figsize=(12, 6))

                value_counts = self.df[col].value_counts().head(10)  # Top 10 categories

                plt.subplot(1, 2, 1)
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                plt.subplot(1, 2, 2)
                value_counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Proportion of {col}')
                plt.ylabel('')

                plt.tight_layout()
                if save_plots:
                    plt.savefig(f'plots/univariate_{col.lower()}.png', dpi=300, bbox_inches='tight')
                plt.show()

    def create_bivariate_plots(self, save_plots: bool = True):
        """
        Create bivariate analysis plots

        Args:
            save_plots: Whether to save plots to files
        """
        print(f"\n=== CREATING BIVARIATE PLOTS ===")

        # Correlation heatmap
        if len(self.numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[self.numerical_cols].corr()

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Variables')
            plt.tight_layout()
            if save_plots:
                plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Scatter plots for key relationships
        key_pairs = [
            ('TotalPremium', 'TotalClaims'),
            ('TotalPremium', 'LossRatio'),
            ('CustomValueEstimate', 'TotalPremium')
        ]

        for x_col, y_col in key_pairs:
            if x_col in self.df.columns and y_col in self.df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} vs {x_col}')

                # Add trend line
                z = np.polyfit(self.df[x_col].dropna(), self.df[y_col].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8)

                plt.tight_layout()
                if save_plots:
                    plt.savefig(f'plots/scatter_{x_col}_{y_col}.png', dpi=300, bbox_inches='tight')
                plt.show()

    def create_advanced_visualizations(self, save_plots: bool = True):
        """
        Create three creative and insightful visualizations

        Args:
            save_plots: Whether to save plots to files
        """
        print(f"\n=== CREATING ADVANCED VISUALIZATIONS ===")

        # Create plots directory
        os.makedirs('plots', exist_ok=True)

        # Visualization 1: Interactive Loss Ratio Dashboard
        self._create_loss_ratio_dashboard(save_plots)

        # Visualization 2: Risk Profile Heatmap
        self._create_risk_profile_heatmap(save_plots)

        # Visualization 3: Temporal Claims Analysis
        self._create_temporal_claims_analysis(save_plots)

    def _create_loss_ratio_dashboard(self, save_plots: bool):
        """Create interactive loss ratio dashboard"""
        if 'Province' not in self.df.columns or 'VehicleType' not in self.df.columns:
            return

        # Calculate loss ratios by Province and VehicleType
        pivot_data = self.df.groupby(['Province', 'VehicleType']).agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum'
        }).reset_index()

        pivot_data['LossRatio'] = pivot_data['TotalClaims'] / pivot_data['TotalPremium']

        # Create interactive heatmap
        fig = px.density_heatmap(
            pivot_data,
            x='Province',
            y='VehicleType',
            z='LossRatio',
            title='Loss Ratio Heatmap: Province vs Vehicle Type',
            color_continuous_scale='RdYlBu_r'
        )

        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )

        if save_plots:
            fig.write_html('plots/loss_ratio_dashboard.html')
        fig.show()

    def _create_risk_profile_heatmap(self, save_plots: bool):
        """Create risk profile heatmap"""
        # Create risk segments based on claims and premiums
        self.df['PremiumSegment'] = pd.qcut(self.df['TotalPremium'],
                                          q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        self.df['ClaimSegment'] = pd.qcut(self.df['TotalClaims'],
                                        q=4, labels=['Low', 'Medium', 'High', 'Very High'])

        # Create cross-tabulation
        risk_profile = pd.crosstab(self.df['PremiumSegment'], self.df['ClaimSegment'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(risk_profile, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Risk Profile Matrix: Premium vs Claims Segments')
        plt.xlabel('Claims Segment')
        plt.ylabel('Premium Segment')
        plt.tight_layout()

        if save_plots:
            plt.savefig('plots/risk_profile_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_temporal_claims_analysis(self, save_plots: bool):
        """Create temporal claims analysis"""
        if not self.date_cols:
            return

        date_col = self.date_cols[0]

        # Monthly aggregation
        monthly_data = self.df.groupby(self.df[date_col].dt.to_period('M')).agg({
            'TotalClaims': ['sum', 'count'],
            'TotalPremium': 'sum'
        })

        monthly_data.columns = ['Total_Claims', 'Claim_Count', 'Total_Premium']
        monthly_data['Loss_Ratio'] = monthly_data['Total_Claims'] / monthly_data['Total_Premium']
        monthly_data.index = monthly_data.index.to_timestamp()

        # Create subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Claims trend
        ax1.plot(monthly_data.index, monthly_data['Total_Claims'], marker='o')
        ax1.set_title('Monthly Total Claims Trend')
        ax1.set_ylabel('Total Claims')
        ax1.tick_params(axis='x', rotation=45)

        # Premium trend
        ax2.plot(monthly_data.index, monthly_data['Total_Premium'], marker='s', color='green')
        ax2.set_title('Monthly Total Premium Trend')
        ax2.set_ylabel('Total Premium')
        ax2.tick_params(axis='x', rotation=45)

        # Loss ratio trend
        ax3.plot(monthly_data.index, monthly_data['Loss_Ratio'], marker='^', color='red')
        ax3.set_title('Monthly Loss Ratio Trend')
        ax3.set_ylabel('Loss Ratio')
        ax3.tick_params(axis='x', rotation=45)

        # Claim frequency
        ax4.bar(monthly_data.index, monthly_data['Claim_Count'], alpha=0.7, color='orange')
        ax4.set_title('Monthly Claim Frequency')
        ax4.set_ylabel('Number of Claims')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/temporal_claims_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def statistical_testing(self) -> Dict:
        """
        Perform statistical tests for key hypotheses

        Returns:
            Dictionary with test results
        """
        test_results = {}

        print(f"\n=== STATISTICAL TESTING ===")

        # Test 1: Gender difference in loss ratios
        if 'Gender' in self.df.columns and 'LossRatio' in self.df.columns:
            male_loss = self.df[self.df['Gender'] == 'Male']['LossRatio'].dropna()
            female_loss = self.df[self.df['Gender'] == 'Female']['LossRatio'].dropna()

            if len(male_loss) > 0 and len(female_loss) > 0:
                t_stat, p_value = stats.ttest_ind(male_loss, female_loss)
                test_results['gender_loss_ratio_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                print(f"Gender Loss Ratio T-Test:")
                print(f"  T-statistic: {t_stat:.4f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

        # Test 2: Province difference in claims
        if 'Province' in self.df.columns and 'TotalClaims' in self.df.columns:
            province_groups = [group['TotalClaims'].dropna() for name, group in self.df.groupby('Province')]

            if len(province_groups) > 2:
                f_stat, p_value = stats.f_oneway(*province_groups)
                test_results['province_claims_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                print(f"\nProvince Claims ANOVA:")
                print(f"  F-statistic: {f_stat:.4f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

        return test_results

    def generate_insights(self) -> List[str]:
        """
        Generate actionable insights from the analysis

        Returns:
            List of key insights
        """
        insights = []

        print(f"\n=== KEY INSIGHTS ===")

        # Overall loss ratio insight
        if 'LossRatio' in self.df.columns:
            overall_loss_ratio = self.df['LossRatio'].mean()
            if overall_loss_ratio > 1:
                insights.append(f"⚠️  Portfolio is unprofitable with loss ratio of {overall_loss_ratio:.3f}")
            else:
                insights.append(f"✅ Portfolio is profitable with loss ratio of {overall_loss_ratio:.3f}")

        # Province insights
        if 'Province' in self.df.columns:
            province_claims = self.df.groupby('Province')['TotalClaims'].mean().sort_values(ascending=False)
            highest_risk_province = province_claims.index[0]
            insights.append(f"🏢 {highest_risk_province} shows highest average claims")

        # Vehicle type insights
        if 'VehicleType' in self.df.columns:
            vehicle_loss = self.df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False)
            if len(vehicle_loss) > 0:
                riskiest_vehicle = vehicle_loss.index[0]
                insights.append(f"🚗 {riskiest_vehicle} vehicles have highest loss ratio")

        # Outlier insights
        if 'TotalClaims' in self.df.columns:
            Q3 = self.df['TotalClaims'].quantile(0.75)
            Q1 = self.df['TotalClaims'].quantile(0.25)
            IQR = Q3 - Q1
            outliers = self.df[self.df['TotalClaims'] > Q3 + 1.5 * IQR]
            outlier_percentage = (len(outliers) / len(self.df)) * 100

            if outlier_percentage > 5:
                insights.append(f"📊 High outlier percentage ({outlier_percentage:.1f}%) suggests need for claims investigation")

        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")

        return insights

    def run_complete_analysis(self) -> Dict:
        """
        Run the complete EDA analysis pipeline

        Returns:
            Dictionary with all analysis results
        """
        print("🚀 Starting Comprehensive Insurance EDA Analysis...")

        # Load data
        self.load_data()

        # Create plots directory
        os.makedirs('plots', exist_ok=True)

        # Run all analyses
        results = {
            'data_overview': self.data_overview(),
            'missing_values': self.missing_value_analysis(),
            'descriptive_stats': self.descriptive_statistics(),
            'loss_ratio_analysis': self.loss_ratio_analysis(),
            'outlier_detection': self.outlier_detection(),
            'temporal_analysis': self.temporal_analysis(),
            'statistical_tests': self.statistical_testing(),
            'insights': self.generate_insights()
        }

        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_univariate_plots()
        self.create_bivariate_plots()
        self.create_advanced_visualizations()

        print("\n✅ Analysis Complete! Check the 'plots' directory for visualizations.")

        return results


def main():
    """
    Main function to run the EDA analysis
    """
    # Initialize EDA class
    eda = InsuranceEDA(data_path="../data/MachineLearningRating_v3.txt")

    # Run complete analysis
    results = eda.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
