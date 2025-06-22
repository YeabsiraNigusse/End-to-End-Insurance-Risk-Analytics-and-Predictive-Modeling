"""
Predictive Modeling for Insurance Risk Analytics
Author: Insurance Analytics Team
Date: 2025-06-18

This module implements comprehensive predictive modeling for:
1. Claim Severity Prediction (Risk Model)
2. Claim Probability Prediction (Binary Classification)
3. Premium Optimization (Pricing Framework)

Key Features:
- Advanced data preprocessing and feature engineering
- Multiple ML algorithms (Linear Regression, Random Forest, XGBoost)
- Comprehensive model evaluation and comparison
- SHAP-based model interpretability
- Risk-based pricing framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
from pathlib import Path

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

# Model Interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

class InsurancePredictiveModeling:
    """
    Comprehensive predictive modeling class for insurance analytics
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the predictive modeling class
        
        Args:
            data_path: Path to the insurance dataset
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Results storage
        self.results = {
            'claim_severity': {},
            'claim_probability': {},
            'premium_optimization': {}
        }
        
        # Feature importance storage
        self.feature_importance = {}
        self.shap_values = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and perform initial data preparation

        Returns:
            Prepared DataFrame
        """
        try:
            if self.data_path and os.path.exists(self.data_path):
                print(f"Loading data from {self.data_path}")
                self.df = pd.read_csv(self.data_path, sep="|", low_memory=False)
                print(f"Data loaded successfully. Shape: {self.df.shape}")

                # Create HasClaim column if it doesn't exist
                self._create_target_variables()

            else:
                if self.data_path:
                    print(f"Data file not found: {self.data_path}")
                    print("This might be a DVC-managed file. Try running 'dvc pull' first.")
                print("Creating sample data for demonstration...")
                self.df = self._create_sample_data()

            return self.df

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample data for demonstration...")
            self.df = self._create_sample_data()
            return self.df

    def _create_target_variables(self):
        """Create target variables for modeling if they don't exist"""

        # Create HasClaim column if it doesn't exist
        if 'HasClaim' not in self.df.columns:
            if 'TotalClaims' in self.df.columns:
                # Convert TotalClaims to numeric, handling any non-numeric values
                self.df['TotalClaims'] = pd.to_numeric(self.df['TotalClaims'], errors='coerce').fillna(0)
                # Create binary HasClaim variable
                self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
                print("✅ Created HasClaim column from TotalClaims")
            else:
                print("⚠️ TotalClaims column not found. Cannot create HasClaim column.")

        # Ensure TotalClaims is numeric
        if 'TotalClaims' in self.df.columns:
            self.df['TotalClaims'] = pd.to_numeric(self.df['TotalClaims'], errors='coerce').fillna(0)

        # Ensure CalculatedPremiumPerTerm is numeric
        if 'CalculatedPremiumPerTerm' in self.df.columns:
            self.df['CalculatedPremiumPerTerm'] = pd.to_numeric(self.df['CalculatedPremiumPerTerm'], errors='coerce')
        elif 'TotalPremium' in self.df.columns:
            # Use TotalPremium as CalculatedPremiumPerTerm if the latter doesn't exist
            self.df['CalculatedPremiumPerTerm'] = pd.to_numeric(self.df['TotalPremium'], errors='coerce')
            print("✅ Created CalculatedPremiumPerTerm from TotalPremium")

        print(f"📊 Target variables summary:")
        if 'HasClaim' in self.df.columns:
            claim_rate = self.df['HasClaim'].mean()
            print(f"  Claim rate: {claim_rate:.3f} ({self.df['HasClaim'].sum():,} claims out of {len(self.df):,} policies)")

        if 'TotalClaims' in self.df.columns:
            total_claims = self.df['TotalClaims'].sum()
            avg_claim = self.df[self.df['TotalClaims'] > 0]['TotalClaims'].mean() if (self.df['TotalClaims'] > 0).any() else 0
            print(f"  Total claims: R{total_claims:,.2f}")
            print(f"  Average claim (when > 0): R{avg_claim:,.2f}")

        if 'CalculatedPremiumPerTerm' in self.df.columns:
            total_premium = self.df['CalculatedPremiumPerTerm'].sum()
            avg_premium = self.df['CalculatedPremiumPerTerm'].mean()
            print(f"  Total premiums: R{total_premium:,.2f}")
            print(f"  Average premium: R{avg_premium:,.2f}")

            if 'TotalClaims' in self.df.columns:
                loss_ratio = total_claims / total_premium if total_premium > 0 else 0
                print(f"  Overall loss ratio: {loss_ratio:.3f}")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample insurance data for demonstration purposes
        """
        np.random.seed(42)
        n_samples = 50000
        
        # Generate sample data with realistic relationships
        provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape', 'Free State', 'Limpopo']
        vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Truck', 'Motorcycle', 'Coupe']
        genders = ['Male', 'Female', 'Not Specified']
        makes = ['Toyota', 'BMW', 'Mercedes', 'Volkswagen', 'Ford', 'Nissan', 'Audi', 'Hyundai']
        cover_types = ['Comprehensive', 'Third Party', 'Third Party Fire & Theft']
        
        data = {
            'PolicyID': range(1, n_samples + 1),
            'TransactionMonth': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'Province': np.random.choice(provinces, n_samples),
            'VehicleType': np.random.choice(vehicle_types, n_samples),
            'Gender': np.random.choice(genders, n_samples, p=[0.45, 0.45, 0.1]),
            'Make': np.random.choice(makes, n_samples),
            'CoverType': np.random.choice(cover_types, n_samples, p=[0.6, 0.25, 0.15]),
            'RegistrationYear': np.random.randint(2010, 2024, n_samples),
            'CustomValueEstimate': np.random.lognormal(4, 0.8, n_samples),
            'SumInsured': np.random.lognormal(4.5, 0.7, n_samples),
            'PostalCode': np.random.randint(1000, 9999, n_samples),
            'CalculatedPremiumPerTerm': np.random.lognormal(3, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic relationships
        # Vehicle age effect
        df['VehicleAge'] = 2024 - df['RegistrationYear']
        
        # Premium adjustments based on features
        premium_base = df['CalculatedPremiumPerTerm'].copy()
        
        # Vehicle type adjustments
        vehicle_multipliers = {'SUV': 1.3, 'Truck': 1.4, 'Motorcycle': 0.8, 'Sedan': 1.0, 'Hatchback': 0.9, 'Coupe': 1.1}
        for vtype, mult in vehicle_multipliers.items():
            df.loc[df['VehicleType'] == vtype, 'CalculatedPremiumPerTerm'] *= mult
        
        # Province adjustments
        province_multipliers = {'Gauteng': 1.2, 'Western Cape': 1.1, 'KwaZulu-Natal': 1.0, 
                               'Eastern Cape': 0.9, 'Free State': 0.8, 'Limpopo': 0.85}
        for prov, mult in province_multipliers.items():
            df.loc[df['Province'] == prov, 'CalculatedPremiumPerTerm'] *= mult
        
        # Age adjustments
        df.loc[df['VehicleAge'] > 10, 'CalculatedPremiumPerTerm'] *= 1.15
        df.loc[df['VehicleAge'] > 15, 'CalculatedPremiumPerTerm'] *= 1.25
        
        # Generate claims with realistic probabilities
        claim_prob_base = 0.15  # 15% base claim probability
        
        # Adjust claim probability based on features
        claim_prob = np.full(n_samples, claim_prob_base)
        
        # Vehicle type effects on claim probability
        claim_prob[df['VehicleType'] == 'Motorcycle'] *= 1.8
        claim_prob[df['VehicleType'] == 'Truck'] *= 1.4
        claim_prob[df['VehicleType'] == 'SUV'] *= 0.8
        
        # Province effects
        claim_prob[df['Province'] == 'Gauteng'] *= 1.3
        claim_prob[df['Province'] == 'Western Cape'] *= 1.1
        
        # Age effects
        claim_prob[df['VehicleAge'] > 10] *= 1.2
        claim_prob[df['VehicleAge'] > 15] *= 1.4
        
        # Generate binary claims
        df['HasClaim'] = np.random.binomial(1, claim_prob)
        
        # Generate claim amounts for policies with claims
        claim_severity_base = np.random.lognormal(3, 1.2, n_samples)
        
        # Adjust claim severity based on features
        claim_severity = claim_severity_base.copy()
        claim_severity[df['VehicleType'] == 'SUV'] *= 1.4
        claim_severity[df['VehicleType'] == 'Truck'] *= 1.6
        claim_severity[df['VehicleType'] == 'Motorcycle'] *= 0.7
        claim_severity[df['Province'] == 'Gauteng'] *= 1.2
        claim_severity[df['VehicleAge'] > 10] *= 1.1
        
        # Only assign claim amounts to policies with claims
        df['TotalClaims'] = np.where(df['HasClaim'] == 1, claim_severity, 0)
        
        # Add some noise and missing values for realism
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'CustomValueEstimate'] = np.nan
        
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_indices, 'Gender'] = np.nan
        
        return df
    
    def comprehensive_data_preprocessing(self) -> pd.DataFrame:
        """
        Comprehensive data preprocessing including feature engineering
        
        Returns:
            Processed DataFrame ready for modeling
        """
        print("🔄 Starting comprehensive data preprocessing...")
        
        df = self.df.copy()
        
        # 1. Handle missing values
        print("📋 Handling missing values...")
        df = self._handle_missing_values(df)
        
        # 2. Feature engineering
        print("🔧 Creating engineered features...")
        df = self._create_engineered_features(df)
        
        # 3. Encode categorical variables
        print("🏷️ Encoding categorical variables...")
        df = self._encode_categorical_variables(df)
        
        # 4. Feature scaling (for numerical features)
        print("📏 Scaling numerical features...")
        df = self._scale_numerical_features(df)

        # 5. Final missing value and data quality check
        print("🔍 Final data quality check...")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️ Found {missing_values} missing values after preprocessing. Cleaning up...")

            # Handle any remaining missing values
            # Numerical columns - use median imputation
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    nan_count = df[col].isnull().sum()
                    median_value = df[col].median()
                    if pd.isna(median_value):  # If median is also NaN, use 0
                        median_value = 0
                    df[col].fillna(median_value, inplace=True)
                    print(f"  Final cleanup: Filled {nan_count} missing values in {col} with median: {median_value:.2f}")

            # Categorical columns - use mode imputation
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    nan_count = df[col].isnull().sum()
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    print(f"  Final cleanup: Filled {nan_count} missing values in {col} with mode: {mode_value}")

            # Final missing value check
            final_missing = df.isnull().sum().sum()
            print(f"✅ Final missing values after cleanup: {final_missing}")
        else:
            print("✅ No missing values found")

        # Check for infinite values
        print("🔍 Checking for infinite values...")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        inf_found = False
        for col in numerical_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_found = True
                print(f"⚠️ Found {inf_count} infinite values in {col}. Replacing with median...")
                median_value = df[col][~np.isinf(df[col])].median()
                if pd.isna(median_value):
                    median_value = 0
                df[col] = df[col].replace([np.inf, -np.inf], median_value)

        if not inf_found:
            print("✅ No infinite values found")

        self.df_processed = df
        print(f"✅ Data preprocessing completed. Final shape: {df.shape}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""

        # Numerical columns - use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  Filled {col} missing values with median: {median_value:.2f}")

        # Categorical columns - use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} missing values with mode: {mode_value}")

        return df

    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better model performance"""

        # Vehicle age (if not already present)
        if 'VehicleAge' not in df.columns and 'RegistrationYear' in df.columns:
            df['VehicleAge'] = 2024 - df['RegistrationYear']

        # Premium per sum insured ratio
        if 'CalculatedPremiumPerTerm' in df.columns and 'SumInsured' in df.columns:
            df['PremiumToSumInsuredRatio'] = df['CalculatedPremiumPerTerm'] / (df['SumInsured'] + 1)

        # Custom value to sum insured ratio
        if 'CustomValueEstimate' in df.columns and 'SumInsured' in df.columns:
            df['CustomValueToSumInsuredRatio'] = df['CustomValueEstimate'] / (df['SumInsured'] + 1)

        # Vehicle age categories
        if 'VehicleAge' in df.columns:
            df['VehicleAgeCategory'] = pd.cut(df['VehicleAge'],
                                            bins=[-1, 3, 7, 12, 20, 100],
                                            labels=['New', 'Recent', 'Medium', 'Old', 'Very Old'])

        # Premium categories
        if 'CalculatedPremiumPerTerm' in df.columns:
            try:
                df['PremiumCategory'] = pd.qcut(df['CalculatedPremiumPerTerm'],
                                              q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                              duplicates='drop')
            except ValueError:
                # If qcut fails due to duplicate values, use cut instead
                df['PremiumCategory'] = pd.cut(df['CalculatedPremiumPerTerm'],
                                             bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Time-based features
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            df['TransactionYear'] = df['TransactionMonth'].dt.year
            df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
            df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
            df['TransactionDayOfWeek'] = df['TransactionMonth'].dt.dayofweek

        # Risk score based on multiple factors
        risk_score = np.zeros(len(df))

        if 'VehicleAge' in df.columns:
            risk_score += (df['VehicleAge'] / 20) * 0.3  # Age factor

        if 'Province' in df.columns:
            high_risk_provinces = ['Gauteng', 'Western Cape']
            risk_score += df['Province'].isin(high_risk_provinces).astype(int) * 0.2

        if 'VehicleType' in df.columns:
            high_risk_vehicles = ['Motorcycle', 'Truck']
            risk_score += df['VehicleType'].isin(high_risk_vehicles).astype(int) * 0.3

        df['RiskScore'] = risk_score

        print(f"  Created engineered features")

        return df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for machine learning"""

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != 'TransactionMonth']  # Exclude datetime

        encoded_df = df.copy()

        for col in categorical_cols:
            if col in encoded_df.columns:
                # Use one-hot encoding for categorical variables with reasonable cardinality
                unique_values = encoded_df[col].nunique()

                if unique_values <= 10:  # One-hot encode if <= 10 unique values
                    # One-hot encoding
                    dummies = pd.get_dummies(encoded_df[col], prefix=col, drop_first=True)
                    encoded_df = pd.concat([encoded_df, dummies], axis=1)
                    encoded_df.drop(col, axis=1, inplace=True)
                    print(f"  One-hot encoded {col} ({unique_values} categories)")

                else:  # Label encode if > 10 unique values
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                    self.encoders[col] = le
                    print(f"  Label encoded {col} ({unique_values} categories)")

        return encoded_df

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features for better model performance"""

        # Identify numerical columns (excluding target variables and IDs)
        exclude_cols = ['PolicyID', 'TotalClaims', 'HasClaim', 'CalculatedPremiumPerTerm']
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if numerical_cols:
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['numerical'] = scaler
            print(f"  Scaled {len(numerical_cols)} numerical features")
            return df_scaled

        return df

    def _robust_nan_cleanup(self, X: pd.DataFrame, context: str = "features") -> pd.DataFrame:
        """
        Ultra-robust NaN cleanup for feature matrices

        Args:
            X: Feature DataFrame
            context: Context for logging (e.g., "features", "claim severity features")

        Returns:
            Cleaned DataFrame with guaranteed no NaN values
        """
        nan_count_before = X.isnull().sum().sum()
        if nan_count_before > 0:
            print(f"⚠️ Found {nan_count_before} NaN values in {context}. Applying ultra-robust cleanup...")

            X_clean = X.copy()

            # First pass: Manual cleanup
            for col in X_clean.columns:
                if X_clean[col].isnull().sum() > 0:
                    nan_count_col = X_clean[col].isnull().sum()

                    # Check data type more comprehensively
                    if pd.api.types.is_numeric_dtype(X_clean[col]):
                        # For numerical columns
                        non_nan_values = X_clean[col].dropna()
                        if len(non_nan_values) > 0:
                            fill_val = non_nan_values.median()
                            if pd.isna(fill_val) or np.isinf(fill_val):
                                fill_val = 0
                        else:
                            fill_val = 0

                        X_clean[col].fillna(fill_val, inplace=True)
                        print(f"    Filled {nan_count_col} NaN values in {col} with median: {fill_val}")

                    else:
                        # For categorical/boolean columns
                        non_nan_values = X_clean[col].dropna()
                        if len(non_nan_values) > 0:
                            mode_val = non_nan_values.mode()
                            if len(mode_val) > 0:
                                fill_val = mode_val.iloc[0]
                            else:
                                fill_val = 'Unknown' if X_clean[col].dtype == 'object' else 0
                        else:
                            fill_val = 'Unknown' if X_clean[col].dtype == 'object' else 0

                        X_clean[col].fillna(fill_val, inplace=True)
                        print(f"    Filled {nan_count_col} NaN values in {col} with mode/default: {fill_val}")

            # Second pass: Use sklearn SimpleImputer as backup
            nan_count_after_first = X_clean.isnull().sum().sum()
            if nan_count_after_first > 0:
                print(f"⚠️ Still {nan_count_after_first} NaN values after first pass. Using SimpleImputer...")

                from sklearn.impute import SimpleImputer

                # Separate by data type
                numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
                categorical_cols = X_clean.select_dtypes(include=['object', 'bool', 'category']).columns

                # Impute numerical columns
                if len(numerical_cols) > 0:
                    num_imputer = SimpleImputer(strategy='median')
                    X_clean[numerical_cols] = num_imputer.fit_transform(X_clean[numerical_cols])

                # Impute categorical columns
                if len(categorical_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    X_clean[categorical_cols] = cat_imputer.fit_transform(X_clean[categorical_cols])

            # Third pass: Nuclear option - replace any remaining NaN with 0
            nan_count_after_second = X_clean.isnull().sum().sum()
            if nan_count_after_second > 0:
                print(f"⚠️ Still {nan_count_after_second} NaN values. Applying nuclear cleanup (replace with 0)...")
                X_clean = X_clean.fillna(0)

            nan_count_final = X_clean.isnull().sum().sum()
            print(f"✅ Ultra-robust cleanup completed. Before: {nan_count_before}, After: {nan_count_final}")

            # Final verification
            if nan_count_final > 0:
                print(f"❌ CRITICAL: {nan_count_final} NaN values still remain! This should not happen.")
                # Emergency fallback
                X_clean = X_clean.fillna(0)
                print("Applied emergency fallback: replaced all remaining NaN with 0")

            return X_clean

        return X

    def build_claim_severity_models(self) -> Dict[str, Any]:
        """
        Build models to predict claim severity (TotalClaims for policies with claims > 0)

        Returns:
            Dictionary with model results
        """
        print("\n🎯 Building Claim Severity Prediction Models...")

        # Check if processed data exists
        if self.df_processed is None or len(self.df_processed) == 0:
            print("❌ No processed data available for modeling")
            return {}

        # Check if TotalClaims column exists
        if 'TotalClaims' not in self.df_processed.columns:
            print("❌ TotalClaims column not found in processed data")
            return {}

        # Filter data to only include policies with claims
        df_claims = self.df_processed[self.df_processed['TotalClaims'] > 0].copy()

        if len(df_claims) == 0:
            print("❌ No claims data available for modeling")
            return {}

        print(f"📊 Training on {len(df_claims)} policies with claims")

        # Prepare features and target
        target_col = 'TotalClaims'
        exclude_cols = ['PolicyID', 'HasClaim', 'CalculatedPremiumPerTerm', 'TransactionMonth']
        feature_cols = [col for col in df_claims.columns if col not in exclude_cols + [target_col]]

        X = df_claims[feature_cols]
        y = df_claims[target_col]

        # Apply robust NaN cleanup
        X = self._robust_nan_cleanup(X, "claim severity features")

        # Final verification that no NaN values remain
        final_nan_count = X.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️ Still found {final_nan_count} NaN values after cleanup. Applying aggressive cleanup...")
            # Use sklearn's SimpleImputer as a fallback
            from sklearn.impute import SimpleImputer

            # Separate numerical and categorical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

            # Impute numerical columns
            if len(numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

            # Impute categorical columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

            # Final check
            final_final_nan_count = X.isnull().sum().sum()
            print(f"✅ Aggressive cleanup completed. Final NaN count: {final_final_nan_count}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        print(f"📈 Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }

        results = {}

        for name, model in models.items():
            print(f"\n🔧 Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Store results
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_names': feature_cols
            }

            print(f"  📊 {name} Results:")
            print(f"    Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"    Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

        self.results['claim_severity'] = results
        self.models['claim_severity'] = {name: result['model'] for name, result in results.items()}

        return results

    def build_claim_probability_models(self) -> Dict[str, Any]:
        """
        Build models to predict claim probability (binary classification)

        Returns:
            Dictionary with model results
        """
        print("\n🎯 Building Claim Probability Prediction Models...")

        # Check if processed data exists and has required columns
        if self.df_processed is None or len(self.df_processed) == 0:
            print("❌ No processed data available for modeling")
            return {}

        # Prepare features and target
        target_col = 'HasClaim'
        exclude_cols = ['PolicyID', 'TotalClaims', 'CalculatedPremiumPerTerm', 'TransactionMonth']

        # Check if target column exists
        if target_col not in self.df_processed.columns:
            print(f"❌ Target column '{target_col}' not found in processed data")
            return {}

        feature_cols = [col for col in self.df_processed.columns if col not in exclude_cols + [target_col]]

        if len(feature_cols) == 0:
            print("❌ No feature columns available for modeling")
            return {}

        X = self.df_processed[feature_cols]
        y = self.df_processed[target_col]

        # Apply robust NaN cleanup
        X = self._robust_nan_cleanup(X, "claim probability features")

        # Final verification that no NaN values remain
        final_nan_count = X.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️ Still found {final_nan_count} NaN values after cleanup. Applying aggressive cleanup...")
            # Use sklearn's SimpleImputer as a fallback
            from sklearn.impute import SimpleImputer

            # Separate numerical and categorical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

            # Impute numerical columns
            if len(numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

            # Impute categorical columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

            # Final check
            final_final_nan_count = X.isnull().sum().sum()
            print(f"✅ Aggressive cleanup completed. Final NaN count: {final_final_nan_count}")

        print(f"📊 Training on {len(X)} policies")
        print(f"📈 Claim rate: {y.mean():.3f} ({y.sum()} claims out of {len(y)} policies)")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📈 Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }

        results = {}

        for name, model in models.items():
            print(f"\n🔧 Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]

            # Evaluate
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_precision = precision_score(y_train, y_pred_train)
            test_precision = precision_score(y_test, y_pred_test)
            train_recall = recall_score(y_train, y_pred_train)
            test_recall = recall_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train)
            test_f1 = f1_score(y_test, y_pred_test)
            train_auc = roc_auc_score(y_train, y_pred_proba_train)
            test_auc = roc_auc_score(y_test, y_pred_proba_test)

            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'feature_names': feature_cols
            }

            print(f"  📊 {name} Results:")
            print(f"    Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print(f"    Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
            print(f"    Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
            print(f"    Test F1: {test_f1:.4f}")

        self.results['claim_probability'] = results
        self.models['claim_probability'] = {name: result['model'] for name, result in results.items()}

        return results

    def build_premium_optimization_models(self) -> Dict[str, Any]:
        """
        Build models for premium optimization

        Returns:
            Dictionary with model results
        """
        print("\n🎯 Building Premium Optimization Models...")

        # Check if processed data exists and has required columns
        if self.df_processed is None or len(self.df_processed) == 0:
            print("❌ No processed data available for modeling")
            return {}

        # Prepare features and target
        target_col = 'CalculatedPremiumPerTerm'
        exclude_cols = ['PolicyID', 'TotalClaims', 'HasClaim', 'TransactionMonth']

        # Check if target column exists
        if target_col not in self.df_processed.columns:
            print(f"❌ Target column '{target_col}' not found in processed data")
            return {}

        feature_cols = [col for col in self.df_processed.columns if col not in exclude_cols + [target_col]]

        if len(feature_cols) == 0:
            print("❌ No feature columns available for modeling")
            return {}

        X = self.df_processed[feature_cols]
        y = self.df_processed[target_col]

        # Apply robust NaN cleanup
        X = self._robust_nan_cleanup(X, "premium optimization features")

        # Final verification that no NaN values remain
        final_nan_count = X.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️ Still found {final_nan_count} NaN values after cleanup. Applying aggressive cleanup...")
            # Use sklearn's SimpleImputer as a fallback
            from sklearn.impute import SimpleImputer

            # Separate numerical and categorical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

            # Impute numerical columns
            if len(numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

            # Impute categorical columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

            # Final check
            final_final_nan_count = X.isnull().sum().sum()
            print(f"✅ Aggressive cleanup completed. Final NaN count: {final_final_nan_count}")

        print(f"📊 Training on {len(X)} policies")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"📈 Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }

        results = {}

        for name, model in models.items():
            print(f"\n🔧 Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Store results
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_names': feature_cols
            }

            print(f"  📊 {name} Results:")
            print(f"    Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"    Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

        self.results['premium_optimization'] = results
        self.models['premium_optimization'] = {name: result['model'] for name, result in results.items()}

        return results

    def compare_model_performance(self) -> pd.DataFrame:
        """
        Compare performance across all models

        Returns:
            DataFrame with model comparison
        """
        print("\n📊 Comparing Model Performance...")

        comparison_data = []

        # Claim Severity Models
        if 'claim_severity' in self.results:
            for model_name, results in self.results['claim_severity'].items():
                comparison_data.append({
                    'Task': 'Claim Severity',
                    'Model': model_name,
                    'Test RMSE': results['test_rmse'],
                    'Test R²': results['test_r2'],
                    'Test MAE': results['test_mae'],
                    'Metric': 'RMSE (lower better)'
                })

        # Claim Probability Models
        if 'claim_probability' in self.results:
            for model_name, results in self.results['claim_probability'].items():
                comparison_data.append({
                    'Task': 'Claim Probability',
                    'Model': model_name,
                    'Test Accuracy': results['test_accuracy'],
                    'Test AUC': results['test_auc'],
                    'Test F1': results['test_f1'],
                    'Metric': 'AUC (higher better)'
                })

        # Premium Optimization Models
        if 'premium_optimization' in self.results:
            for model_name, results in self.results['premium_optimization'].items():
                comparison_data.append({
                    'Task': 'Premium Optimization',
                    'Model': model_name,
                    'Test RMSE': results['test_rmse'],
                    'Test R²': results['test_r2'],
                    'Test MAE': results['test_mae'],
                    'Metric': 'RMSE (lower better)'
                })

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            print("\n📈 Model Performance Summary:")
            print(comparison_df.to_string(index=False))

        return comparison_df

    def analyze_feature_importance(self, task: str = 'claim_severity', model_name: str = 'XGBoost') -> Dict[str, Any]:
        """
        Analyze feature importance for the specified model

        Args:
            task: Task type ('claim_severity', 'claim_probability', 'premium_optimization')
            model_name: Name of the model to analyze

        Returns:
            Dictionary with feature importance analysis
        """
        print(f"\n🔍 Analyzing Feature Importance for {task} - {model_name}...")

        if task not in self.results or model_name not in self.results[task]:
            print(f"❌ Model {model_name} not found for task {task}")
            return {}

        model = self.results[task][model_name]['model']
        feature_names = self.results[task][model_name]['feature_names']

        importance_data = {}

        # Get feature importance from tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            importance_data['feature_importance'] = feature_importance_df

            print(f"🏆 Top 10 Most Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))

        # Get coefficients from linear models
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:  # Regression
                coefficients = model.coef_
            else:  # Classification
                coefficients = model.coef_[0]

            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            importance_data['coefficients'] = coef_df

            print(f"🏆 Top 10 Most Important Features (by coefficient magnitude):")
            print(coef_df.head(10)[['feature', 'coefficient']].to_string(index=False))

        self.feature_importance[f"{task}_{model_name}"] = importance_data

        return importance_data

    def shap_analysis(self, task: str = 'claim_severity', model_name: str = 'XGBoost',
                     sample_size: int = 1000) -> Dict[str, Any]:
        """
        Perform SHAP analysis for model interpretability

        Args:
            task: Task type
            model_name: Name of the model to analyze
            sample_size: Number of samples to use for SHAP analysis

        Returns:
            Dictionary with SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available. Install with: pip install shap")
            return {}

        print(f"\n🔍 Performing SHAP Analysis for {task} - {model_name}...")

        if task not in self.results or model_name not in self.results[task]:
            print(f"❌ Model {model_name} not found for task {task}")
            return {}

        model = self.results[task][model_name]['model']
        feature_names = self.results[task][model_name]['feature_names']

        # Prepare data for SHAP analysis
        if task == 'claim_severity':
            # Use only policies with claims for claim severity
            df_sample = self.df_processed[self.df_processed['TotalClaims'] > 0].copy()
        else:
            df_sample = self.df_processed.copy()

        # Sample data for faster SHAP computation
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(n=sample_size, random_state=42)

        X_sample = df_sample[feature_names]

        try:
            # Create SHAP explainer
            if model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model)
            elif model_name == 'Random Forest':
                explainer = shap.TreeExplainer(model)
            else:  # Linear models
                explainer = shap.LinearExplainer(model, X_sample)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, use positive class SHAP values
            if len(shap_values.shape) == 3:  # Binary classification
                shap_values = shap_values[:, :, 1]

            # Calculate feature importance from SHAP values
            feature_importance = np.abs(shap_values).mean(0)
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)

            # Store results
            shap_results = {
                'shap_values': shap_values,
                'feature_importance': shap_importance_df,
                'explainer': explainer,
                'sample_data': X_sample
            }

            self.shap_values[f"{task}_{model_name}"] = shap_results

            print(f"🏆 Top 10 Features by SHAP Importance:")
            print(shap_importance_df.head(10).to_string(index=False))

            # Generate business insights
            self._generate_shap_insights(shap_importance_df, task)

            return shap_results

        except Exception as e:
            print(f"❌ Error in SHAP analysis: {e}")
            return {}

    def _generate_shap_insights(self, shap_importance_df: pd.DataFrame, task: str):
        """Generate business insights from SHAP analysis"""

        print(f"\n💡 Business Insights from SHAP Analysis ({task}):")

        top_features = shap_importance_df.head(5)['feature'].tolist()

        insights = []

        for feature in top_features:
            if 'VehicleAge' in feature:
                insights.append(f"🚗 Vehicle age is a key risk factor - older vehicles show higher {task.replace('_', ' ')}")
            elif 'Province' in feature or any(prov in feature for prov in ['Gauteng', 'Western Cape']):
                insights.append(f"🏢 Geographic location significantly impacts {task.replace('_', ' ')} - regional risk variations detected")
            elif 'VehicleType' in feature or any(vtype in feature for vtype in ['Motorcycle', 'Truck', 'SUV']):
                insights.append(f"🚙 Vehicle type is crucial for {task.replace('_', ' ')} prediction - different vehicle categories show distinct risk profiles")
            elif 'RiskScore' in feature:
                insights.append(f"📊 Composite risk score effectively captures multiple risk factors for {task.replace('_', ' ')}")
            elif 'Premium' in feature:
                insights.append(f"💰 Premium-related features are important predictors, indicating pricing-risk relationship")

        for insight in insights[:3]:  # Show top 3 insights
            print(f"  {insight}")

    def create_model_visualizations(self, save_plots: bool = True):
        """
        Create comprehensive visualizations for model results

        Args:
            save_plots: Whether to save plots to files
        """
        print("\n📊 Creating Model Visualizations...")

        # Create plots directory
        os.makedirs('plots/models', exist_ok=True)

        # 1. Model Performance Comparison
        self._plot_model_comparison(save_plots)

        # 2. Feature Importance Plots
        self._plot_feature_importance(save_plots)

        # 3. SHAP Visualizations
        self._plot_shap_analysis(save_plots)

        print("✅ Model visualizations created successfully!")

    def _plot_model_comparison(self, save_plots: bool):
        """Plot model performance comparison"""

        comparison_df = self.compare_model_performance()

        if comparison_df.empty:
            return

        # Create subplots for different tasks
        tasks = comparison_df['Task'].unique()
        n_tasks = len(tasks)

        fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(tasks):
            task_data = comparison_df[comparison_df['Task'] == task]

            if task in ['Claim Severity', 'Premium Optimization']:
                # Plot RMSE and R²
                ax1 = axes[i]
                ax2 = ax1.twinx()

                models = task_data['Model']
                rmse_values = task_data['Test RMSE'] if 'Test RMSE' in task_data.columns else []
                r2_values = task_data['Test R²'] if 'Test R²' in task_data.columns else []

                if len(rmse_values) > 0:
                    bars1 = ax1.bar([f"{m}\n(RMSE)" for m in models], rmse_values, alpha=0.7, color='red', label='RMSE')
                if len(r2_values) > 0:
                    bars2 = ax2.bar([f"{m}\n(R²)" for m in models], r2_values, alpha=0.7, color='blue', label='R²')

                ax1.set_ylabel('RMSE (lower better)', color='red')
                ax2.set_ylabel('R² (higher better)', color='blue')
                ax1.set_title(f'{task} Model Comparison')

            else:  # Claim Probability
                models = task_data['Model']
                auc_values = task_data['Test AUC'] if 'Test AUC' in task_data.columns else []

                if len(auc_values) > 0:
                    axes[i].bar(models, auc_values, alpha=0.7, color='green')
                    axes[i].set_ylabel('AUC (higher better)')
                    axes[i].set_title(f'{task} Model Comparison')
                    axes[i].set_ylim(0, 1)

        plt.tight_layout()
        if save_plots:
            plt.savefig('plots/models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_feature_importance(self, save_plots: bool):
        """Plot feature importance for best models"""

        if not self.feature_importance:
            return

        for model_key, importance_data in self.feature_importance.items():
            task, model_name = model_key.split('_', 1)

            plt.figure(figsize=(12, 8))

            if 'feature_importance' in importance_data:
                # Tree-based model importance
                top_features = importance_data['feature_importance'].head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance - {task.replace("_", " ").title()} ({model_name})')

            elif 'coefficients' in importance_data:
                # Linear model coefficients
                top_features = importance_data['coefficients'].head(15)
                colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
                plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Coefficient Value')
                plt.title(f'Feature Coefficients - {task.replace("_", " ").title()} ({model_name})')

            plt.gca().invert_yaxis()
            plt.tight_layout()

            if save_plots:
                plt.savefig(f'plots/models/feature_importance_{task}_{model_name.lower().replace(" ", "_")}.png',
                           dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_shap_analysis(self, save_plots: bool):
        """Plot SHAP analysis results"""

        if not SHAP_AVAILABLE or not self.shap_values:
            return

        for model_key, shap_data in self.shap_values.items():
            task, model_name = model_key.split('_', 1)

            try:
                # SHAP Summary Plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_data['shap_values'],
                    shap_data['sample_data'],
                    plot_type="bar",
                    show=False
                )
                plt.title(f'SHAP Feature Importance - {task.replace("_", " ").title()} ({model_name})')

                if save_plots:
                    plt.savefig(f'plots/models/shap_importance_{task}_{model_name.lower().replace(" ", "_")}.png',
                               dpi=300, bbox_inches='tight')
                plt.show()

                # SHAP Detailed Plot (if sample size is reasonable)
                if len(shap_data['sample_data']) <= 500:
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_data['shap_values'],
                        shap_data['sample_data'],
                        show=False
                    )
                    plt.title(f'SHAP Detailed Analysis - {task.replace("_", " ").title()} ({model_name})')

                    if save_plots:
                        plt.savefig(f'plots/models/shap_detailed_{task}_{model_name.lower().replace(" ", "_")}.png',
                                   dpi=300, bbox_inches='tight')
                    plt.show()

            except Exception as e:
                print(f"❌ Error creating SHAP plots for {model_key}: {e}")

    def calculate_risk_based_premium(self, policy_features: pd.DataFrame,
                                   expense_loading: float = 0.15,
                                   profit_margin: float = 0.10) -> pd.DataFrame:
        """
        Calculate risk-based premium using the advanced framework:
        Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin

        Args:
            policy_features: DataFrame with policy features
            expense_loading: Expense loading factor (default 15%)
            profit_margin: Profit margin factor (default 10%)

        Returns:
            DataFrame with risk-based premium calculations
        """
        print("\n💰 Calculating Risk-Based Premiums...")

        if 'claim_probability' not in self.models or 'claim_severity' not in self.models:
            print("❌ Required models not available for premium calculation")
            return pd.DataFrame()

        # Get best models (assuming XGBoost performs best)
        prob_model = self.models['claim_probability'].get('XGBoost')
        severity_model = self.models['claim_severity'].get('XGBoost')

        if prob_model is None or severity_model is None:
            print("❌ XGBoost models not available")
            return pd.DataFrame()

        # Predict claim probability
        claim_probability = prob_model.predict_proba(policy_features)[:, 1]

        # Predict claim severity
        claim_severity = severity_model.predict(policy_features)

        # Calculate expected claim cost
        expected_claim_cost = claim_probability * claim_severity

        # Calculate risk-based premium
        base_premium = expected_claim_cost
        expense_cost = base_premium * expense_loading
        profit_cost = base_premium * profit_margin

        risk_based_premium = base_premium + expense_cost + profit_cost

        # Create results DataFrame
        results_df = pd.DataFrame({
            'ClaimProbability': claim_probability,
            'ExpectedClaimSeverity': claim_severity,
            'ExpectedClaimCost': expected_claim_cost,
            'ExpenseLoading': expense_cost,
            'ProfitMargin': profit_cost,
            'RiskBasedPremium': risk_based_premium
        })

        print(f"📊 Risk-Based Premium Summary:")
        print(f"  Average Claim Probability: {claim_probability.mean():.3f}")
        print(f"  Average Expected Claim Severity: R{claim_severity.mean():,.2f}")
        print(f"  Average Risk-Based Premium: R{risk_based_premium.mean():,.2f}")

        return results_df

    def run_complete_modeling_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete predictive modeling pipeline

        Returns:
            Dictionary with all modeling results
        """
        print("🚀 Starting Complete Predictive Modeling Pipeline...")

        # 1. Load and prepare data
        self.load_and_prepare_data()

        # 2. Comprehensive preprocessing
        self.comprehensive_data_preprocessing()

        # 3. Build all models
        print("\n" + "="*60)
        claim_severity_results = self.build_claim_severity_models()

        print("\n" + "="*60)
        claim_probability_results = self.build_claim_probability_models()

        print("\n" + "="*60)
        premium_optimization_results = self.build_premium_optimization_models()

        # 4. Model comparison
        print("\n" + "="*60)
        comparison_df = self.compare_model_performance()

        # 5. Feature importance analysis
        print("\n" + "="*60)
        for task in ['claim_severity', 'claim_probability', 'premium_optimization']:
            if task in self.results:
                self.analyze_feature_importance(task, 'XGBoost')

        # 6. SHAP analysis
        print("\n" + "="*60)
        for task in ['claim_severity', 'claim_probability']:
            if task in self.results:
                self.shap_analysis(task, 'XGBoost')

        # 7. Create visualizations
        print("\n" + "="*60)
        self.create_model_visualizations()

        # 8. Calculate risk-based premiums (sample)
        print("\n" + "="*60)
        if self.df_processed is not None and len(self.df_processed) > 0:
            sample_features = self.df_processed.head(100)
            feature_cols = [col for col in sample_features.columns
                          if col not in ['PolicyID', 'TotalClaims', 'HasClaim', 'CalculatedPremiumPerTerm', 'TransactionMonth']]
            risk_premiums = self.calculate_risk_based_premium(sample_features[feature_cols])

        # Compile all results
        all_results = {
            'claim_severity': claim_severity_results,
            'claim_probability': claim_probability_results,
            'premium_optimization': premium_optimization_results,
            'model_comparison': comparison_df,
            'feature_importance': self.feature_importance,
            'shap_analysis': self.shap_values
        }

        print("\n✅ Complete Predictive Modeling Pipeline Completed!")
        print("📁 All visualizations saved in 'plots/models' directory")

        return all_results


def main():
    """
    Main function to run the predictive modeling pipeline
    """
    # Initialize modeling class
    modeling = InsurancePredictiveModeling(data_path="../data/MachineLearningRating_v3.txt")

    # Run complete pipeline
    results = modeling.run_complete_modeling_pipeline()

    return results


if __name__ == "__main__":
    main()
