"""
Data preprocessing module for insurance risk analytics and hypothesis testing.

This module provides functions to clean, transform, and prepare insurance data
for statistical hypothesis testing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings

class InsuranceDataProcessor:
    """
    A class to handle preprocessing of insurance data for hypothesis testing.
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the insurance data file
            df: Pre-loaded DataFrame (alternative to data_path)
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = self.load_data(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.original_shape = self.df.shape
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load insurance data from file."""
        try:
            # Assuming pipe-separated file based on EDA notebook
            df = pd.read_csv(data_path, sep="|", low_memory=False)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the insurance data for analysis.
        
        Returns:
            Cleaned DataFrame
        """
        print("Starting data cleaning...")
        
        # Convert date columns
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(
                self.df['TransactionMonth'], errors='coerce'
            )
        
        # Convert numeric columns
        numeric_columns = ['TotalClaims', 'TotalPremium', 'SumInsured', 
                          'CalculatedPremiumPerTerm']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove rows with missing critical data
        critical_columns = ['TotalPremium', 'TotalClaims', 'Province', 'PostalCode']
        before_cleaning = len(self.df)
        
        for col in critical_columns:
            if col in self.df.columns:
                self.df = self.df.dropna(subset=[col])
        
        after_cleaning = len(self.df)
        print(f"Removed {before_cleaning - after_cleaning} rows with missing critical data")
        
        return self.df
    
    def create_risk_metrics(self) -> pd.DataFrame:
        """
        Create risk metrics for hypothesis testing.
        
        Returns:
            DataFrame with additional risk metrics
        """
        print("Creating risk metrics...")
        
        # Claim Frequency: Binary indicator if policy has claims
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        # Claim Severity: Average claim amount given a claim occurred
        # For policies with claims, severity = TotalClaims
        # For policies without claims, severity = 0 (will be excluded in severity analysis)
        self.df['ClaimSeverity'] = np.where(
            self.df['HasClaim'] == 1, 
            self.df['TotalClaims'], 
            np.nan
        )
        
        # Margin: TotalPremium - TotalClaims
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Loss Ratio: TotalClaims / TotalPremium
        self.df['LossRatio'] = np.where(
            self.df['TotalPremium'] != 0,
            self.df['TotalClaims'] / self.df['TotalPremium'],
            np.nan
        )
        
        print("Risk metrics created successfully")
        return self.df
    
    def prepare_hypothesis_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for each hypothesis test.
        
        Returns:
            Dictionary with prepared datasets for each hypothesis
        """
        print("Preparing data for hypothesis testing...")
        
        # Clean data and create metrics
        self.clean_data()
        self.create_risk_metrics()
        
        # Prepare datasets for each hypothesis
        hypothesis_data = {}
        
        # H1: Risk differences across provinces
        hypothesis_data['provinces'] = self.df[
            ['Province', 'HasClaim', 'ClaimSeverity', 'Margin', 'LossRatio', 
             'TotalPremium', 'TotalClaims']
        ].dropna(subset=['Province'])
        
        # H2: Risk differences between zip codes
        hypothesis_data['zip_codes'] = self.df[
            ['PostalCode', 'HasClaim', 'ClaimSeverity', 'Margin', 'LossRatio',
             'TotalPremium', 'TotalClaims']
        ].dropna(subset=['PostalCode'])
        
        # H3: Margin differences between zip codes (same as H2 but focus on margin)
        hypothesis_data['zip_codes_margin'] = hypothesis_data['zip_codes'].copy()
        
        # H4: Risk differences between genders
        hypothesis_data['gender'] = self.df[
            ['Gender', 'HasClaim', 'ClaimSeverity', 'Margin', 'LossRatio',
             'TotalPremium', 'TotalClaims']
        ].dropna(subset=['Gender'])
        
        # Print summary statistics
        for key, data in hypothesis_data.items():
            print(f"{key.upper()} dataset: {data.shape[0]} records")
        
        return hypothesis_data
    
    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for key variables by grouping factors.

        Returns:
            Dictionary with summary statistics
        """
        # Ensure risk metrics are created first
        if 'HasClaim' not in self.df.columns:
            self.clean_data()
            self.create_risk_metrics()

        summaries = {}

        # Province summary
        if 'Province' in self.df.columns:
            province_summary = self.df.groupby('Province').agg({
                'HasClaim': ['count', 'mean'],
                'ClaimSeverity': ['mean', 'std'],
                'Margin': ['mean', 'std'],
                'LossRatio': ['mean', 'std'],
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).round(4)
            summaries['province'] = province_summary

        # Gender summary
        if 'Gender' in self.df.columns:
            gender_summary = self.df.groupby('Gender').agg({
                'HasClaim': ['count', 'mean'],
                'ClaimSeverity': ['mean', 'std'],
                'Margin': ['mean', 'std'],
                'LossRatio': ['mean', 'std'],
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).round(4)
            summaries['gender'] = gender_summary

        return summaries
    
    def sample_data_for_testing(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Create a sample of the data for testing purposes.
        
        Args:
            sample_size: Number of records to sample
            
        Returns:
            Sampled DataFrame
        """
        if len(self.df) > sample_size:
            sampled_df = self.df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} records from {len(self.df)} total records")
            return sampled_df
        else:
            print(f"Dataset has {len(self.df)} records, returning full dataset")
            return self.df
