#!/usr/bin/env python3
"""
Quick fix script to handle missing columns in real insurance data
This script creates the necessary target variables for modeling
"""

import pandas as pd
import numpy as np
import sys
import os

def fix_insurance_data(data_path):
    """
    Fix insurance data by creating necessary columns for modeling
    
    Args:
        data_path: Path to the insurance dataset
    """
    try:
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, sep="|", low_memory=False)
        print(f"Original data shape: {df.shape}")
        
        # Display available columns
        print(f"\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Create HasClaim column if it doesn't exist
        if 'HasClaim' not in df.columns:
            if 'TotalClaims' in df.columns:
                # Convert TotalClaims to numeric, handling any non-numeric values
                df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
                # Create binary HasClaim variable
                df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
                print("✅ Created HasClaim column from TotalClaims")
            else:
                print("⚠️ TotalClaims column not found. Cannot create HasClaim column.")
                return None
        
        # Ensure TotalClaims is numeric
        if 'TotalClaims' in df.columns:
            df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
        
        # Handle premium column
        if 'CalculatedPremiumPerTerm' not in df.columns:
            if 'TotalPremium' in df.columns:
                df['CalculatedPremiumPerTerm'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
                print("✅ Created CalculatedPremiumPerTerm from TotalPremium")
            else:
                print("⚠️ No premium column found")
                return None
        else:
            df['CalculatedPremiumPerTerm'] = pd.to_numeric(df['CalculatedPremiumPerTerm'], errors='coerce')
        
        # Display summary statistics
        print(f"\n📊 Data Summary After Processing:")
        print(f"Dataset shape: {df.shape}")
        
        if 'HasClaim' in df.columns:
            claim_rate = df['HasClaim'].mean()
            claim_count = df['HasClaim'].sum()
            print(f"Claim rate: {claim_rate:.3f} ({claim_count:,} claims out of {len(df):,} policies)")
        
        if 'TotalClaims' in df.columns:
            total_claims = df['TotalClaims'].sum()
            claims_with_amount = df[df['TotalClaims'] > 0]['TotalClaims']
            avg_claim = claims_with_amount.mean() if len(claims_with_amount) > 0 else 0
            print(f"Total claims: R{total_claims:,.2f}")
            print(f"Average claim (when > 0): R{avg_claim:,.2f}")
        
        if 'CalculatedPremiumPerTerm' in df.columns:
            total_premium = df['CalculatedPremiumPerTerm'].sum()
            avg_premium = df['CalculatedPremiumPerTerm'].mean()
            print(f"Total premiums: R{total_premium:,.2f}")
            print(f"Average premium: R{avg_premium:,.2f}")
            
            if 'TotalClaims' in df.columns:
                loss_ratio = total_claims / total_premium if total_premium > 0 else 0
                print(f"Overall loss ratio: {loss_ratio:.3f}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def main():
    """Main function"""
    
    # Default data path
    data_path = "data/MachineLearningRating_v3.txt"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Available files in data directory:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                print(f"  - {file}")
        return
    
    # Process the data
    df = fix_insurance_data(data_path)
    
    if df is not None:
        print("\n✅ Data processing completed successfully!")
        print("You can now run the predictive modeling pipeline.")
        
        # Optionally save the processed data
        output_path = "data/processed_insurance_data.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
        
        # Show sample of processed data
        print(f"\n📋 Sample of processed data:")
        key_columns = ['HasClaim', 'TotalClaims', 'CalculatedPremiumPerTerm']
        available_key_cols = [col for col in key_columns if col in df.columns]
        
        if available_key_cols:
            print(df[available_key_cols].head(10))
        else:
            print(df.head())
    else:
        print("❌ Data processing failed!")


if __name__ == "__main__":
    main()
