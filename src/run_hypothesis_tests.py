"""
Standalone script to run hypothesis testing analysis.

This script can be run independently to perform all hypothesis tests
and generate results and reports.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import InsuranceDataProcessor
from statistical_utils import StatisticalTester
from hypothesis_testing import InsuranceHypothesisTester

def create_sample_data(n_samples=10000):
    """
    Create sample insurance data for demonstration purposes.
    
    Args:
        n_samples: Number of sample records to create
        
    Returns:
        DataFrame with sample insurance data
    """
    print(f"Creating sample data with {n_samples} records...")
    
    np.random.seed(42)
    
    # Create realistic sample data
    provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape', 
                'Free State', 'Mpumalanga', 'Limpopo', 'North West', 'Northern Cape']
    
    # Create data with some realistic patterns
    df = pd.DataFrame({
        'PolicyID': range(1, n_samples + 1),
        'Province': np.random.choice(provinces, n_samples, 
                                   p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.06, 0.05, 0.03]),
        'PostalCode': np.random.choice(range(1000, 9999), n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'Age': np.random.normal(40, 15, n_samples).clip(18, 80),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck'], n_samples),
    })
    
    # Create correlated premium and claims data with province effects
    province_risk_multiplier = {
        'Gauteng': 1.2,  # Higher risk
        'Western Cape': 0.9,  # Lower risk
        'KwaZulu-Natal': 1.1,
        'Eastern Cape': 1.0,
        'Free State': 0.8,
        'Mpumalanga': 1.0,
        'Limpopo': 0.9,
        'North West': 0.95,
        'Northern Cape': 0.85
    }
    
    # Gender risk effects (subtle)
    gender_risk_multiplier = {'Male': 1.05, 'Female': 0.98, 'Other': 1.0}
    
    # Calculate base premium
    base_premium = 100 + df['Age'] * 2 + np.random.exponential(50, n_samples)
    
    # Apply province and gender effects
    df['risk_multiplier'] = df['Province'].map(province_risk_multiplier) * df['Gender'].map(gender_risk_multiplier)
    df['TotalPremium'] = base_premium * df['risk_multiplier'] + np.random.normal(0, 20, n_samples)
    df['TotalPremium'] = df['TotalPremium'].clip(10, None)  # Minimum premium
    
    # Create claims with realistic patterns
    claim_probability = 0.25 * df['risk_multiplier']  # Base 25% claim rate
    has_claim = np.random.binomial(1, claim_probability.clip(0, 1), n_samples)
    
    # Claim amounts for those with claims
    claim_amounts = np.random.exponential(df['TotalPremium'] * 0.8, n_samples) * has_claim
    df['TotalClaims'] = claim_amounts
    
    # Clean up temporary columns
    df = df.drop(['risk_multiplier'], axis=1)
    
    print(f"Sample data created successfully!")
    print(f"  - Provinces: {df['Province'].nunique()}")
    print(f"  - Postal codes: {df['PostalCode'].nunique()}")
    print(f"  - Overall claim frequency: {(df['TotalClaims'] > 0).mean():.3f}")
    print(f"  - Average premium: {df['TotalPremium'].mean():.2f}")
    print(f"  - Average claims: {df['TotalClaims'].mean():.2f}")
    
    return df

def load_or_create_data(data_path="../data/MachineLearningRating_v3.txt"):
    """
    Load real data if available, otherwise create sample data.
    
    Args:
        data_path: Path to the real data file
        
    Returns:
        DataFrame with insurance data
    """
    try:
        print(f"Attempting to load data from {data_path}...")
        df = pd.read_csv(data_path, sep="|", low_memory=False)
        print(f"Real data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Real data file not found at {data_path}")
        print("Creating sample data for demonstration...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def save_results(results, output_dir="../results"):
    """
    Save hypothesis testing results to files.
    
    Args:
        results: Dictionary with test results
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_results = []
    hypothesis_names = {
        'H1_provinces': 'H₁: Risk differences across provinces',
        'H2_zip_codes': 'H₂: Risk differences between zip codes',
        'H3_zip_margin': 'H₃: Margin differences between zip codes',
        'H4_gender': 'H₄: Risk differences between genders'
    }
    
    for hypothesis_key, hypothesis_name in hypothesis_names.items():
        if hypothesis_key in results:
            result = results[hypothesis_key]
            
            if 'tests' in result:
                for test_name, test_result in result['tests'].items():
                    detailed_results.append({
                        'Hypothesis': hypothesis_name,
                        'Test_Type': test_name,
                        'Statistical_Test': test_result.get('test_name', 'N/A'),
                        'Statistic': test_result.get('statistic', np.nan),
                        'P_Value': test_result.get('p_value', np.nan),
                        'Effect_Size': test_result.get('effect_size', np.nan),
                        'Significant': test_result.get('significant', False),
                        'Decision': result.get('conclusion', {}).get('decision', 'N/A'),
                        'Interpretation': test_result.get('interpretation', 'N/A')
                    })
    
    # Save to CSV
    results_df = pd.DataFrame(detailed_results)
    results_file = os.path.join(output_dir, 'hypothesis_testing_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Detailed results saved to: {results_file}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, 'hypothesis_testing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("INSURANCE RISK ANALYTICS - HYPOTHESIS TESTING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'summary' in results:
            summary = results['summary']
            f.write(f"Total hypotheses tested: {summary.get('total_hypotheses_tested', 0)}\n\n")
            
            f.write(f"REJECTED HYPOTHESES ({len(summary.get('rejected_hypotheses', []))}):\n")
            for rejected in summary.get('rejected_hypotheses', []):
                f.write(f"  • {rejected.get('hypothesis', 'N/A')}\n")
                f.write(f"    Evidence: {rejected.get('evidence', 'N/A')}\n\n")
            
            f.write(f"FAILED TO REJECT HYPOTHESES ({len(summary.get('failed_to_reject_hypotheses', []))}):\n")
            for not_rejected in summary.get('failed_to_reject_hypotheses', []):
                f.write(f"  • {not_rejected.get('hypothesis', 'N/A')}\n")
                f.write(f"    Evidence: {not_rejected.get('evidence', 'N/A')}\n\n")
            
            f.write("BUSINESS RECOMMENDATIONS:\n")
            for i, recommendation in enumerate(summary.get('business_recommendations', []), 1):
                f.write(f"  {i}. {recommendation}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("DETAILED RESULTS:\n")
        f.write("=" * 60 + "\n")
        
        for hypothesis_key, hypothesis_name in hypothesis_names.items():
            if hypothesis_key in results:
                result = results[hypothesis_key]
                f.write(f"\n{hypothesis_name}\n")
                f.write("-" * len(hypothesis_name) + "\n")
                f.write(f"Null Hypothesis: {result.get('hypothesis', 'N/A')}\n\n")
                
                if 'tests' in result:
                    for test_name, test_result in result['tests'].items():
                        f.write(f"{test_name.upper()} TEST:\n")
                        f.write(f"  Test: {test_result.get('test_name', 'N/A')}\n")
                        f.write(f"  Statistic: {test_result.get('statistic', 'N/A'):.4f}\n")
                        f.write(f"  P-value: {test_result.get('p_value', 'N/A'):.6f}\n")
                        f.write(f"  Significant: {test_result.get('significant', 'N/A')}\n")
                        f.write(f"  Interpretation: {test_result.get('interpretation', 'N/A')}\n\n")
                
                if 'conclusion' in result:
                    conclusion = result['conclusion']
                    f.write(f"CONCLUSION:\n")
                    f.write(f"  Decision: {conclusion.get('decision', 'N/A')}\n")
                    f.write(f"  Evidence: {conclusion.get('evidence', 'N/A')}\n\n")
    
    print(f"Summary report saved to: {summary_file}")

def main():
    """Main function to run the hypothesis testing analysis."""
    print("INSURANCE RISK ANALYTICS - HYPOTHESIS TESTING")
    print("=" * 50)
    
    # Load data
    df = load_or_create_data()
    
    # Initialize hypothesis tester
    print("\nInitializing hypothesis tester...")
    hypothesis_tester = InsuranceHypothesisTester(df=df, alpha=0.05)
    
    # Run all tests
    print("\nRunning comprehensive hypothesis testing...")
    results = hypothesis_tester.run_all_hypothesis_tests()
    
    # Display summary
    print("\n" + "=" * 50)
    print("HYPOTHESIS TESTING COMPLETED")
    print("=" * 50)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"Total hypotheses tested: {summary.get('total_hypotheses_tested', 0)}")
        print(f"Rejected hypotheses: {len(summary.get('rejected_hypotheses', []))}")
        print(f"Failed to reject: {len(summary.get('failed_to_reject_hypotheses', []))}")
    
    # Save results
    print("\nSaving results...")
    save_results(results)
    
    print("\nAnalysis complete! Check the results directory for detailed outputs.")

if __name__ == "__main__":
    main()
