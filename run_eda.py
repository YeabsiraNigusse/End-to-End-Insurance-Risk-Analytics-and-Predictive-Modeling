#!/usr/bin/env python3
"""
Insurance Risk Analytics - EDA Runner Script

This script runs the comprehensive EDA analysis for insurance data.
It can be used for both automated analysis and demonstration purposes.

Usage:
    python run_eda.py [--data-path PATH] [--save-plots] [--verbose]

Author: Insurance Analytics Team
Date: 2025-06-18
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.eda_analysis import InsuranceEDA


def main():
    """Main function to run EDA analysis"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run comprehensive EDA analysis for insurance data'
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='data/MachineLearningRating_v3.txt',
        help='Path to the insurance dataset'
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help='Save plots to files'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run with sample data for demonstration'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 Insurance Risk Analytics - EDA Analysis")
    print("=" * 50)
    
    try:
        # Initialize EDA class
        if args.demo:
            print("📊 Running with sample data for demonstration...")
            eda = InsuranceEDA(data_path=None)  # Will create sample data
        else:
            print(f"📁 Data path: {args.data_path}")
            eda = InsuranceEDA(data_path=args.data_path)
        
        # Run comprehensive analysis
        print("\n🔍 Starting comprehensive EDA analysis...")
        results = eda.run_complete_analysis()
        
        # Print summary of results
        print("\n📋 ANALYSIS SUMMARY")
        print("-" * 30)
        
        if 'data_overview' in results:
            overview = results['data_overview']
            print(f"Dataset shape: {overview['shape']}")
            print(f"Memory usage: {overview['memory_usage']}")
            print(f"Missing values: {overview['missing_values_total']}")
        
        if 'loss_ratio_analysis' in results and 'overall_loss_ratio' in results['loss_ratio_analysis']:
            loss_ratio = results['loss_ratio_analysis']['overall_loss_ratio']
            print(f"Overall Loss Ratio: {loss_ratio:.4f}")
            
            if loss_ratio > 1:
                print("⚠️  Portfolio is currently unprofitable")
            else:
                print("✅ Portfolio is profitable")
        
        # Print key insights
        if 'insights' in results and results['insights']:
            print(f"\n🎯 KEY INSIGHTS ({len(results['insights'])} found):")
            for i, insight in enumerate(results['insights'][:5], 1):  # Show top 5
                print(f"{i}. {insight}")
        
        # Statistical test results
        if 'statistical_tests' in results and results['statistical_tests']:
            print(f"\n🧪 STATISTICAL TESTS:")
            for test_name, test_result in results['statistical_tests'].items():
                significance = "✅ Significant" if test_result['significant'] else "❌ Not significant"
                print(f"  {test_name.replace('_', ' ').title()}: {significance} (p={test_result['p_value']:.4f})")
        
        # Visualization info
        plots_dir = Path('plots')
        if plots_dir.exists():
            plot_files = list(plots_dir.glob('*.png')) + list(plots_dir.glob('*.html'))
            print(f"\n📊 VISUALIZATIONS: {len(plot_files)} files created in 'plots' directory")
            
            if args.verbose and plot_files:
                print("  Created files:")
                for plot_file in plot_files:
                    print(f"    - {plot_file.name}")
        
        print(f"\n✅ EDA Analysis completed successfully!")
        print(f"📁 Results saved in 'plots' directory")
        
        # Return results for programmatic use
        return results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def run_quick_demo():
    """Run a quick demonstration of the EDA capabilities"""
    print("🎬 Running Quick EDA Demo...")
    print("=" * 40)
    
    # Initialize with sample data
    eda = InsuranceEDA()
    
    # Load sample data
    df = eda.load_data()
    print(f"✅ Sample data created: {df.shape}")
    
    # Run key analyses
    print("\n📊 Running key analyses...")
    
    # Data overview
    overview = eda.data_overview()
    
    # Loss ratio analysis
    loss_analysis = eda.loss_ratio_analysis()
    
    # Generate insights
    insights = eda.generate_insights()
    
    print(f"\n🎯 Generated {len(insights)} insights")
    print("✅ Demo completed successfully!")
    
    return True


if __name__ == "__main__":
    # Check if running as demo
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick demo...")
        run_quick_demo()
    else:
        main()
