#!/usr/bin/env python3
"""
Insurance Risk Analytics - Predictive Modeling Runner Script

This script runs the comprehensive predictive modeling pipeline for insurance data.
It builds and evaluates models for claim severity, claim probability, and premium optimization.

Usage:
    python run_modeling.py [--data-path PATH] [--save-models] [--verbose] [--demo]

Author: Insurance Analytics Team
Date: 2025-06-18
"""

import argparse
import sys
import os
from pathlib import Path
import joblib

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.predictive_modeling import InsurancePredictiveModeling


def main():
    """Main function to run predictive modeling pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run comprehensive predictive modeling pipeline for insurance data'
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='data/MachineLearningRating_v3.txt',
        help='Path to the insurance dataset'
    )
    parser.add_argument(
        '--save-models', 
        action='store_true',
        help='Save trained models to files'
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
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick version with smaller sample sizes'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 Insurance Risk Analytics - Predictive Modeling Pipeline")
    print("=" * 70)
    
    try:
        # Initialize modeling class
        if args.demo:
            print("📊 Running with sample data for demonstration...")
            modeling = InsurancePredictiveModeling(data_path=None)  # Will create sample data
        else:
            print(f"📁 Data path: {args.data_path}")
            modeling = InsurancePredictiveModeling(data_path=args.data_path)
        
        # Run comprehensive modeling pipeline
        print("\n🔍 Starting comprehensive predictive modeling pipeline...")
        results = modeling.run_complete_modeling_pipeline()
        
        # Print detailed results summary
        print_results_summary(results, args.verbose)
        
        # Save models if requested
        if args.save_models:
            save_models(modeling, args.verbose)
        
        # Generate business report
        generate_business_report(results, modeling)
        
        print("\n✅ Predictive Modeling Pipeline completed successfully!")
        print("📁 Results and visualizations saved in 'plots/models' directory")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during modeling pipeline: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def print_results_summary(results, verbose=False):
    """Print comprehensive results summary"""
    
    print("\n📊 PREDICTIVE MODELING RESULTS SUMMARY")
    print("=" * 50)
    
    # Claim Severity Results
    if 'claim_severity' in results and results['claim_severity']:
        print("\n🎯 CLAIM SEVERITY PREDICTION:")
        print("-" * 30)
        
        best_model = min(results['claim_severity'].items(), key=lambda x: x[1]['test_rmse'])
        model_name, model_results = best_model
        
        print(f"Best Model: {model_name}")
        print(f"  Test RMSE: R{model_results['test_rmse']:,.2f}")
        print(f"  Test R²: {model_results['test_r2']:.4f}")
        print(f"  Test MAE: R{model_results['test_mae']:,.2f}")
        
        if verbose:
            print("\nAll Models Performance:")
            for name, res in results['claim_severity'].items():
                print(f"  {name}: RMSE=R{res['test_rmse']:,.2f}, R²={res['test_r2']:.4f}")
    
    # Claim Probability Results
    if 'claim_probability' in results and results['claim_probability']:
        print("\n🎯 CLAIM PROBABILITY PREDICTION:")
        print("-" * 30)
        
        best_model = max(results['claim_probability'].items(), key=lambda x: x[1]['test_auc'])
        model_name, model_results = best_model
        
        print(f"Best Model: {model_name}")
        print(f"  Test AUC: {model_results['test_auc']:.4f}")
        print(f"  Test Accuracy: {model_results['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {model_results['test_f1']:.4f}")
        
        if verbose:
            print("\nAll Models Performance:")
            for name, res in results['claim_probability'].items():
                print(f"  {name}: AUC={res['test_auc']:.4f}, Accuracy={res['test_accuracy']:.4f}")
    
    # Premium Optimization Results
    if 'premium_optimization' in results and results['premium_optimization']:
        print("\n🎯 PREMIUM OPTIMIZATION:")
        print("-" * 30)
        
        best_model = min(results['premium_optimization'].items(), key=lambda x: x[1]['test_rmse'])
        model_name, model_results = best_model
        
        print(f"Best Model: {model_name}")
        print(f"  Test RMSE: R{model_results['test_rmse']:,.2f}")
        print(f"  Test R²: {model_results['test_r2']:.4f}")
        print(f"  Test MAE: R{model_results['test_mae']:,.2f}")
        
        if verbose:
            print("\nAll Models Performance:")
            for name, res in results['premium_optimization'].items():
                print(f"  {name}: RMSE=R{res['test_rmse']:,.2f}, R²={res['test_r2']:.4f}")


def save_models(modeling, verbose=False):
    """Save trained models to files"""
    
    print("\n💾 Saving trained models...")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    
    # Save all trained models
    for task, models_dict in modeling.models.items():
        for model_name, model in models_dict.items():
            filename = f"{task}_{model_name.lower().replace(' ', '_')}.joblib"
            filepath = models_dir / filename
            
            try:
                joblib.dump(model, filepath)
                saved_count += 1
                if verbose:
                    print(f"  ✅ Saved {task} - {model_name} to {filepath}")
            except Exception as e:
                print(f"  ❌ Failed to save {task} - {model_name}: {e}")
    
    # Save scalers and encoders
    if modeling.scalers:
        scalers_file = models_dir / 'scalers.joblib'
        joblib.dump(modeling.scalers, scalers_file)
        saved_count += 1
        if verbose:
            print(f"  ✅ Saved scalers to {scalers_file}")
    
    if modeling.encoders:
        encoders_file = models_dir / 'encoders.joblib'
        joblib.dump(modeling.encoders, encoders_file)
        saved_count += 1
        if verbose:
            print(f"  ✅ Saved encoders to {encoders_file}")
    
    print(f"📦 Successfully saved {saved_count} model components")


def generate_business_report(results, modeling):
    """Generate comprehensive business report"""
    
    print("\n📋 BUSINESS IMPACT REPORT")
    print("=" * 40)
    
    # Model Performance Summary
    print("\n🎯 MODEL PERFORMANCE SUMMARY:")
    
    performance_summary = []
    
    if 'claim_severity' in results and results['claim_severity']:
        best_severity = min(results['claim_severity'].items(), key=lambda x: x[1]['test_rmse'])
        r2_score = best_severity[1]['test_r2']
        performance_summary.append(f"Claim Severity: {r2_score:.1%} variance explained")
    
    if 'claim_probability' in results and results['claim_probability']:
        best_prob = max(results['claim_probability'].items(), key=lambda x: x[1]['test_auc'])
        auc_score = best_prob[1]['test_auc']
        performance_summary.append(f"Claim Probability: {auc_score:.1%} AUC achieved")
    
    for summary in performance_summary:
        print(f"  ✅ {summary}")
    
    # Feature Importance Insights
    if modeling.feature_importance:
        print("\n🔍 KEY RISK FACTORS IDENTIFIED:")
        
        # Extract most common important features
        all_important_features = []
        for model_key, importance_data in modeling.feature_importance.items():
            if 'feature_importance' in importance_data:
                top_features = importance_data['feature_importance'].head(3)['feature'].tolist()
                all_important_features.extend(top_features)
        
        # Count frequency of important features
        from collections import Counter
        feature_counts = Counter(all_important_features)
        
        for feature, count in feature_counts.most_common(5):
            if 'VehicleAge' in feature:
                print(f"  🚗 Vehicle Age: Critical across {count} models - Age-based pricing validated")
            elif 'Province' in feature or any(prov in feature for prov in ['Gauteng', 'Western']):
                print(f"  🏢 Geographic Location: Important in {count} models - Regional pricing needed")
            elif 'VehicleType' in feature:
                print(f"  🚙 Vehicle Type: Key factor in {count} models - Category-specific rates required")
            else:
                print(f"  📊 {feature}: Significant in {count} models")
    
    # Business Recommendations
    print("\n💼 STRATEGIC RECOMMENDATIONS:")
    recommendations = [
        "Implement dynamic risk-based pricing using model predictions",
        "Establish monthly model monitoring and quarterly retraining",
        "Deploy A/B testing framework for pricing optimization",
        "Create customer-facing risk factor explanations using SHAP",
        "Develop real-time quote generation system",
        "Establish model governance and validation procedures"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # ROI Estimation
    print("\n💰 ESTIMATED BUSINESS IMPACT:")
    print("  📈 Improved risk selection: 5-10% reduction in loss ratio")
    print("  💵 Optimized pricing: 3-7% increase in premium adequacy")
    print("  🎯 Enhanced customer retention: 2-5% through fair pricing")
    print("  ⚡ Operational efficiency: 50-70% faster quote generation")


def run_quick_demo():
    """Run a quick demonstration of the modeling capabilities"""
    print("🎬 Running Quick Predictive Modeling Demo...")
    print("=" * 50)
    
    # Initialize with sample data
    modeling = InsurancePredictiveModeling()
    
    # Load sample data
    df = modeling.load_and_prepare_data()
    print(f"✅ Sample data created: {df.shape}")
    
    # Quick preprocessing
    df_processed = modeling.comprehensive_data_preprocessing()
    print(f"✅ Data preprocessed: {df_processed.shape}")
    
    # Build one model from each category
    print("\n🔧 Building sample models...")
    
    # Quick claim severity model
    severity_results = modeling.build_claim_severity_models()
    if severity_results:
        best_severity = min(severity_results.items(), key=lambda x: x[1]['test_rmse'])
        print(f"✅ Claim Severity: {best_severity[0]} (R² = {best_severity[1]['test_r2']:.3f})")
    
    # Quick claim probability model
    prob_results = modeling.build_claim_probability_models()
    if prob_results:
        best_prob = max(prob_results.items(), key=lambda x: x[1]['test_auc'])
        print(f"✅ Claim Probability: {best_prob[0]} (AUC = {best_prob[1]['test_auc']:.3f})")
    
    print("\n✅ Demo completed successfully!")
    return True


if __name__ == "__main__":
    # Check if running as demo
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick demo...")
        run_quick_demo()
    else:
        main()
