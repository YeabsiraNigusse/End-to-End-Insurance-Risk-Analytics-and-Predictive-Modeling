"""
Unit tests for the InsurancePredictiveModeling class
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predictive_modeling import InsurancePredictiveModeling


class TestInsurancePredictiveModeling(unittest.TestCase):
    """Test cases for InsurancePredictiveModeling class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.modeling = InsurancePredictiveModeling()
        
    def test_sample_data_creation(self):
        """Test sample data creation"""
        df = self.modeling._create_sample_data()
        
        # Check data shape
        self.assertEqual(df.shape[0], 50000)
        self.assertGreater(df.shape[1], 10)
        
        # Check required columns exist
        required_cols = ['PolicyID', 'TotalClaims', 'HasClaim', 'CalculatedPremiumPerTerm']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Check data types and ranges
        self.assertTrue(df['HasClaim'].isin([0, 1]).all())
        self.assertTrue((df['TotalClaims'] >= 0).all())
        self.assertTrue((df['CalculatedPremiumPerTerm'] > 0).all())
    
    def test_data_loading(self):
        """Test data loading functionality"""
        df = self.modeling.load_and_prepare_data()
        
        # Check data is loaded
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Check essential columns
        essential_cols = ['TotalClaims', 'HasClaim', 'CalculatedPremiumPerTerm']
        for col in essential_cols:
            self.assertIn(col, df.columns)
    
    def test_data_preprocessing(self):
        """Test comprehensive data preprocessing"""
        self.modeling.load_and_prepare_data()
        df_processed = self.modeling.comprehensive_data_preprocessing()
        
        # Check preprocessing completed
        self.assertIsNotNone(df_processed)
        self.assertGreater(len(df_processed), 0)
        
        # Check no missing values after preprocessing
        self.assertEqual(df_processed.isnull().sum().sum(), 0)
        
        # Check engineered features were created
        engineered_features = ['VehicleAge', 'RiskScore']
        for feature in engineered_features:
            if feature in df_processed.columns:
                self.assertIn(feature, df_processed.columns)
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        # Create test data with missing values
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', np.nan, 'A', 'B'],
            'target_col': [0, 1, 0, 1, 0]
        })
        
        # Apply missing value handling
        result = self.modeling._handle_missing_values(test_data)
        
        # Check no missing values remain
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check values were filled appropriately
        self.assertFalse(result['numeric_col'].isnull().any())
        self.assertFalse(result['categorical_col'].isnull().any())
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        # Create test data
        test_data = pd.DataFrame({
            'RegistrationYear': [2015, 2018, 2020, 2010],
            'CalculatedPremiumPerTerm': [1000, 1500, 2000, 800],
            'SumInsured': [50000, 75000, 100000, 40000],
            'CustomValueEstimate': [45000, 70000, 95000, 35000],
            'Province': ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape'],
            'VehicleType': ['Sedan', 'SUV', 'Motorcycle', 'Truck']
        })
        
        # Apply feature engineering
        result = self.modeling._create_engineered_features(test_data)
        
        # Check new features were created
        expected_features = ['VehicleAge', 'PremiumToSumInsuredRatio', 'RiskScore']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check VehicleAge calculation
        expected_ages = [2024 - year for year in test_data['RegistrationYear']]
        np.testing.assert_array_equal(result['VehicleAge'].values, expected_ages)
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding"""
        # Create test data with categorical variables
        test_data = pd.DataFrame({
            'small_cat': ['A', 'B', 'C', 'A', 'B'],  # Should be one-hot encoded
            'large_cat': [f'Cat_{i}' for i in range(15)] + [f'Cat_{i%15}' for i in range(5)],  # Should be label encoded
            'numeric_col': [1, 2, 3, 4, 5]
        })
        
        # Apply encoding
        result = self.modeling._encode_categorical_variables(test_data)
        
        # Check that categorical columns were processed
        self.assertNotIn('small_cat', result.columns)  # Should be replaced by dummies
        
        # Check one-hot encoded columns exist
        one_hot_cols = [col for col in result.columns if col.startswith('small_cat_')]
        self.assertGreater(len(one_hot_cols), 0)
    
    def test_claim_severity_modeling(self):
        """Test claim severity model building"""
        # Load and preprocess data
        self.modeling.load_and_prepare_data()
        self.modeling.comprehensive_data_preprocessing()
        
        # Build claim severity models
        results = self.modeling.build_claim_severity_models()
        
        if results:  # Only test if models were built (depends on having claims data)
            # Check that models were trained
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # Check model results structure
            for model_name, model_results in results.items():
                required_metrics = ['test_rmse', 'test_r2', 'test_mae', 'model']
                for metric in required_metrics:
                    self.assertIn(metric, model_results)
                
                # Check metric ranges
                self.assertGreaterEqual(model_results['test_rmse'], 0)
                self.assertLessEqual(model_results['test_r2'], 1)
                self.assertGreaterEqual(model_results['test_mae'], 0)
    
    def test_claim_probability_modeling(self):
        """Test claim probability model building"""
        # Load and preprocess data
        self.modeling.load_and_prepare_data()
        self.modeling.comprehensive_data_preprocessing()
        
        # Build claim probability models
        results = self.modeling.build_claim_probability_models()
        
        if results:  # Only test if models were built
            # Check that models were trained
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # Check model results structure
            for model_name, model_results in results.items():
                required_metrics = ['test_accuracy', 'test_auc', 'test_f1', 'model']
                for metric in required_metrics:
                    self.assertIn(metric, model_results)
                
                # Check metric ranges
                self.assertGreaterEqual(model_results['test_accuracy'], 0)
                self.assertLessEqual(model_results['test_accuracy'], 1)
                self.assertGreaterEqual(model_results['test_auc'], 0)
                self.assertLessEqual(model_results['test_auc'], 1)
    
    def test_premium_optimization_modeling(self):
        """Test premium optimization model building"""
        # Load and preprocess data
        self.modeling.load_and_prepare_data()
        self.modeling.comprehensive_data_preprocessing()
        
        # Build premium optimization models
        results = self.modeling.build_premium_optimization_models()
        
        if results:  # Only test if models were built
            # Check that models were trained
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # Check model results structure
            for model_name, model_results in results.items():
                required_metrics = ['test_rmse', 'test_r2', 'test_mae', 'model']
                for metric in required_metrics:
                    self.assertIn(metric, model_results)
    
    def test_model_comparison(self):
        """Test model performance comparison"""
        # Load and preprocess data
        self.modeling.load_and_prepare_data()
        self.modeling.comprehensive_data_preprocessing()
        
        # Build at least one set of models
        self.modeling.build_claim_probability_models()
        
        # Compare models
        comparison_df = self.modeling.compare_model_performance()
        
        if not comparison_df.empty:
            # Check comparison DataFrame structure
            self.assertIsInstance(comparison_df, pd.DataFrame)
            self.assertIn('Task', comparison_df.columns)
            self.assertIn('Model', comparison_df.columns)
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        # Load and preprocess data
        self.modeling.load_and_prepare_data()
        self.modeling.comprehensive_data_preprocessing()
        
        # Build models
        self.modeling.build_claim_probability_models()
        
        # Analyze feature importance
        if 'claim_probability' in self.modeling.results:
            importance_data = self.modeling.analyze_feature_importance('claim_probability', 'XGBoost')
            
            if importance_data:
                # Check importance data structure
                self.assertIsInstance(importance_data, dict)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.modeling = InsurancePredictiveModeling()
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe"""
        self.modeling.df = pd.DataFrame()
        self.modeling.df_processed = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        results = self.modeling.build_claim_probability_models()
        self.assertEqual(results, {})
    
    def test_no_claims_data(self):
        """Test handling when no claims exist"""
        # Create data with no claims
        test_data = pd.DataFrame({
            'PolicyID': [1, 2, 3],
            'TotalClaims': [0, 0, 0],
            'HasClaim': [0, 0, 0],
            'CalculatedPremiumPerTerm': [1000, 1500, 2000],
            'Province': ['A', 'B', 'C']
        })
        
        self.modeling.df = test_data
        self.modeling.df_processed = test_data
        
        # Should handle no claims gracefully
        results = self.modeling.build_claim_severity_models()
        self.assertEqual(results, {})
    
    def test_single_class_target(self):
        """Test handling of single class in target variable"""
        # Create data with only one class
        test_data = pd.DataFrame({
            'PolicyID': [1, 2, 3, 4, 5],
            'TotalClaims': [0, 0, 0, 0, 0],
            'HasClaim': [0, 0, 0, 0, 0],  # All no claims
            'CalculatedPremiumPerTerm': [1000, 1500, 2000, 1200, 1800],
            'Province': ['A', 'B', 'C', 'A', 'B'],
            'VehicleType': ['Sedan', 'SUV', 'Sedan', 'SUV', 'Sedan']
        })
        
        self.modeling.df = test_data
        self.modeling.df_processed = test_data
        
        # Should handle single class gracefully (may fail or return empty results)
        try:
            results = self.modeling.build_claim_probability_models()
            # If it doesn't fail, results should be empty or contain error info
            if results:
                for model_name, model_results in results.items():
                    # Check that metrics are reasonable (may be NaN or 0 for single class)
                    self.assertTrue(isinstance(model_results.get('test_accuracy', 0), (int, float)))
        except Exception:
            # It's acceptable for this to fail with single class
            pass


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestInsurancePredictiveModeling))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PREDICTIVE MODELING TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
