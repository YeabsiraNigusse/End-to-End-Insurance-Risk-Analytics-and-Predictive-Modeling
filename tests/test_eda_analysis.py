"""
Unit tests for the InsuranceEDA class
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../src')

from src.eda_analysis import InsuranceEDA


class TestInsuranceEDA(unittest.TestCase):
    """Test cases for InsuranceEDA class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.eda = InsuranceEDA()
        
    def test_sample_data_creation(self):
        """Test sample data creation"""
        df = self.eda._create_sample_data()
        
        # Check data shape
        self.assertEqual(df.shape[0], 10000)
        self.assertGreater(df.shape[1], 5)
        
        # Check required columns exist
        required_cols = ['PolicyID', 'TotalPremium', 'TotalClaims']
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_data_loading(self):
        """Test data loading functionality"""
        df = self.eda.load_data()
        
        # Check data is loaded
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Check loss ratio calculation
        self.assertIn('LossRatio', df.columns)
    
    def test_column_identification(self):
        """Test column type identification"""
        self.eda.load_data()
        
        # Check column lists are populated
        self.assertGreater(len(self.eda.numerical_cols), 0)
        self.assertGreater(len(self.eda.categorical_cols), 0)
    
    def test_data_overview(self):
        """Test data overview functionality"""
        self.eda.load_data()
        overview = self.eda.data_overview()
        
        # Check overview contains expected keys
        expected_keys = ['shape', 'numerical_columns', 'categorical_columns']
        for key in expected_keys:
            self.assertIn(key, overview)
    
    def test_missing_value_analysis(self):
        """Test missing value analysis"""
        self.eda.load_data()
        missing_stats = self.eda.missing_value_analysis()
        
        # Check return type
        self.assertIsInstance(missing_stats, pd.DataFrame)
        
        # Check required columns
        expected_cols = ['Column', 'Missing_Count', 'Missing_Percentage']
        for col in expected_cols:
            self.assertIn(col, missing_stats.columns)
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics"""
        self.eda.load_data()
        stats = self.eda.descriptive_statistics()
        
        # Check return type
        self.assertIsInstance(stats, dict)
        
        # Check for numerical stats if numerical columns exist
        if self.eda.numerical_cols:
            self.assertIn('numerical', stats)
    
    def test_loss_ratio_analysis(self):
        """Test loss ratio analysis"""
        self.eda.load_data()
        loss_analysis = self.eda.loss_ratio_analysis()
        
        # Check return type
        self.assertIsInstance(loss_analysis, dict)
        
        # Check overall loss ratio is calculated
        if 'overall_loss_ratio' in loss_analysis:
            self.assertIsInstance(loss_analysis['overall_loss_ratio'], (int, float))
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        self.eda.load_data()
        outlier_info = self.eda.outlier_detection()
        
        # Check return type
        self.assertIsInstance(outlier_info, dict)
        
        # Check outlier info structure
        for var, info in outlier_info.items():
            self.assertIn('iqr_outliers', info)
            self.assertIn('iqr_percentage', info)
    
    def test_statistical_testing(self):
        """Test statistical testing functionality"""
        self.eda.load_data()
        test_results = self.eda.statistical_testing()
        
        # Check return type
        self.assertIsInstance(test_results, dict)
    
    def test_insights_generation(self):
        """Test insights generation"""
        self.eda.load_data()
        insights = self.eda.generate_insights()
        
        # Check return type
        self.assertIsInstance(insights, list)
        
        # Check insights are generated
        self.assertGreater(len(insights), 0)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.eda = InsuranceEDA()
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        self.eda.df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        overview = self.eda.data_overview()
        self.assertEqual(overview['shape'], (0, 0))
    
    def test_missing_columns(self):
        """Test handling of missing key columns"""
        # Create dataframe without key columns
        self.eda.df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        self.eda._identify_column_types()
        
        # Should handle missing columns gracefully
        loss_analysis = self.eda.loss_ratio_analysis()
        self.assertEqual(loss_analysis, {})
    
    def test_data_types(self):
        """Test data type handling"""
        self.eda.load_data()
        
        # Check numerical columns are numeric
        for col in self.eda.numerical_cols:
            if col in self.eda.df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(self.eda.df[col]))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestInsuranceEDA))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
