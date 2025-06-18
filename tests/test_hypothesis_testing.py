"""
Unit tests for hypothesis testing modules.

This module contains comprehensive tests for the insurance risk analytics
hypothesis testing functionality.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import InsuranceDataProcessor
from statistical_utils import StatisticalTester
from hypothesis_testing import InsuranceHypothesisTester

class TestInsuranceDataProcessor(unittest.TestCase):
    """Test cases for InsuranceDataProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'PolicyID': range(1, 1001),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], 1000),
            'PostalCode': np.random.choice(range(1000, 9999), 1000),
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'TotalPremium': np.random.exponential(100, 1000),
            'TotalClaims': np.random.exponential(80, 1000) * np.random.binomial(1, 0.3, 1000)
        })
        
        self.processor = InsuranceDataProcessor(df=self.sample_data)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.df)
        self.assertEqual(self.processor.original_shape, (1000, 6))
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        cleaned_df = self.processor.clean_data()
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertGreater(len(cleaned_df), 0)
    
    def test_create_risk_metrics(self):
        """Test risk metrics creation."""
        metrics_df = self.processor.create_risk_metrics()
        
        # Check if new columns are created
        expected_columns = ['HasClaim', 'ClaimSeverity', 'Margin', 'LossRatio']
        for col in expected_columns:
            self.assertIn(col, metrics_df.columns)
        
        # Check HasClaim is binary
        self.assertTrue(metrics_df['HasClaim'].isin([0, 1]).all())
        
        # Check Margin calculation
        expected_margin = metrics_df['TotalPremium'] - metrics_df['TotalClaims']
        pd.testing.assert_series_equal(metrics_df['Margin'], expected_margin, check_names=False)
    
    def test_prepare_hypothesis_data(self):
        """Test hypothesis data preparation."""
        hypothesis_data = self.processor.prepare_hypothesis_data()
        
        expected_keys = ['provinces', 'zip_codes', 'zip_codes_margin', 'gender']
        for key in expected_keys:
            self.assertIn(key, hypothesis_data)
            self.assertIsInstance(hypothesis_data[key], pd.DataFrame)
    
    def test_sample_data_for_testing(self):
        """Test data sampling functionality."""
        sampled_df = self.processor.sample_data_for_testing(sample_size=500)
        self.assertEqual(len(sampled_df), 500)
        
        # Test when sample size is larger than data
        large_sample = self.processor.sample_data_for_testing(sample_size=2000)
        self.assertEqual(len(large_sample), 1000)  # Should return full dataset

class TestStatisticalTester(unittest.TestCase):
    """Test cases for StatisticalTester class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.tester = StatisticalTester(alpha=0.05)
        
        # Create test data
        self.group1 = pd.Series(np.random.normal(10, 2, 100))
        self.group2 = pd.Series(np.random.normal(12, 2, 100))
        self.categorical_group = pd.Series(['A'] * 50 + ['B'] * 50)
        self.binary_outcome = pd.Series([0] * 30 + [1] * 20 + [0] * 35 + [1] * 15)
    
    def test_chi_square_test(self):
        """Test chi-square test functionality."""
        result = self.tester.chi_square_test(self.categorical_group, self.binary_outcome)
        
        # Check result structure
        expected_keys = ['test_name', 'statistic', 'p_value', 'effect_size', 
                        'contingency_table', 'significant', 'interpretation']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types
        self.assertIsInstance(result['statistic'], (int, float))
        self.assertIsInstance(result['p_value'], (int, float))
        self.assertIsInstance(result['significant'], bool)
    
    def test_two_sample_t_test(self):
        """Test two-sample t-test functionality."""
        result = self.tester.two_sample_t_test(self.group1, self.group2)
        
        # Check result structure
        expected_keys = ['test_name', 'statistic', 'p_value', 'effect_size',
                        'group1_mean', 'group2_mean', 'significant', 'interpretation']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that means are calculated correctly
        self.assertAlmostEqual(result['group1_mean'], self.group1.mean(), places=5)
        self.assertAlmostEqual(result['group2_mean'], self.group2.mean(), places=5)
    
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test functionality."""
        result = self.tester.mann_whitney_u_test(self.group1, self.group2)
        
        # Check result structure
        expected_keys = ['test_name', 'statistic', 'p_value', 'effect_size',
                        'group1_median', 'group2_median', 'significant', 'interpretation']
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_one_way_anova(self):
        """Test one-way ANOVA functionality."""
        group3 = pd.Series(np.random.normal(8, 2, 100))
        groups = [self.group1, self.group2, group3]
        
        result = self.tester.one_way_anova(groups)
        
        # Check result structure
        expected_keys = ['test_name', 'statistic', 'p_value', 'effect_size',
                        'group_means', 'group_sizes', 'significant', 'interpretation']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that group means are calculated correctly
        self.assertEqual(len(result['group_means']), 3)
        self.assertEqual(len(result['group_sizes']), 3)

class TestInsuranceHypothesisTester(unittest.TestCase):
    """Test cases for InsuranceHypothesisTester class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create more comprehensive test data
        self.sample_data = pd.DataFrame({
            'PolicyID': range(1, 1001),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], 1000),
            'PostalCode': np.random.choice(range(1000, 1010), 1000),  # Limited zip codes for testing
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'TotalPremium': np.random.exponential(100, 1000),
            'TotalClaims': np.random.exponential(80, 1000) * np.random.binomial(1, 0.3, 1000)
        })
        
        self.hypothesis_tester = InsuranceHypothesisTester(df=self.sample_data, alpha=0.05)
    
    def test_initialization(self):
        """Test hypothesis tester initialization."""
        self.assertIsNotNone(self.hypothesis_tester.processor)
        self.assertIsNotNone(self.hypothesis_tester.tester)
        self.assertEqual(self.hypothesis_tester.alpha, 0.05)
    
    def test_province_risk_differences(self):
        """Test province risk differences testing."""
        hypothesis_data = self.hypothesis_tester.processor.prepare_hypothesis_data()
        result = self.hypothesis_tester.test_province_risk_differences(hypothesis_data['provinces'])
        
        # Check result structure
        self.assertIn('hypothesis', result)
        self.assertIn('tests', result)
        self.assertIn('conclusion', result)
        
        # Check that tests were performed
        self.assertGreater(len(result['tests']), 0)
    
    def test_gender_risk_differences(self):
        """Test gender risk differences testing."""
        hypothesis_data = self.hypothesis_tester.processor.prepare_hypothesis_data()
        result = self.hypothesis_tester.test_gender_risk_differences(hypothesis_data['gender'])
        
        # Check result structure
        self.assertIn('hypothesis', result)
        self.assertIn('tests', result)
        self.assertIn('conclusion', result)
    
    def test_run_all_hypothesis_tests(self):
        """Test running all hypothesis tests."""
        results = self.hypothesis_tester.run_all_hypothesis_tests()
        
        # Check that all hypothesis tests are included
        expected_hypotheses = ['H1_provinces', 'H2_zip_codes', 'H3_zip_margin', 'H4_gender', 'summary']
        for hypothesis in expected_hypotheses:
            self.assertIn(hypothesis, results)
        
        # Check summary structure
        summary = results['summary']
        expected_summary_keys = ['total_hypotheses_tested', 'rejected_hypotheses', 
                               'failed_to_reject_hypotheses', 'business_recommendations']
        for key in expected_summary_keys:
            self.assertIn(key, summary)
    
    def test_normality_assumption_check(self):
        """Test normality assumption checking."""
        # Test with normal data
        normal_groups = [pd.Series(np.random.normal(0, 1, 100)) for _ in range(3)]
        self.assertTrue(self.hypothesis_tester._check_normality_assumption(normal_groups))
        
        # Test with small sample
        small_groups = [pd.Series(np.random.normal(0, 1, 10)) for _ in range(3)]
        self.assertFalse(self.hypothesis_tester._check_normality_assumption(small_groups))
        
        # Test with highly skewed data
        skewed_groups = [pd.Series(np.random.exponential(1, 100)) for _ in range(3)]
        # This might be True or False depending on the specific random data, so we just check it runs
        result = self.hypothesis_tester._check_normality_assumption(skewed_groups)
        self.assertIsInstance(result, bool)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete hypothesis testing workflow."""
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'PolicyID': range(1, 501),
            'Province': np.random.choice(['Gauteng', 'Western Cape'], 500),
            'PostalCode': np.random.choice(range(1000, 1005), 500),
            'Gender': np.random.choice(['Male', 'Female'], 500),
            'TotalPremium': np.random.exponential(100, 500),
            'TotalClaims': np.random.exponential(80, 500) * np.random.binomial(1, 0.3, 500)
        })
        
        # Run complete analysis
        hypothesis_tester = InsuranceHypothesisTester(df=sample_data, alpha=0.05)
        results = hypothesis_tester.run_all_hypothesis_tests()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('summary', results)
        
        # Verify that we can extract business recommendations
        summary = results['summary']
        self.assertIn('business_recommendations', summary)
        self.assertIsInstance(summary['business_recommendations'], list)

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestInsuranceDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestStatisticalTester))
    test_suite.addTest(unittest.makeSuite(TestInsuranceHypothesisTester))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
