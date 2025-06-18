"""
Main hypothesis testing module for insurance risk analytics.

This module conducts statistical hypothesis testing to validate or reject
key hypotheses about risk drivers in insurance data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
from data_preprocessing import InsuranceDataProcessor
from statistical_utils import StatisticalTester

class InsuranceHypothesisTester:
    """
    Main class for conducting hypothesis tests on insurance risk data.
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None, alpha: float = 0.05):
        """
        Initialize the hypothesis tester.
        
        Args:
            data_path: Path to insurance data file
            df: Pre-loaded DataFrame
            alpha: Significance level for tests
        """
        self.processor = InsuranceDataProcessor(data_path=data_path, df=df)
        self.tester = StatisticalTester(alpha=alpha)
        self.alpha = alpha
        self.results = {}
        
    def run_all_hypothesis_tests(self) -> Dict[str, Any]:
        """
        Run all four hypothesis tests and return comprehensive results.
        
        Returns:
            Dictionary containing all test results and interpretations
        """
        print("Starting comprehensive hypothesis testing...")
        
        # Prepare data for testing
        hypothesis_data = self.processor.prepare_hypothesis_data()
        
        # Run each hypothesis test
        self.results['H1_provinces'] = self.test_province_risk_differences(hypothesis_data['provinces'])
        self.results['H2_zip_codes'] = self.test_zip_code_risk_differences(hypothesis_data['zip_codes'])
        self.results['H3_zip_margin'] = self.test_zip_code_margin_differences(hypothesis_data['zip_codes_margin'])
        self.results['H4_gender'] = self.test_gender_risk_differences(hypothesis_data['gender'])
        
        # Generate summary report
        self.results['summary'] = self._generate_summary_report()
        
        print("All hypothesis tests completed!")
        return self.results
    
    def test_province_risk_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences across provinces.
        
        Args:
            data: DataFrame with province and risk data
            
        Returns:
            Dictionary with test results
        """
        print("Testing H1: Risk differences across provinces...")
        
        results = {
            'hypothesis': 'H₀: There are no risk differences across provinces',
            'tests': {}
        }
        
        # Test 1: Claim Frequency differences (Chi-square test)
        results['tests']['claim_frequency'] = self.tester.chi_square_test(
            data['Province'], data['HasClaim']
        )
        
        # Test 2: Claim Severity differences (ANOVA/Kruskal-Wallis)
        severity_data = data[data['HasClaim'] == 1]  # Only policies with claims
        if len(severity_data) > 0:
            province_groups = [
                severity_data[severity_data['Province'] == province]['ClaimSeverity']
                for province in severity_data['Province'].unique()
            ]
            
            # Check normality assumption (simplified)
            if self._check_normality_assumption(province_groups):
                results['tests']['claim_severity'] = self.tester.one_way_anova(province_groups)
            else:
                results['tests']['claim_severity'] = self.tester.kruskal_wallis_test(province_groups)
        
        # Test 3: Loss Ratio differences
        loss_ratio_groups = [
            data[data['Province'] == province]['LossRatio']
            for province in data['Province'].unique()
        ]
        
        if self._check_normality_assumption(loss_ratio_groups):
            results['tests']['loss_ratio'] = self.tester.one_way_anova(loss_ratio_groups)
        else:
            results['tests']['loss_ratio'] = self.tester.kruskal_wallis_test(loss_ratio_groups)
        
        # Overall conclusion
        results['conclusion'] = self._conclude_hypothesis_test(results['tests'])
        
        return results
    
    def test_zip_code_risk_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences between zip codes.
        
        Args:
            data: DataFrame with zip code and risk data
            
        Returns:
            Dictionary with test results
        """
        print("Testing H2: Risk differences between zip codes...")
        
        # For zip codes, we'll sample top zip codes by volume to make analysis manageable
        top_zip_codes = data['PostalCode'].value_counts().head(10).index.tolist()
        zip_data = data[data['PostalCode'].isin(top_zip_codes)]
        
        results = {
            'hypothesis': 'H₀: There are no risk differences between zip codes',
            'note': f'Analysis limited to top {len(top_zip_codes)} zip codes by volume',
            'zip_codes_analyzed': top_zip_codes,
            'tests': {}
        }
        
        # Test 1: Claim Frequency differences
        results['tests']['claim_frequency'] = self.tester.chi_square_test(
            zip_data['PostalCode'], zip_data['HasClaim']
        )
        
        # Test 2: Claim Severity differences
        severity_data = zip_data[zip_data['HasClaim'] == 1]
        if len(severity_data) > 0:
            zip_groups = [
                severity_data[severity_data['PostalCode'] == zip_code]['ClaimSeverity']
                for zip_code in top_zip_codes
                if len(severity_data[severity_data['PostalCode'] == zip_code]) > 0
            ]
            
            if len(zip_groups) > 1:
                if self._check_normality_assumption(zip_groups):
                    results['tests']['claim_severity'] = self.tester.one_way_anova(zip_groups)
                else:
                    results['tests']['claim_severity'] = self.tester.kruskal_wallis_test(zip_groups)
        
        # Test 3: Loss Ratio differences
        loss_ratio_groups = [
            zip_data[zip_data['PostalCode'] == zip_code]['LossRatio']
            for zip_code in top_zip_codes
        ]
        
        if self._check_normality_assumption(loss_ratio_groups):
            results['tests']['loss_ratio'] = self.tester.one_way_anova(loss_ratio_groups)
        else:
            results['tests']['loss_ratio'] = self.tester.kruskal_wallis_test(loss_ratio_groups)
        
        results['conclusion'] = self._conclude_hypothesis_test(results['tests'])
        
        return results
    
    def test_zip_code_margin_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no significant margin (profit) differences between zip codes.
        
        Args:
            data: DataFrame with zip code and margin data
            
        Returns:
            Dictionary with test results
        """
        print("Testing H3: Margin differences between zip codes...")
        
        # Use same top zip codes approach
        top_zip_codes = data['PostalCode'].value_counts().head(10).index.tolist()
        zip_data = data[data['PostalCode'].isin(top_zip_codes)]
        
        results = {
            'hypothesis': 'H₀: There are no significant margin (profit) differences between zip codes',
            'note': f'Analysis limited to top {len(top_zip_codes)} zip codes by volume',
            'zip_codes_analyzed': top_zip_codes,
            'tests': {}
        }
        
        # Test margin differences
        margin_groups = [
            zip_data[zip_data['PostalCode'] == zip_code]['Margin']
            for zip_code in top_zip_codes
        ]
        
        if self._check_normality_assumption(margin_groups):
            results['tests']['margin'] = self.tester.one_way_anova(margin_groups)
        else:
            results['tests']['margin'] = self.tester.kruskal_wallis_test(margin_groups)
        
        results['conclusion'] = self._conclude_hypothesis_test(results['tests'])
        
        return results
    
    def test_gender_risk_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no significant risk differences between Women and Men.
        
        Args:
            data: DataFrame with gender and risk data
            
        Returns:
            Dictionary with test results
        """
        print("Testing H4: Risk differences between genders...")
        
        # Filter for Male and Female only
        gender_data = data[data['Gender'].isin(['Male', 'Female'])]
        
        results = {
            'hypothesis': 'H₀: There are no significant risk differences between Women and Men',
            'tests': {}
        }
        
        if len(gender_data) == 0:
            results['error'] = 'No Male/Female data available'
            return results
        
        # Test 1: Claim Frequency differences
        results['tests']['claim_frequency'] = self.tester.chi_square_test(
            gender_data['Gender'], gender_data['HasClaim']
        )
        
        # Test 2: Claim Severity differences
        male_severity = gender_data[
            (gender_data['Gender'] == 'Male') & (gender_data['HasClaim'] == 1)
        ]['ClaimSeverity']
        female_severity = gender_data[
            (gender_data['Gender'] == 'Female') & (gender_data['HasClaim'] == 1)
        ]['ClaimSeverity']
        
        if len(male_severity) > 0 and len(female_severity) > 0:
            # Check normality
            if self._check_normality_simple([male_severity, female_severity]):
                results['tests']['claim_severity'] = self.tester.two_sample_t_test(
                    male_severity, female_severity
                )
            else:
                results['tests']['claim_severity'] = self.tester.mann_whitney_u_test(
                    male_severity, female_severity
                )
        
        # Test 3: Loss Ratio differences
        male_loss_ratio = gender_data[gender_data['Gender'] == 'Male']['LossRatio']
        female_loss_ratio = gender_data[gender_data['Gender'] == 'Female']['LossRatio']
        
        if self._check_normality_simple([male_loss_ratio, female_loss_ratio]):
            results['tests']['loss_ratio'] = self.tester.two_sample_t_test(
                male_loss_ratio, female_loss_ratio
            )
        else:
            results['tests']['loss_ratio'] = self.tester.mann_whitney_u_test(
                male_loss_ratio, female_loss_ratio
            )
        
        # Test 4: Margin differences
        male_margin = gender_data[gender_data['Gender'] == 'Male']['Margin']
        female_margin = gender_data[gender_data['Gender'] == 'Female']['Margin']
        
        if self._check_normality_simple([male_margin, female_margin]):
            results['tests']['margin'] = self.tester.two_sample_t_test(
                male_margin, female_margin
            )
        else:
            results['tests']['margin'] = self.tester.mann_whitney_u_test(
                male_margin, female_margin
            )
        
        results['conclusion'] = self._conclude_hypothesis_test(results['tests'])
        
        return results
    
    def _check_normality_assumption(self, groups: List[pd.Series]) -> bool:
        """
        Simple check for normality assumption (for choosing parametric vs non-parametric tests).
        
        Args:
            groups: List of data groups
            
        Returns:
            True if normality assumption is reasonable
        """
        # Simplified approach: check if all groups have reasonable size and skewness
        for group in groups:
            if len(group) < 30:  # Small sample size
                return False
            if abs(group.skew()) > 2:  # Highly skewed
                return False
        return True
    
    def _check_normality_simple(self, groups: List[pd.Series]) -> bool:
        """Simple normality check for two-group comparisons."""
        return self._check_normality_assumption(groups)
    
    def _conclude_hypothesis_test(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate conclusion for a hypothesis test based on multiple sub-tests.
        
        Args:
            tests: Dictionary of test results
            
        Returns:
            Dictionary with conclusion
        """
        significant_tests = []
        non_significant_tests = []
        
        for test_name, test_result in tests.items():
            if test_result.get('significant', False):
                significant_tests.append(test_name)
            else:
                non_significant_tests.append(test_name)
        
        if len(significant_tests) > 0:
            decision = "REJECT NULL HYPOTHESIS"
            evidence = f"Significant differences found in: {', '.join(significant_tests)}"
        else:
            decision = "FAIL TO REJECT NULL HYPOTHESIS"
            evidence = "No significant differences found in any test"
        
        return {
            'decision': decision,
            'evidence': evidence,
            'significant_tests': significant_tests,
            'non_significant_tests': non_significant_tests,
            'total_tests': len(tests)
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all hypothesis tests."""
        summary = {
            'total_hypotheses_tested': 4,
            'rejected_hypotheses': [],
            'failed_to_reject_hypotheses': [],
            'business_recommendations': []
        }
        
        # Analyze each hypothesis
        for hypothesis_key, result in self.results.items():
            if hypothesis_key == 'summary':
                continue
                
            if result.get('conclusion', {}).get('decision') == "REJECT NULL HYPOTHESIS":
                summary['rejected_hypotheses'].append({
                    'hypothesis': hypothesis_key,
                    'description': result.get('hypothesis', ''),
                    'evidence': result.get('conclusion', {}).get('evidence', '')
                })
            else:
                summary['failed_to_reject_hypotheses'].append({
                    'hypothesis': hypothesis_key,
                    'description': result.get('hypothesis', ''),
                    'evidence': result.get('conclusion', {}).get('evidence', '')
                })
        
        # Generate business recommendations
        summary['business_recommendations'] = self._generate_business_recommendations()
        
        return summary
    
    def _generate_business_recommendations(self) -> List[str]:
        """Generate business recommendations based on test results."""
        recommendations = []
        
        # Check each hypothesis for business implications
        if 'H1_provinces' in self.results:
            if self.results['H1_provinces'].get('conclusion', {}).get('decision') == "REJECT NULL HYPOTHESIS":
                recommendations.append(
                    "Consider implementing province-specific pricing strategies due to significant risk variations across provinces."
                )
        
        if 'H2_zip_codes' in self.results:
            if self.results['H2_zip_codes'].get('conclusion', {}).get('decision') == "REJECT NULL HYPOTHESIS":
                recommendations.append(
                    "Implement zip code-level risk adjustments in pricing models to account for geographic risk variations."
                )
        
        if 'H3_zip_margin' in self.results:
            if self.results['H3_zip_margin'].get('conclusion', {}).get('decision') == "REJECT NULL HYPOTHESIS":
                recommendations.append(
                    "Review pricing strategy by zip code to optimize profit margins and ensure competitive positioning."
                )
        
        if 'H4_gender' in self.results:
            if self.results['H4_gender'].get('conclusion', {}).get('decision') == "REJECT NULL HYPOTHESIS":
                recommendations.append(
                    "Consider gender-based risk factors in underwriting (subject to regulatory compliance and fairness considerations)."
                )
        
        if not recommendations:
            recommendations.append(
                "Current risk segmentation appears adequate. Consider exploring other risk factors for potential improvements."
            )
        
        return recommendations
