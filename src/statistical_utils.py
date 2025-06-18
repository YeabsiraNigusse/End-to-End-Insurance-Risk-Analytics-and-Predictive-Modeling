"""
Statistical utility functions for hypothesis testing in insurance risk analytics.

This module provides statistical test functions and utilities for conducting
A/B hypothesis testing on insurance data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, f_oneway, kruskal
from typing import Tuple, Dict, List, Any
import warnings

class StatisticalTester:
    """
    A class to perform various statistical tests for insurance risk analysis.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical tester.
        
        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        
    def chi_square_test(self, group_col: pd.Series, outcome_col: pd.Series) -> Dict[str, Any]:
        """
        Perform chi-square test for independence between categorical variables.
        
        Args:
            group_col: Grouping variable (e.g., Province, Gender)
            outcome_col: Binary outcome variable (e.g., HasClaim)
            
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        contingency_table = pd.crosstab(group_col, outcome_col)
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        result = {
            'test_name': 'Chi-Square Test of Independence',
            'statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size': cramers_v,
            'contingency_table': contingency_table,
            'expected_frequencies': expected,
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_chi_square(p_value, cramers_v)
        }
        
        return result
    
    def two_sample_t_test(self, group1: pd.Series, group2: pd.Series, 
                         equal_var: bool = False) -> Dict[str, Any]:
        """
        Perform two-sample t-test for comparing means between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            equal_var: Whether to assume equal variances
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        group1_clean = group1.dropna()
        group2_clean = group2.dropna()
        
        # Perform t-test
        t_stat, p_value = ttest_ind(group1_clean, group2_clean, equal_var=equal_var)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_clean) - 1) * group1_clean.var() + 
                             (len(group2_clean) - 1) * group2_clean.var()) / 
                            (len(group1_clean) + len(group2_clean) - 2))
        cohens_d = (group1_clean.mean() - group2_clean.mean()) / pooled_std
        
        result = {
            'test_name': 'Two-Sample T-Test',
            'statistic': t_stat,
            'p_value': p_value,
            'effect_size': cohens_d,
            'group1_mean': group1_clean.mean(),
            'group2_mean': group2_clean.mean(),
            'group1_std': group1_clean.std(),
            'group2_std': group2_clean.std(),
            'group1_n': len(group1_clean),
            'group2_n': len(group2_clean),
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_t_test(p_value, cohens_d)
        }
        
        return result
    
    def mann_whitney_u_test(self, group1: pd.Series, group2: pd.Series) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        group1_clean = group1.dropna()
        group2_clean = group2.dropna()
        
        # Perform Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1_clean), len(group2_clean)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        result = {
            'test_name': 'Mann-Whitney U Test',
            'statistic': u_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'group1_median': group1_clean.median(),
            'group2_median': group2_clean.median(),
            'group1_n': n1,
            'group2_n': n2,
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_mann_whitney(p_value, effect_size)
        }
        
        return result
    
    def one_way_anova(self, groups: List[pd.Series]) -> Dict[str, Any]:
        """
        Perform one-way ANOVA for comparing means across multiple groups.
        
        Args:
            groups: List of group data series
            
        Returns:
            Dictionary with test results
        """
        # Clean groups and remove NaN values
        clean_groups = [group.dropna() for group in groups]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*clean_groups)
        
        # Calculate effect size (eta-squared)
        group_means = [group.mean() for group in clean_groups]
        group_sizes = [len(group) for group in clean_groups]
        overall_mean = np.concatenate(clean_groups).mean()
        
        ss_between = sum(n * (mean - overall_mean)**2 for n, mean in zip(group_sizes, group_means))
        ss_total = sum((np.concatenate(clean_groups) - overall_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            'test_name': 'One-Way ANOVA',
            'statistic': f_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'group_means': group_means,
            'group_sizes': group_sizes,
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_anova(p_value, eta_squared)
        }
        
        return result
    
    def kruskal_wallis_test(self, groups: List[pd.Series]) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis test (non-parametric alternative to ANOVA).
        
        Args:
            groups: List of group data series
            
        Returns:
            Dictionary with test results
        """
        # Clean groups and remove NaN values
        clean_groups = [group.dropna() for group in groups]
        
        # Perform Kruskal-Wallis test
        h_stat, p_value = kruskal(*clean_groups)
        
        # Calculate effect size (epsilon-squared)
        n_total = sum(len(group) for group in clean_groups)
        epsilon_squared = (h_stat - len(clean_groups) + 1) / (n_total - len(clean_groups))
        
        result = {
            'test_name': 'Kruskal-Wallis Test',
            'statistic': h_stat,
            'p_value': p_value,
            'effect_size': epsilon_squared,
            'group_medians': [group.median() for group in clean_groups],
            'group_sizes': [len(group) for group in clean_groups],
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_kruskal_wallis(p_value, epsilon_squared)
        }
        
        return result
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if cramers_v < 0.1:
            effect = "negligible"
        elif cramers_v < 0.3:
            effect = "small"
        elif cramers_v < 0.5:
            effect = "medium"
        else:
            effect = "large"
            
        return f"The association is {significance} (p={p_value:.4f}) with a {effect} effect size (Cramér's V={cramers_v:.4f})"
    
    def _interpret_t_test(self, p_value: float, cohens_d: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"
            
        return f"The difference is {significance} (p={p_value:.4f}) with a {effect} effect size (Cohen's d={cohens_d:.4f})"
    
    def _interpret_mann_whitney(self, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            effect = "negligible"
        elif abs_effect < 0.3:
            effect = "small"
        elif abs_effect < 0.5:
            effect = "medium"
        else:
            effect = "large"
            
        return f"The difference is {significance} (p={p_value:.4f}) with a {effect} effect size (r={effect_size:.4f})"
    
    def _interpret_anova(self, p_value: float, eta_squared: float) -> str:
        """Interpret ANOVA results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if eta_squared < 0.01:
            effect = "negligible"
        elif eta_squared < 0.06:
            effect = "small"
        elif eta_squared < 0.14:
            effect = "medium"
        else:
            effect = "large"
            
        return f"The group differences are {significance} (p={p_value:.4f}) with a {effect} effect size (η²={eta_squared:.4f})"
    
    def _interpret_kruskal_wallis(self, p_value: float, epsilon_squared: float) -> str:
        """Interpret Kruskal-Wallis test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if epsilon_squared < 0.01:
            effect = "negligible"
        elif epsilon_squared < 0.06:
            effect = "small"
        elif epsilon_squared < 0.14:
            effect = "medium"
        else:
            effect = "large"
            
        return f"The group differences are {significance} (p={p_value:.4f}) with a {effect} effect size (ε²={epsilon_squared:.4f})"
