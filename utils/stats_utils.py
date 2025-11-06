"""
Statistical utilities for GCaMP analysis.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats


def compute_cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of values
    group2 : np.ndarray
        Second group of values
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Hedges' g effect size (bias-corrected Cohen's d).
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of values
    group2 : np.ndarray
        Second group of values
        
    Returns
    -------
    float
        Hedges' g effect size
    """
    d = compute_cohen_d(group1, group2)
    n = len(group1) + len(group2)
    correction = 1 - (3 / (4 * n - 9))
    return d * correction


def perform_permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 1000,
    statistic: str = 'mean_diff'
) -> Tuple[float, float]:
    """
    Perform a permutation test to compare two groups.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of values
    group2 : np.ndarray
        Second group of values
    n_permutations : int
        Number of permutations to perform
    statistic : str
        Statistic to use ('mean_diff' or 'median_diff')
        
    Returns
    -------
    Tuple[float, float]
        (observed statistic, p-value)
    """
    # Compute observed statistic
    if statistic == 'mean_diff':
        obs_stat = np.mean(group1) - np.mean(group2)
    elif statistic == 'median_diff':
        obs_stat = np.median(group1) - np.median(group2)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Combine groups
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutation test
    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        if statistic == 'mean_diff':
            perm_stat = np.mean(perm_group1) - np.mean(perm_group2)
        else:
            perm_stat = np.median(perm_group1) - np.median(perm_group2)
        
        perm_stats.append(perm_stat)
    
    # Compute p-value (two-tailed)
    perm_stats = np.array(perm_stats)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    
    return obs_stat, p_value


def compute_bootstrap_ci(
    data: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    statistic: str = 'mean'
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    n_iterations : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    statistic : str
        Statistic to compute ('mean', 'median', 'std')
        
    Returns
    -------
    Tuple[float, float, float]
        (point estimate, lower bound, upper bound)
    """
    # Select statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'std':
        stat_func = np.std
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Compute point estimate
    point_est = stat_func(data)
    
    # Bootstrap
    boot_stats = []
    n = len(data)
    for _ in range(n_iterations):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(boot_sample))
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(boot_stats, lower_percentile)
    ci_upper = np.percentile(boot_stats, upper_percentile)
    
    return point_est, ci_lower, ci_upper


def compare_distributions(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = 'ttest'
) -> Tuple[float, float]:
    """
    Compare two distributions using various statistical tests.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of values
    group2 : np.ndarray
        Second group of values
    test : str
        Test to use ('ttest', 'mannwhitneyu', 'ks')
        
    Returns
    -------
    Tuple[float, float]
        (test statistic, p-value)
    """
    if test == 'ttest':
        stat, pval = stats.ttest_ind(group1, group2)
    elif test == 'mannwhitneyu':
        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    elif test == 'ks':
        stat, pval = stats.ks_2samp(group1, group2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return stat, pval


def multiple_comparison_correction(
    pvalues: np.ndarray,
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple comparison correction to p-values.
    
    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values
    method : str
        Correction method ('bonferroni', 'fdr_bh')
    alpha : float
        Significance level
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (corrected p-values, reject null hypothesis boolean array)
    """
    n = len(pvalues)
    
    if method == 'bonferroni':
        corrected_pvals = pvalues * n
        corrected_pvals = np.minimum(corrected_pvals, 1.0)
        reject = corrected_pvals < alpha
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        # Compute critical values
        critical_values = alpha * (np.arange(n) + 1) / n
        
        # Find largest i where p_i <= alpha * i / n
        reject_sorted = sorted_pvals <= critical_values
        
        if np.any(reject_sorted):
            max_idx = np.where(reject_sorted)[0][-1]
            threshold = critical_values[max_idx]
        else:
            threshold = 0
        
        reject = pvalues <= threshold
        
        # Compute adjusted p-values
        corrected_pvals = np.zeros(n)
        for i in range(n):
            corrected_pvals[sorted_idx[i]] = min(
                sorted_pvals[i] * n / (i + 1),
                1.0
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corrected_pvals, reject


def detect_outliers(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    method : str
        Method to use ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    np.ndarray
        Boolean array indicating outliers
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = (data < lower) | (data > upper)
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers


def compute_correlation_significance(
    corr_matrix: np.ndarray,
    n_samples: int,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Compute significance of correlations in a correlation matrix.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix
    n_samples : int
        Number of samples used to compute correlations
    alpha : float
        Significance level
        
    Returns
    -------
    np.ndarray
        Boolean matrix indicating significant correlations
    """
    # Compute t-statistic for each correlation
    t_stat = corr_matrix * np.sqrt((n_samples - 2) / (1 - corr_matrix**2 + 1e-10))
    
    # Compute p-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - 2))
    
    # Apply significance threshold
    significant = p_values < alpha
    
    return significant
