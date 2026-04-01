"""Aggregate reproducibility analysis across conditions."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mri_harmonization.analysis.cv import compute_cv
from mri_harmonization.analysis.icc import compute_icc
from mri_harmonization.analysis.kruskal import compute_kruskal_wallis

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityResult:
    """Per-feature reproducibility metrics."""

    feature_name: str
    icc: float
    cv: float
    kw_statistic: float
    kw_pvalue: float
    is_reproducible: bool  # ICC > threshold


def analyze_reproducibility(
    df: pd.DataFrame,
    icc_threshold: float = 0.75,
    alpha: float = 0.05,
) -> list[ReproducibilityResult]:
    """Analyze feature reproducibility across sites.

    Args:
        df: Feature DataFrame with subject_id, site, and feature columns.
        icc_threshold: ICC threshold for "reproducible" classification.
        alpha: Significance level for Kruskal-Wallis test.

    Returns:
        List of ReproducibilityResult, one per feature.
    """
    meta_cols = {"subject_id", "site"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    sites = df["site"]

    results: list[ReproducibilityResult] = []
    for feat in feature_cols:
        values = df[feat]

        icc = compute_icc(values, sites)
        cv = compute_cv(values)
        kw_stat, kw_pval = compute_kruskal_wallis(values, sites)

        results.append(
            ReproducibilityResult(
                feature_name=feat,
                icc=icc,
                cv=cv,
                kw_statistic=kw_stat,
                kw_pvalue=kw_pval,
                is_reproducible=icc >= icc_threshold,
            )
        )

    return results


def summarize_conditions(
    results_by_condition: dict[str, list[ReproducibilityResult]],
) -> pd.DataFrame:
    """Summarize reproducibility across harmonization conditions.

    Args:
        results_by_condition: Dict mapping condition name to its results.

    Returns:
        Summary DataFrame with one row per condition.
    """
    rows = []
    for condition, results in results_by_condition.items():
        if not results:
            continue

        iccs = [r.icc for r in results]
        n_reproducible = sum(1 for r in results if r.is_reproducible)
        n_significant = sum(1 for r in results if r.kw_pvalue < 0.05)

        rows.append({
            "condition": condition,
            "n_features": len(results),
            "pct_reproducible": 100.0 * n_reproducible / len(results),
            "median_icc": float(np.median(iccs)),
            "mean_icc": float(np.mean(iccs)),
            "n_significant_site_effect": n_significant,
        })

    return pd.DataFrame(rows)
