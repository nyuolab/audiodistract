"""Shared statistics used across the three analyses.

All accuracy differences are reported in percentage points (pp), never as relative changes.
Paired quantities are always computed within item / encounter.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_accuracy_diff(correct_ref: np.ndarray, correct_cmp: np.ndarray) -> tuple[float, float]:
    """Paired accuracy difference (cmp - ref) in pp and its standard error.

    SE uses the binomial-covariance formula from the MedDistractQA paper:
    sqrt([p1(1-p1) + p2(1-p2) - 2(p12 - p1 p2)] / n) * 100, where p12 is the joint-correct proportion.
    """
    a = np.asarray(correct_ref, dtype=float)
    b = np.asarray(correct_cmp, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    n = a.size
    p1, p2, p12 = a.mean(), b.mean(), (a * b).mean()
    var = (p1 * (1 - p1) + p2 * (1 - p2) - 2 * (p12 - p1 * p2)) / n
    return float((p2 - p1) * 100), float(np.sqrt(max(var, 0.0)) * 100)


def mcnemar_exact(correct_ref: np.ndarray, correct_cmp: np.ndarray) -> float:
    """Two-sided exact McNemar (binomial) test on discordant pairs."""
    a = np.asarray(correct_ref, dtype=bool)
    b = np.asarray(correct_cmp, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    return float(stats.binomtest(min(b01, b10), n_disc, 0.5, alternative="two-sided").pvalue)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 statistic=np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a statistic of a 1-D sample. Returns (point, low, high)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = statistic(x[idx], axis=1)
    return float(statistic(x)), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_bootstrap_ci(ref: np.ndarray, cmp: np.ndarray, n_boot: int = 2000, seed: int = 0,
                        scale: float = 100.0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean(cmp - ref) over paired units, scaled to pp by default."""
    d = (np.asarray(cmp, dtype=float) - np.asarray(ref, dtype=float)) * scale
    return bootstrap_ci(d, n_boot=n_boot, seed=seed)


def welch_t(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    """Welch t-test treating models as independent observations. alternative in {two-sided, less, greater}."""
    res = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares y ~ x with Pearson r, r^2 and p-value of the slope."""
    res = stats.linregress(np.asarray(x, float), np.asarray(y, float))
    return {"slope": float(res.slope), "intercept": float(res.intercept), "r": float(res.rvalue),
            "r2": float(res.rvalue ** 2), "p": float(res.pvalue), "stderr": float(res.stderr)}


def wilcoxon_signed(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)
