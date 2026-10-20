import os
import math
import subprocess
import time
import json
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional
from scipy import stats

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for binomial proportions.
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    spread = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (round(lower, 4), round(upper, 4))

def mcnemar_exact(y_a: List[int], y_b: List[int]) -> Dict[str, Any]:
    """
    Exact two-sided McNemar test for paired binary classification / retrieval hit vectors.
    Evaluates discordant pairs using exact binomial test with p=0.5.
    """
    assert len(y_a) == len(y_b), "Vector lengths must match."
    b = sum(1 for a, b_val in zip(y_a, y_b) if a == 1 and b_val == 0)
    c = sum(1 for a, b_val in zip(y_a, y_b) if a == 0 and b_val == 1)
    n_disc = b + c

    if n_disc == 0:
        p_val = 1.0
    else:
        res = stats.binomtest(b, n_disc, p=0.5, alternative='two-sided')
        p_val = float(res.pvalue)

    return {
        'b': b,
        'c': c,
        'n_discordant': n_disc,
        'p_value': round(p_val, 4),
        'significant': p_val < 0.05
    }

def bootstrap_ci(
    data: List[float], 
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000, 
    confidence: float = 0.95, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval computation. Reports standard error (se) with ddof=1.
    """
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {'mean': 0.0, 'se': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    se_val = float(np.std(boot_stats, ddof=1)) if n_bootstrap > 1 else 0.0

    return {
        'mean': round(float(np.mean(boot_stats)), 4),
        'se': round(se_val, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
    }

def paired_bootstrap_ci(
    a: List[float],
    b: List[float],
    stat: str = "mean_diff",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Paired bootstrap confidence interval resampling paired observations jointly.
    Supported stats:
      - 'mean_diff': E[a] - E[b]
      - 'ratio_of_means': E[a] / E[b]
    """
    assert len(a) == len(b), f"Mismatched pair lengths: len(a)={len(a)}, len(b)={len(b)}"
    assert stat in {"mean_diff", "ratio_of_means"}, f"Unsupported stat '{stat}'. Must be in {{'mean_diff', 'ratio_of_means'}}"
    
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    n = len(arr_a)
    if n == 0:
        return {'stat': stat, 'mean': 0.0, 'se': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        samp_a = arr_a[idx]
        samp_b = arr_b[idx]
        mean_a = float(np.mean(samp_a))
        mean_b = float(np.mean(samp_b))
        if stat == "mean_diff":
            boot_stats[i] = mean_a - mean_b
        elif stat == "ratio_of_means":
            boot_stats[i] = (mean_a / mean_b) if mean_b != 0.0 else np.nan

    alpha = 1.0 - confidence
    valid_boots = boot_stats[~np.isnan(boot_stats)]
    if len(valid_boots) == 0:
        return {'stat': stat, 'mean': 0.0, 'se': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    ci_lower = float(np.percentile(valid_boots, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(valid_boots, 100.0 * (1.0 - alpha / 2.0)))
    se_val = float(np.std(valid_boots, ddof=1)) if len(valid_boots) > 1 else 0.0

    return {
        'stat': stat,
        'mean': round(float(np.mean(valid_boots)), 4),
        'se': round(se_val, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
    }

def cluster_bootstrap_ci(
    df,
    group_col: str,
    metric_fn: Callable[[Any], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Cluster bootstrap resampling groups (e.g. client_id / speaker) with replacement.
    """
    groups = df[group_col].unique()
    n_groups = len(groups)
    rng = np.random.default_rng(seed)

    grouped = {g: group_df for g, group_df in df.groupby(group_col)}

    import pandas as pd
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        sampled_dfs = [grouped[g] for g in sampled_groups]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)
        boot_stats[i] = metric_fn(boot_df)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    se_val = float(np.std(boot_stats, ddof=1)) if n_bootstrap > 1 else 0.0

    return {
        'mean': round(float(np.mean(boot_stats)), 4),
        'se': round(se_val, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
    }

def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Step-down Holm-Bonferroni correction across multiple hypothesis tests.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(running_max, 4)

    return adjusted

def get_git_commit() -> str:
    """Gets current git commit hash."""
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return out.decode('utf-8').strip()
    except Exception:
        return "unknown"

def get_evalkit_sha256() -> Dict[str, str]:
    """Computes SHA-256 for each Python source file in evalkit."""
    evalkit_dir = os.path.dirname(__file__)
    shas = {}
    for fname in sorted(os.listdir(evalkit_dir)):
        if fname.endswith(".py"):
            fpath = os.path.join(evalkit_dir, fname)
            with open(fpath, "rb") as f:
                shas[fname] = hashlib.sha256(f.read()).hexdigest()
    return shas

def get_config_sha256(config_path: str = "configs/final_protocol.json") -> Optional[str]:
    """Computes SHA-256 of the active protocol configuration file."""
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return None

def get_file_sha256(filepath: Optional[str]) -> Optional[str]:
    """Computes SHA-256 for a specific file path."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return None

def save_result_file(
    filepath: str,
    results_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    notebook_path: Optional[str] = None,
    notebook_sha256: Optional[str] = None
):
    """
    Stores result data alongside complete protocol config copy, git commit,
    sha256 of every evalkit source file, sha256 of protocol config, notebook sha256,
    device, and timestamp.
    """
    resolved_nb_sha = notebook_sha256 or get_file_sha256(notebook_path)
    payload = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'git_commit': get_git_commit(),
            'device': device or "unknown",
            'evalkit_sha256': get_evalkit_sha256(),
            'config_sha256': get_config_sha256(),
            'notebook_sha256': resolved_nb_sha,
            'config_copy': config or {},
        },
        'results': results_data
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

