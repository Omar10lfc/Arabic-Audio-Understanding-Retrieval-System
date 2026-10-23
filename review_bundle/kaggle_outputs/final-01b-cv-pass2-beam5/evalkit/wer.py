from typing import Tuple, Dict, Any, Union, List

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    jiwer = None
    _HAS_JIWER = False

def _pure_python_wer_breakdown(
    reference: Union[str, List[str]], 
    hypothesis: Union[str, List[str]]
) -> Tuple[int, int, int, int, int, float]:
    """
    Exact, pure-Python Wagner-Fischer DP word alignment algorithm matching jiwer / rapidfuzz.
    Requires zero external dependencies.
    """
    if isinstance(reference, str):
        ref_words = reference.split()
    else:
        ref_words = [w for r in reference for w in r.split()]

    if isinstance(hypothesis, str):
        hyp_words = hypothesis.split()
    else:
        hyp_words = [w for h in hypothesis for w in h.split()]

    n, m = len(ref_words), len(hyp_words)
    if n == 0 and m == 0:
        return 0, 0, 0, 0, 0, 0.0
    if n == 0:
        return 0, 0, m, 0, 0, 1.0

    # Wagner-Fischer DP tracking (cost, -hits) to match rapidfuzz/jiwer opcodes exactly
    dp = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cand_equal = (dp[i - 1][j - 1][0], dp[i - 1][j - 1][1] - 1)
            else:
                cand_equal = (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1])
            cand_del = (dp[i - 1][j][0] + 1, dp[i - 1][j][1])
            cand_ins = (dp[i][j - 1][0] + 1, dp[i][j - 1][1])
            dp[i][j] = min(cand_equal, cand_del, cand_ins)

    i, j = n, m
    s, d, ins, h = 0, 0, 0, 0
    while i > 0 or j > 0:
        cost, neg_h = dp[i][j]
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i - 1][j - 1] == (cost, neg_h + 1):
            h += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i - 1][j - 1] == (cost - 1, neg_h):
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i - 1][j] == (cost - 1, neg_h):
            d += 1
            i -= 1
        elif j > 0 and dp[i][j - 1] == (cost - 1, neg_h):
            ins += 1
            j -= 1
        else:
            if i > 0 and j > 0 and cost == dp[i - 1][j - 1][0] + 1:
                s += 1
                i -= 1
                j -= 1
            elif i > 0 and cost == dp[i - 1][j][0] + 1:
                d += 1
                i -= 1
            else:
                ins += 1
                j -= 1

    wer = (s + d + ins) / n if n > 0 else 0.0
    return s, d, ins, h, n, wer

def compute_wer(
    reference: Union[str, List[str]], 
    hypothesis: Union[str, List[str]]
) -> float:
    """
    Computes Word Error Rate.
    Returns WER = (S + D + I) / (H + S + D).
    """
    s, d, i, h, n, wer = wer_breakdown(reference, hypothesis)
    return wer

def wer_breakdown(
    reference: Union[str, List[str]], 
    hypothesis: Union[str, List[str]]
) -> Tuple[int, int, int, int, int, float]:
    """
    Computes detailed alignment statistics.
    Returns: (S, D, I, H, N, WER)
      S: Substitutions
      D: Deletions
      I: Insertions
      H: Hits
      N: Total reference words = H + S + D
      WER: (S + D + I) / N
    Uses jiwer when available, with pure-Python fallback.
    """
    if _HAS_JIWER:
        if isinstance(reference, str):
            ref_list = [reference]
        else:
            ref_list = reference
        if isinstance(hypothesis, str):
            hyp_list = [hypothesis]
        else:
            hyp_list = hypothesis

        res = jiwer.process_words(ref_list, hyp_list)
        s = res.substitutions
        d = res.deletions
        i = res.insertions
        h = res.hits
        n = h + s + d

        if n == 0:
            wer = 0.0 if i == 0 else 1.0
        else:
            wer = (s + d + i) / n

        return s, d, i, h, n, wer
    else:
        return _pure_python_wer_breakdown(reference, hypothesis)

def compute_dual_wer_record(raw_ref: str, raw_hyp: str, **kwargs) -> Dict[str, Any]:
    """
    Computes and formats alignment metrics under both legacy and v2 normalization protocols.
    Saves raw ref, raw hyp, legacy-normalized and v2-normalized text, plus S/D/I/H/N and WER under both.
    """
    from .normalize import normalize_arabic_legacy, normalize_arabic_v2
    ref_leg = normalize_arabic_legacy(raw_ref)
    hyp_leg = normalize_arabic_legacy(raw_hyp)
    s_leg, d_leg, i_leg, h_leg, n_leg, wer_leg = wer_breakdown(ref_leg, hyp_leg)

    ref_v2 = normalize_arabic_v2(raw_ref)
    hyp_v2 = normalize_arabic_v2(raw_hyp)
    s_v2, d_v2, i_v2, h_v2, n_v2, wer_v2 = wer_breakdown(ref_v2, hyp_v2)

    return {
        'raw_ref': raw_ref,
        'raw_hyp': raw_hyp,
        'ref_legacy': ref_leg,
        'hyp_legacy': hyp_leg,
        'normalized_hyp_legacy': hyp_leg,
        'legacy_s': s_leg,
        'legacy_d': d_leg,
        'legacy_i': i_leg,
        'legacy_h': h_leg,
        'legacy_n': n_leg,
        'legacy_wer': round(wer_leg, 6),
        'legacy_breakdown': {
            'substitutions': s_leg,
            'deletions': d_leg,
            'insertions': i_leg,
            'hits': h_leg,
            'words': n_leg,
            'wer': round(wer_leg, 6)
        },
        'ref_v2': ref_v2,
        'hyp_v2': hyp_v2,
        'normalized_hyp_v2': hyp_v2,
        'v2_s': s_v2,
        'v2_d': d_v2,
        'v2_i': i_v2,
        'v2_h': h_v2,
        'v2_n': n_v2,
        'v2_wer': round(wer_v2, 6),
        'v2_breakdown': {
            'substitutions': s_v2,
            'deletions': d_v2,
            'insertions': i_v2,
            'hits': h_v2,
            'words': n_v2,
            'wer': round(wer_v2, 6)
        }
    }
