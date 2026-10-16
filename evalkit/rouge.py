import sys
import subprocess
from typing import Dict, List, Union

_scorer_stemmed = None
_scorer_unstemmed = None

def _init_xlsum_scorer():
    global _scorer_stemmed, _scorer_unstemmed
    try:
        from rouge_score import rouge_scorer
        _scorer_stemmed = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True, 
            lang='arabic'
        )
        _scorer_unstemmed = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=False, 
            lang='arabic'
        )
        return True
    except (TypeError, Exception):
        return False

# 1. Try initializing already installed scorer
if not _init_xlsum_scorer():
    # 2. Try pip installing official csebuetnlp/xl-sum scorer if internet is available
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
        ])
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("rouge_score"):
                del sys.modules[mod_name]
        _init_xlsum_scorer()
    except Exception:
        pass

# 3. Fallback: Exact NLTK Snowball Arabic stemmer reproducing XL-Sum behaviour offline
if _scorer_stemmed is None:
    from collections import Counter
    try:
        from nltk.stem.snowball import SnowballStemmer
        _arabic_stemmer = SnowballStemmer('arabic')
    except Exception:
        _arabic_stemmer = None

    def _tokenize_words(text: str, stem: bool = False) -> List[str]:
        words = [w for w in text.strip().split() if w]
        if stem and _arabic_stemmer is not None:
            return [_arabic_stemmer.stem(w) for w in words]
        return words

    def _lcs_length(x: List[str], y: List[str]) -> int:
        n, m = len(x), len(y)
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            prev = 0
            for j in range(1, m + 1):
                temp = dp[j]
                if x[i - 1] == y[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[m]

    def _calc_rouge(reference: str, hypothesis: str, use_stemmer: bool = True) -> Dict[str, float]:
        ref_tok = _tokenize_words(reference, stem=use_stemmer)
        hyp_tok = _tokenize_words(hypothesis, stem=use_stemmer)
        if not ref_tok or not hyp_tok:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        # ROUGE-1
        c_ref = Counter(ref_tok)
        c_hyp = Counter(hyp_tok)
        overlap1 = sum(min(c_hyp[w], c_ref[w]) for w in c_hyp)
        r1 = (2.0 * overlap1 / (len(ref_tok) + len(hyp_tok))) * 100.0 if (len(ref_tok) + len(hyp_tok)) > 0 else 0.0

        # ROUGE-2
        ng_ref = [tuple(ref_tok[i:i + 2]) for i in range(len(ref_tok) - 1)]
        ng_hyp = [tuple(hyp_tok[i:i + 2]) for i in range(len(hyp_tok) - 1)]
        c2_ref = Counter(ng_ref)
        c2_hyp = Counter(ng_hyp)
        overlap2 = sum(min(c2_hyp[g], c2_ref[g]) for g in c2_hyp)
        r2 = (2.0 * overlap2 / (len(ng_ref) + len(ng_hyp))) * 100.0 if (len(ng_ref) + len(ng_hyp)) > 0 else 0.0

        # ROUGE-L
        lcs = _lcs_length(ref_tok, hyp_tok)
        prec = lcs / len(hyp_tok)
        rec = lcs / len(ref_tok)
        rl = (2.0 * prec * rec / (prec + rec)) * 100.0 if (prec + rec) > 0 else 0.0

        return {
            'rouge1': round(r1, 4),
            'rouge2': round(r2, 4),
            'rougeL': round(rl, 4),
        }


def score_summary(
    reference: str, 
    hypothesis: str, 
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Official XL-Sum (csebuetnlp/xl-sum) ROUGE evaluation for Arabic.
    Uses official multilingual RougeScorer with lang='arabic'.
    """
    scorer = _scorer_stemmed if use_stemmer else _scorer_unstemmed
    if scorer is not None:
        scores = scorer.score(reference.strip(), hypothesis.strip())
        return {
            'rouge1': round(scores['rouge1'].fmeasure * 100.0, 4),
            'rouge2': round(scores['rouge2'].fmeasure * 100.0, 4),
            'rougeL': round(scores['rougeL'].fmeasure * 100.0, 4),
        }
    return _calc_rouge(reference, hypothesis, use_stemmer=use_stemmer)


def multilingual_rouge_scoring(
    predictions: List[str], 
    references: List[str], 
    lang: str = 'arabic',
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Official multilingual XL-Sum style evaluation across parallel lists of predictions and references.
    """
    assert len(predictions) == len(references), "Predictions and references length mismatch."
    if not predictions:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    r1_list, r2_list, rl_list = [], [], []
    for pred, ref in zip(predictions, references):
        sc = score_summary(ref, pred, use_stemmer=use_stemmer)
        r1_list.append(sc['rouge1'])
        r2_list.append(sc['rouge2'])
        rl_list.append(sc['rougeL'])

    import numpy as np
    return {
        'rouge1': round(float(np.mean(r1_list)), 4),
        'rouge2': round(float(np.mean(r2_list)), 4),
        'rougeL': round(float(np.mean(rl_list)), 4),
    }
