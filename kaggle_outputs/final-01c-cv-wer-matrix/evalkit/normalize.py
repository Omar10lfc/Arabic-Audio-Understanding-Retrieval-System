import re
from typing import Optional

def normalize_arabic_legacy(text: Optional[str]) -> str:
    r"""
    Deterministic Arabic normalization identical to evaluation baseline and training notebook
    (legacy/whisper-large-v3-qlora.ipynb):
    unifies alef variants, dotless ya, ta-marbuta, removes diacritics & tatweel.
    Preserves Arabic punctuation, Arabic-Indic digits, Latin characters via \w and \u0600-\u06FF.
    Does not collapse ؤ or ئ to isolated ء.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[إأآٱ]', 'ا', text)              # alef forms -> ا
    text = re.sub(r'ى', 'ي', text)                    # dotless ya -> ي
    text = re.sub(r'ة', 'ه', text)                    # ta-marbuta -> ه
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text) # strip diacritics
    text = re.sub(r'ـ', '', text)                      # strip tatweel
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)  # keep Arabic letters & digits
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_arabic_v2(text: Optional[str]) -> str:
    r"""
    Stricter Arabic normalization (v2) applied in downstream Speech-RAG / retrieval evaluation:
    
    Documented differences vs normalize_arabic_legacy:
    1. Arabic punctuation (،, ؛, ؟): Legacy retains them via \u0600-\u06FF.
       V2 strips them completely to spaces, keeping only \u0621-\u064A and 0-9.
    2. Arabic-Indic digits (٠-٩): Legacy retains Arabic-Indic digits.
       V2 strips non-ASCII digits (unless mapped), keeping only 0-9.
    3. Latin words: Legacy retains ASCII Latin words/numbers via \w.
       V2 strips all Latin words and characters.
    4. Hyphens and underscores: Legacy retains underscores via \w.
       V2 strips hyphens and underscores to spaces.
    5. Hamza on waw and ya (ؤ, ئ): Legacy leaves ؤ and ئ unchanged.
       V2 replaces ؤ -> ء and ئ -> ء.
    6. Diacritic stripping order: V2 strips diacritics before letter substitutions;
       legacy normalizes alef/ya/ta-marbuta first.
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # diacritics
    text = re.sub(r'ـ', '', text)                      # tatweel
    text = re.sub(r'[إأآٱ]', 'ا', text)                # alef variants
    text = re.sub(r'ى', 'ي', text)                     # dotless ya
    text = re.sub(r'ة', 'ه', text)                     # ta-marbuta
    text = re.sub(r'ؤ', 'ء', text)                     # hamza on waw
    text = re.sub(r'ئ', 'ء', text)                     # hamza on ya
    text = re.sub(r'[^\u0621-\u064A0-9\s]', ' ', text) # strip non-arabic letters/digits
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_arabic(text: Optional[str], mode: str = "legacy") -> str:
    """
    Dispatch wrapper for Arabic normalization. Defaults to 'legacy' as primary protocol.
    """
    if mode == "v2":
        return normalize_arabic_v2(text)
    return normalize_arabic_legacy(text)


def clean_for_summary(text: Optional[str]) -> str:
    """
    Lighter cleanup preserving orthographic distinctions required by AraBART's BPE tokenizer.
    - Strips diacritics and tatweel
    - Collapses whitespace
    - Preserves alef variants, ta-marbuta, and sentence punctuation
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
