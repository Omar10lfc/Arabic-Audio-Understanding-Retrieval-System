import os
import re
import pytest
import nbformat
import pandas as pd
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2, normalize_arabic

def extract_training_normalizer():
    nb_path = "legacy/whisper-large-v3-qlora.ipynb"
    if not os.path.exists(nb_path):
        nb_path = "Notebooks/legacy/whisper-large-v3-qlora.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    target_code = None
    for cell in nb.cells:
        if cell.cell_type == "code" and "def normalize_arabic" in cell.source:
            target_code = cell.source
            break
    assert target_code is not None, "Could not find normalize_arabic cell in training notebook"
    
    # Isolate function definition specifically
    fn_code_lines = []
    in_fn = False
    for line in target_code.splitlines():
        if line.strip().startswith("def normalize_arabic"):
            in_fn = True
        elif in_fn and line.startswith("def ") and not line.strip().startswith("def normalize_arabic"):
            break
        if in_fn:
            fn_code_lines.append(line)
    
    fn_code = "\n".join(fn_code_lines)
    from typing import Optional, List, Dict
    local_env = {"re": re, "pd": pd, "os": os, "Optional": Optional, "List": List, "Dict": Dict}
    exec(fn_code, local_env)
    return local_env["normalize_arabic"]


def test_normalizer_parity_extracted_vs_evalkit():
    training_fn = extract_training_normalizer()
    
    # 1. 20 Edge cases
    edge_cases = [
        "مرحبا، كيف الحال",                   # Arabic comma
        "هل أنت هنا؟",                         # Arabic question mark
        "عام ٢٠٢٤ و ٣ كتب",                   # Arabic-Indic digits
        "تقنية AI و Deep Learning",           # Latin words
        "شبه-جزيرة العرب",                    # Hyphen
        "مسؤول ورؤية وتفاؤل",                 # Hamza on waw (ؤ)
        "شاطئ ومبتدئين وهيئة",                # Hamza on ya (ئ)
        "قال: «السلام عليكم!»،",              # Complex punctuation
        "السَّـلَامُ عَلَيْكُمْ",              # Diacritics and tatweel
        "سماء وهواء وفضاء",                   # Isolated hamza (ء)
        "أحمد وإبراهيم وآدم وٱبن",            # Alef forms
        "موسى وعيسى ومستشفى",                  # Dotless ya
        "مدرسة وجامعة وحياة",                 # Ta-marbuta
        "كلمة   كثيرة     المسافات",          # Excessive whitespace
        "Model_v3-large tested",              # Latin + underscore + hyphen
        "كود Python 3.10 فعال",               # Mixed Arabic-Latin with version
        "اجتهد الطالب؛ فنجح بتفوق",           # Arabic semicolon
        "اسم_المستخدم_الرسمي",                # Underscores
        "نسبة 50% إلى 75%",                   # Percent and numbers
        "   \t\n   ",                         # Pure whitespace
    ]
    
    for case in edge_cases:
        expected = training_fn(case)
        actual = normalize_arabic_legacy(case)
        assert actual == expected, f"Parity mismatch on edge case '{case}': expected '{expected}', got '{actual}'"

    # 2. 500 Sentences from Common Voice test.tsv
    tsv_path = "scratch/test.tsv"
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        text_col = "sentence" if "sentence" in df.columns else "text"
        sentences = df[text_col].dropna().iloc[:500].tolist()
        for idx, s in enumerate(sentences):
            expected = training_fn(s)
            actual = normalize_arabic_legacy(s)
            assert actual == expected, f"Mismatch at sentence {idx}: '{s}' -> expected '{expected}', got '{actual}'"


def test_normalizer_v2_differences_documented():
    # Verify documented differences
    # 1. ؤ and ئ converted to ء in v2, preserved in legacy
    assert normalize_arabic_legacy("مسؤول") == "مسؤول"
    assert normalize_arabic_v2("مسؤول") == "مسءول"
    assert normalize_arabic_legacy("شاطئ") == "شاطئ"
    assert normalize_arabic_v2("شاطئ") == "شاطء"
    
    # 2. Arabic punctuation stripped in v2, preserved in legacy
    assert normalize_arabic_legacy("نعم، صحيح؟") == "نعم، صحيح؟"
    assert normalize_arabic_v2("نعم، صحيح؟") == "نعم صحيح"
    
    # 3. Latin characters stripped in v2, preserved in legacy
    assert normalize_arabic_legacy("AI نموذج") == "AI نموذج"
    assert normalize_arabic_v2("AI نموذج") == "نموذج"
