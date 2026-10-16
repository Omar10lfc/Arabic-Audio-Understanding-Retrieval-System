import jiwer

def pure_python_wer_breakdown(reference, hypothesis):
    if isinstance(reference, str):
        ref_words = reference.split()
    else:
        ref_words = [w for r in reference for w in r.split()]

    if isinstance(hypothesis, str):
        hyp_words = hypothesis.split()
    else:
        hyp_words = [w for h in hypothesis for w in h.split()]

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0 and m == 0:
        return 0, 0, 0, 0, 0, 0.0
    if n == 0:
        return 0, 0, m, 0, 0, float(m)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, delete)

    i, j = n, m
    s, d, ins_cnt, h = 0, 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            h += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins_cnt += 1
            j -= 1
        else:
            raise RuntimeError("Backtrack error")

    wer = (s + d + ins_cnt) / n if n > 0 else float(ins_cnt)
    return s, d, ins_cnt, h, n, wer

# Test pairs
test_cases = [
    ("ذهب الطالب الى المدرسة صباحا", "ذهب طالب الى مدرسة صباحا باكرا"),
    ("هذا نص تجريبي", "هذا نص تجريبي"),
    ("واحد اثنان ثلاثة", "اثنان ثلاثة أربعة"),
    ("أ ب ج د", "أ ج د هـ"),
    ("كل عام وانتم بخير", ""),
    ("", "كل عام وانتم بخير"),
    ("قال الوزير ان الاقتصاد يتعافى بشكل ملحوظ", "الوزير قال ان اقتصاد يتعافى بوضوح تام"),
    ("السلام عليكم ورحمة الله وبركاته", "وعليكم السلام ورحمة الله"),
]

for idx, (ref, hyp) in enumerate(test_cases):
    s_py, d_py, i_py, h_py, n_py, wer_py = pure_python_wer_breakdown(ref, hyp)
    res_jw = jiwer.process_words(ref, hyp)
    jw_wer = jiwer.wer(ref, hyp)
    
    assert abs(wer_py - jw_wer) < 1e-12, f"WER mismatch: {wer_py} vs {jw_wer} for ({ref}, {hyp})"
    assert s_py == res_jw.substitutions, f"Sub mismatch: {s_py} vs {res_jw.substitutions}"
    assert d_py == res_jw.deletions, f"Del mismatch: {d_py} vs {res_jw.deletions}"
    assert i_py == res_jw.insertions, f"Ins mismatch: {i_py} vs {res_jw.insertions}"
    assert h_py == res_jw.hits, f"Hits mismatch: {h_py} vs {res_jw.hits}"
    print(f"Test {idx+1} PASSED: WER={wer_py:.4f} (S={s_py}, D={d_py}, I={i_py}, H={h_py})")

print("\nALL TEST CASES MATCHED JIWWER EXACTLY!")
