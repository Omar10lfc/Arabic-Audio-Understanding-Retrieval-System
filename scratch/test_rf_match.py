import rapidfuzz

def levenshtein_opcodes_pure(r, h):
    # DP tracking (cost, -hits)
    # We want to minimize cost; if cost is equal, maximize hits (minimize -hits)
    n, m = len(r), len(h)
    dp = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                # Equal: cost + 0, hits + 1
                cand_equal = (dp[i - 1][j - 1][0], dp[i - 1][j - 1][1] - 1)
            else:
                cand_equal = (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1])

            cand_del = (dp[i - 1][j][0] + 1, dp[i - 1][j][1])
            cand_ins = (dp[i][j - 1][0] + 1, dp[i][j - 1][1])

            dp[i][j] = min(cand_equal, cand_del, cand_ins)

    # Backtrack
    i, j = n, m
    s, d, ins, hits = 0, 0, 0, 0
    while i > 0 or j > 0:
        cost, neg_h = dp[i][j]
        if i > 0 and j > 0 and r[i - 1] == h[j - 1] and dp[i - 1][j - 1] == (cost, neg_h + 1):
            hits += 1
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
            # Fallback
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

    return s, d, ins, hits

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

for ref, hyp in test_cases:
    r_words = ref.split()
    h_words = hyp.split()
    s_py, d_py, i_py, h_py = levenshtein_opcodes_pure(r_words, h_words)
    ops = rapidfuzz.distance.Levenshtein.opcodes(r_words, h_words)
    s_rf = sum(i2 - i1 for tag, i1, i2, j1, j2 in ops if tag == 'replace')
    d_rf = sum(i2 - i1 for tag, i1, i2, j1, j2 in ops if tag == 'delete')
    i_rf = sum(j2 - j1 for tag, i1, i2, j1, j2 in ops if tag == 'insert')
    h_rf = sum(i2 - i1 for tag, i1, i2, j1, j2 in ops if tag == 'equal')

    assert s_py == s_rf and d_py == d_rf and i_py == i_rf and h_py == h_rf, (
        f"Mismatch on ({ref}, {hyp}): py=(S={s_py}, D={d_py}, I={i_py}, H={h_py}) vs rf=(S={s_rf}, D={d_rf}, I={i_rf}, H={h_rf})"
    )

print("ALL 8 TEST CASES MATCHED RAPIDFUZZ EXACTLY!")
