"""
Unit tests for the pure helpers in pipeline.py — no models loaded, no GPU
required, no internet. These should run in well under a second.
"""

import pytest

from pipeline import (
    normalize_arabic, clean_for_summary, chunk_text, jaccard,
    CHEAT_QUERIES, CHUNK_WORDS,
)


# =============================================================================
# normalize_arabic — full normalization, must match embedding-eval.ipynb
# =============================================================================
class TestNormalizeArabic:

    def test_empty(self):
        assert normalize_arabic("") == ""
        assert normalize_arabic(None) == ""  # type: ignore[arg-type]

    def test_alef_unification(self):
        assert normalize_arabic("أحمد") == "احمد"
        assert normalize_arabic("إيمان") == "ايمان"
        assert normalize_arabic("آية") == "ايه"     # also ta-marbuta → ha

    def test_dotless_ya(self):
        assert normalize_arabic("على") == "علي"

    def test_ta_marbuta(self):
        assert normalize_arabic("مدرسة") == "مدرسه"

    def test_diacritics_stripped(self):
        # fatha + damma + kasra etc.
        assert normalize_arabic("مَكْتَبَة") == "مكتبه"

    def test_tatweel_stripped(self):
        assert normalize_arabic("سـلام") == "سلام"

    def test_whitespace_collapsed(self):
        assert normalize_arabic("  مرحبا   بك  ") == "مرحبا بك"

    def test_combined(self):
        assert normalize_arabic("أهْـلًا  بِكَ ") == "اهلا بك"


# =============================================================================
# clean_for_summary — lighter cleanup, keeps alef forms / ta-marbuta intact
# =============================================================================
class TestCleanForSummary:

    def test_keeps_alef_forms(self):
        # Unlike normalize_arabic, clean_for_summary preserves AraBART's
        # training-time tokenization assumptions.
        assert clean_for_summary("أحمد") == "أحمد"
        assert clean_for_summary("مدرسة") == "مدرسة"
        assert clean_for_summary("على") == "على"

    def test_strips_diacritics(self):
        assert clean_for_summary("مَكْتَبَة") == "مكتبة"

    def test_strips_tatweel(self):
        assert clean_for_summary("سـلام") == "سلام"

    def test_empty(self):
        assert clean_for_summary("") == ""


# =============================================================================
# chunk_text — 50-word windows, drop tiny tails
# =============================================================================
class TestChunkText:

    def test_short_text_one_chunk(self):
        # chunk_text passes input through normalize_arabic, so "كلمة" → "كلمه"
        # in the output (ta-marbuta unification). That's intentional — chunks
        # must live in the same normalized space as the embedder's queries.
        text = " ".join(["كلمة"] * 30)
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].count("كلمه") == 30

    def test_default_chunk_size_is_50(self):
        text = " ".join([f"كلمة{i}" for i in range(120)])
        chunks = chunk_text(text)
        # 120 / 50 = 2 full chunks + a 20-word tail (kept — > 5 words)
        assert len(chunks) == 3
        assert len(chunks[0].split()) == 50
        assert len(chunks[1].split()) == 50
        assert len(chunks[2].split()) == 20

    def test_drops_tiny_tail(self):
        text = " ".join([f"كلمة{i}" for i in range(53)])
        chunks = chunk_text(text)
        # 50 + 3-word tail → tail dropped (< 5 words)
        assert len(chunks) == 1
        assert len(chunks[0].split()) == 50

    def test_custom_chunk_size(self):
        text = " ".join([f"w{i}" for i in range(30)])
        chunks = chunk_text(text, chunk_words=10)
        assert len(chunks) == 3
        assert all(len(c.split()) == 10 for c in chunks)

    def test_chunks_are_normalized(self):
        # Chunks should already be passed through normalize_arabic (otherwise
        # the embedding-space alignment is broken).
        chunks = chunk_text("أحمد " * 10)
        assert "أ" not in chunks[0]
        assert "احمد" in chunks[0]

    def test_empty(self):
        assert chunk_text("") == []


# =============================================================================
# jaccard — token-set overlap for takeaway dedup
# =============================================================================
class TestJaccard:

    def test_identical(self):
        assert jaccard("a b c", "a b c") == 1.0

    def test_disjoint(self):
        assert jaccard("a b c", "d e f") == 0.0

    def test_partial(self):
        # {a, b, c} ∩ {b, c, d} = {b, c}; ∪ = {a, b, c, d}
        assert jaccard("a b c", "b c d") == pytest.approx(2 / 4)

    def test_empty_strings(self):
        assert jaccard("", "") == 0.0


# =============================================================================
# Cheat-sheet config sanity
# =============================================================================
class TestCheatQueriesConfig:

    def test_five_sections(self):
        assert len(CHEAT_QUERIES) == 5

    def test_titles_unique(self):
        titles = [t for t, _ in CHEAT_QUERIES]
        assert len(set(titles)) == len(titles)

    def test_queries_in_arabic(self):
        # Every query should contain at least one Arabic letter.
        import re
        ar = re.compile(r"[؀-ۿ]")
        for _, q in CHEAT_QUERIES:
            assert ar.search(q), f"Query not in Arabic: {q!r}"


def test_chunk_words_default_is_50():
    """50-word chunking is the configuration that won the eval (P@1 0.86 with
       reranking) — guard against accidental changes."""
    assert CHUNK_WORDS == 50
