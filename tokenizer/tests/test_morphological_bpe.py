"""
Tests for the morphological BPE tokenizer.

Run with:
    python -m pytest tokenizer/tests/test_morphological_bpe.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morphological_bpe import MorphologicalBPETrainer


# Shared test corpus — contains words with common morphemes
TEST_TEXT = """
The unhappiness of the disconnected international community was unreasonable.
They were rethinking their preexisting assumptions about interconnected systems.
The disorganization was overwhelming and the misinformation spread quickly.
Scientists were rediscovering fundamental truths about the natural world.
The government was reorganizing its departments for greater effectiveness.
Teachers found that rewriting the curriculum improved student engagement.
The remarkable transformation of the industry was unprecedented in history.
Understanding the relationship between education and innovation requires careful analysis.
The international community expressed disagreement about environmental regulations.
Researchers were investigating the interconnection of biological systems.
"""

# Repeat to give BPE enough data to find patterns
TEST_CORPUS = TEST_TEXT * 50


def test_forced_morphemes_basic():
    """Train with forced morphemes, verify encode/decode round-trip."""
    trainer = MorphologicalBPETrainer()
    morphemes = ["un", "re", "dis", "ing", "tion", "ness", "able", "ment"]
    trainer.train(TEST_CORPUS, vocab_size=512, forced_morphemes=morphemes)

    # Verify forced merges were created
    assert trainer.forced_merge_count > 0
    print(f"Forced merge count: {trainer.forced_merge_count}")

    # Round-trip test
    test = "The unhappiness and disconnection was unreasonable"
    ids = trainer.encode(test)
    decoded = trainer.decode(ids)
    assert decoded == test, f"Round-trip failed: {decoded!r} != {test!r}"
    print(f"Round-trip OK: {len(test)} chars -> {len(ids)} tokens")


def test_forced_morphemes_prefix_sharing():
    """
    Train with morphemes that share prefixes.
    "in" should be merged first (2 bytes -> 1 merge).
    "ing" needs: i+n (reuse) then in+g (1 new merge) = 1 new merge.
    "inter" needs: i+n (reuse) then in+t (new) then int+e (new) then inte+r (new) = 3 new merges.
    Total: 1 + 1 + 3 = 5 merges (not 2 + 2 + 4 = 8 without sharing).
    """
    trainer = MorphologicalBPETrainer()
    morphemes = ["in", "ing", "inter"]
    trainer.train(TEST_CORPUS, vocab_size=512, forced_morphemes=morphemes)

    # "in" = 1 merge (i,n)->256
    # "ing" = 1 new merge: (in,g)->257  [reuses (i,n)->256]
    # "inter" = 3 new merges: (256,t)->258, (258,e)->259, (259,r)->260
    #   [reuses (i,n)->256 from "in"]
    # Total new merges = 5
    assert trainer.forced_merge_count == 5, \
        f"Expected 5 forced merges, got {trainer.forced_merge_count}"
    print(f"Prefix sharing OK: 3 morphemes -> {trainer.forced_merge_count} merges (not 8)")


def test_forced_morphemes_tiktoken_export():
    """Verify tiktoken encoding matches trainer encoding with forced merges."""
    trainer = MorphologicalBPETrainer()
    morphemes = ["un", "re", "dis", "ing", "tion"]
    trainer.train(TEST_CORPUS, vocab_size=512, forced_morphemes=morphemes)

    enc = trainer.to_tiktoken()

    # Compare encoding on several test strings
    test_strings = [
        "unhappiness",
        "disconnection",
        "rethinking",
        "international",
        "The quick brown fox jumps over the lazy dog",
    ]
    for text in test_strings:
        trainer_ids = trainer.encode(text)
        tiktoken_ids = enc.encode(text)
        assert trainer_ids == tiktoken_ids, \
            f"Mismatch on {text!r}: trainer={trainer_ids} tiktoken={tiktoken_ids}"

    # Also verify decode round-trip through tiktoken
    for text in test_strings:
        ids = enc.encode(text)
        decoded = enc.decode(ids)
        assert decoded == text, f"tiktoken round-trip failed: {decoded!r} != {text!r}"

    print("tiktoken export matches trainer encoding")


def test_forced_morphemes_backward_compat():
    """Verify forced_morphemes=None produces identical results to no parameter."""
    trainer_a = MorphologicalBPETrainer()
    trainer_b = MorphologicalBPETrainer()

    trainer_a.train(TEST_CORPUS, vocab_size=300, forced_morphemes=None)
    trainer_b.train(TEST_CORPUS, vocab_size=300)

    # Same merges
    assert trainer_a.merges == trainer_b.merges, "Merges differ with None vs no morphemes"
    assert trainer_a.forced_merge_count == 0
    assert trainer_b.forced_merge_count == 0

    # Same encoding
    test = "The unhappiness of the disconnected community"
    assert trainer_a.encode(test) == trainer_b.encode(test)
    print("Backward compatibility OK: None == no morphemes")


def test_forced_morphemes_single_byte_skipped():
    """Single-byte morphemes (like 'a', 'I') should be skipped."""
    trainer = MorphologicalBPETrainer()
    morphemes = ["a", "I", "un", "re"]  # 'a' and 'I' are single bytes
    trainer.train(TEST_CORPUS, vocab_size=512, forced_morphemes=morphemes)

    # Only "un" and "re" should create forced merges (1 merge each)
    assert trainer.forced_merge_count == 2, \
        f"Expected 2 forced merges (skipping single-byte), got {trainer.forced_merge_count}"
    print("Single-byte morphemes correctly skipped")


def test_encode_decode_roundtrip_no_morphemes():
    """Basic sanity: vanilla BPE encode/decode round-trip."""
    trainer = MorphologicalBPETrainer()
    trainer.train(TEST_CORPUS, vocab_size=512)

    test_strings = [
        "Hello world!",
        "The quick brown fox",
        "Numbers: 123, 456",
        "Special chars: @#$%",
    ]
    for text in test_strings:
        ids = trainer.encode(text)
        decoded = trainer.decode(ids)
        assert decoded == text, f"Round-trip failed: {decoded!r} != {text!r}"
    print("Vanilla BPE round-trip OK")


def test_train_from_iterator():
    """Test the streaming iterator training path."""
    def text_iter():
        for _ in range(50):
            yield TEST_TEXT

    trainer = MorphologicalBPETrainer()
    morphemes = ["un", "re", "ing"]
    trainer.train_from_iterator(text_iter(), vocab_size=512, forced_morphemes=morphemes)

    assert trainer.forced_merge_count > 0
    assert len(trainer.merges) > trainer.forced_merge_count

    test = "understanding and rethinking"
    ids = trainer.encode(test)
    decoded = trainer.decode(ids)
    assert decoded == test
    print(f"Iterator training OK: {len(trainer.merges)} merges")


def test_morpheme_tokens_are_stable():
    """
    Forced morphemes should always get the same token IDs regardless
    of the training corpus, since they're assigned before statistical BPE.
    """
    morphemes = ["un", "re", "dis"]

    trainer_a = MorphologicalBPETrainer()
    trainer_a.train(TEST_CORPUS, vocab_size=512, forced_morphemes=morphemes)

    # Different corpus
    other_corpus = "A completely different text about cats and dogs. " * 500
    trainer_b = MorphologicalBPETrainer()
    trainer_b.train(other_corpus, vocab_size=512, forced_morphemes=morphemes)

    # The forced merge IDs should be identical
    # "un": u=117, n=110 -> (117,110) -> 256
    # "re": r=114, e=101 -> (114,101) -> 257
    # "dis": d=100, i=105 -> (100,105) -> 258, then (258, s=115) -> 259
    assert trainer_a.merges[(117, 110)] == trainer_b.merges[(117, 110)] == 256
    assert trainer_a.merges[(114, 101)] == trainer_b.merges[(114, 101)] == 257
    print("Morpheme token IDs are corpus-independent")


if __name__ == "__main__":
    test_forced_morphemes_basic()
    test_forced_morphemes_prefix_sharing()
    test_forced_morphemes_tiktoken_export()
    test_forced_morphemes_backward_compat()
    test_forced_morphemes_single_byte_skipped()
    test_encode_decode_roundtrip_no_morphemes()
    test_train_from_iterator()
    test_morpheme_tokens_are_stable()
    print("\nAll tests passed!")
