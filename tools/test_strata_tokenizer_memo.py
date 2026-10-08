"""The tokenizer's segment memo (incremental chat encoding): same ids as a memo-less encode, whatever the history.

    python -m unittest tools.test_strata_tokenizer_memo -v

A synthetic byte-level vocabulary (no pack needed); the real GLM vocabulary is checked on the model's own chat
turns and against llama-tokenize (tools/test_glm_tokenizer_parity.py).
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def tokenizer():
    byte_tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    tokens = byte_tokens + ["th", "the", "\u0120the", "<|user|>", "<|assistant|>", "<think>"]
    types = [1] * (len(tokens) - 3) + [3, 3, 4]
    return ST.Tokenizer(tokens, ["t h", "th e", "\u0120 the"], types, pre="glm4")


def reference():
    tok = tokenizer()
    tok._encode_segment = tok._encode_plain           # every segment through BPE again
    return tok


WORDS = ["the", "then", "other", "x = 1;", "\n", "  ", "caf\u00e9", "\u4f60\u597d", "\U0001F600", "<|user|>",
         "<|assistant|>", "<think>", "<|use", "think>", "\t"]


def text(rng, n):
    return "".join(rng.choice(WORDS) + rng.choice(["", " "]) for _ in range(n))


class SegmentMemo(unittest.TestCase):
    def test_chat_turns_match_a_fresh_encode(self):
        rng = random.Random(7)
        tok, ref = tokenizer(), reference()
        conv = text(rng, 400)
        for turn in range(60):
            if turn % 4 == 3:                                   # an edit inside an old segment
                cut = rng.randrange(len(conv))
                conv = conv[:cut] + "edited" + conv[cut:]
            else:                                               # an appended turn
                conv += "<|assistant|>" + text(rng, rng.randrange(5, 300)) + "<|user|>" + text(rng, 200)
            for special in (True, False):
                with self.subTest(turn=turn, parse_special=special):
                    self.assertEqual(tok.encode(conv, parse_special=special), ref.encode(conv, parse_special=special))
        self.assertGreater(len(tok._memo), 0)

    def test_eviction_keeps_the_budget_and_the_ids(self):
        rng = random.Random(11)
        with mock.patch.object(ST, "MEMO_MAX_CHARS", 3000):
            tok, ref = tokenizer(), reference()
            for _ in range(80):
                s = "<|user|>".join(text(rng, rng.randrange(60, 200)) for _ in range(3))
                self.assertEqual(tok.encode(s, parse_special=True), ref.encode(s, parse_special=True))
                self.assertLessEqual(tok._memo_chars, 3000)
                self.assertEqual(tok._memo_chars, sum(len(k) for k in tok._memo))

    def test_callers_cannot_corrupt_the_memo(self):
        tok = tokenizer()
        s = "the other " * 60
        first = tok.encode(s)
        first.append(-1)
        self.assertEqual(tok.encode(s), reference().encode(s))
        both = tok.encode(s + "<|user|>" + s, parse_special=True)
        both.clear()
        self.assertEqual(tok.encode(s, parse_special=True), reference().encode(s, parse_special=True))


if __name__ == "__main__":
    unittest.main()
