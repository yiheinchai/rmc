"""SealQA evidence enrichment tests."""

from __future__ import annotations

import unittest

from rose.sealqa_evidence import classify_snippets, infer_axis, skill_for_axis


class TestSealQAEvidence(unittest.TestCase):
    def test_classify_garbled(self) -> None:
        self.assertEqual(classify_snippets(["the world.-,Highest-grossing films,-[12]"]), "garbled")

    def test_classify_usable(self) -> None:
        self.assertEqual(
            classify_snippets(["Serban Ghenea is the most frequent winner with five awards."]),
            "usable",
        )

    def test_infer_axis_priority(self) -> None:
        self.assertEqual(
            infer_axis("q", ["false-premise", "entity/event disambiguation"]),
            "no-guess",
        )
        self.assertEqual(
            infer_axis("How many items?", ["advanced reasoning"]),
            "stated-count",
        )

    def test_skill_for_axis(self) -> None:
        self.assertIn("Answer", skill_for_axis("answer-format"))


if __name__ == "__main__":
    unittest.main()
