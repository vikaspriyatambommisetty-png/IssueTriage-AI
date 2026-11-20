import io
import json
import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from triage import ROOT, SAMPLE, analyze, dataset


class TriageTests(unittest.TestCase):
    def test_sample(self):
        draft = analyze(SAMPLE)
        self.assertEqual(draft.label, "Bug")
        self.assertIn(draft.evidence, SAMPLE)
        self.assertIn("## Follow-up", draft.markdown())

    def test_invalid_input(self):
        for invalid in (" ", "x" * 4001):
            with self.assertRaises(ValueError):
                analyze(invalid)

    def test_evaluation_separate(self):
        train = {r["report"] for r in dataset("training.csv")}
        test = {r["report"] for r in dataset("evaluation.csv")}
        self.assertEqual(len(test), 30)
        self.assertFalse(train & test)

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"})
    @patch("triage.urlopen")
    def test_cloud_quote_validation(self, request):
        for quote in ("two bookings", "payment was refunded"):
            fields = dict(title="Duplicate booking", label="Bug", evidence=quote, question="Did the stock count change?")
            body = {"choices": [{"message": {"content": json.dumps(fields)}}]}
            request.return_value = io.BytesIO(json.dumps(body).encode())
            if quote in SAMPLE:
                self.assertEqual(analyze(SAMPLE, cloud=True).evidence, quote)
            else:
                with self.assertRaisesRegex(ValueError, "exact-quote"):
                    analyze(SAMPLE, cloud=True)

    def test_ui(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.button[0].click().run()
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[1].value, "Bug")
        app.text_input[0].set_value("Two bookings for one camera").run()
        self.assertIn("Two bookings for one camera", app.code[0].value)
        app.text_area[0].set_value("How do I export my report?").run()
        self.assertEqual(len(app.code), 0)
