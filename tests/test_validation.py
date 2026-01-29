"""
Tests for validation functions.

These tests cover edge cases in decision validation.
"""
from __future__ import annotations

import unittest

from react_usc._internal.validation import (
    validate_judge_decision_dict,
    validate_reasoner_decision_dict,
)


# =============================================================================
# Test: Validation Functions
# =============================================================================


class TestValidationFunctions(unittest.TestCase):
    """Test validation functions handle edge cases correctly."""

    def test_validate_reasoner_final_decision(self):
        """Test validating a correct FINAL reasoner decision."""
        d = {
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "The answer is 42",
            "brief_rationale": "Computed from observations",
        }

        decision, errors = validate_reasoner_decision_dict(d)

        self.assertIsNotNone(decision)
        self.assertEqual(len(errors), 0)
        self.assertTrue(decision.is_final)

    def test_validate_reasoner_tool_call_decision(self):
        """Test validating a correct TOOL_CALL reasoner decision."""
        d = {
            "decision_type": "TOOL_CALL",
            "tool_name": "test_tool",
            "tool_args": {"query": "hello"},
            "final_answer": None,
            "brief_rationale": "Need to search for information",
        }

        decision, errors = validate_reasoner_decision_dict(d)

        self.assertIsNotNone(decision)
        self.assertEqual(len(errors), 0)
        self.assertTrue(decision.is_tool_call)

    def test_validate_reasoner_missing_rationale(self):
        """Test that missing brief_rationale is caught."""
        d = {
            "decision_type": "FINAL",
            "final_answer": "Answer",
            # Missing brief_rationale
        }

        decision, errors = validate_reasoner_decision_dict(d)

        self.assertIsNone(decision)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("brief_rationale" in e for e in errors))

    def test_validate_judge_final_decision(self):
        """Test validating a correct FINAL judge decision."""
        d = {
            "decision_type": "FINAL",
            "selected_index": None,
            "tool_name": None,
            "tool_args": None,
            "final_answer": "The answer is 42",
            "justification": "Based on candidate analysis",
        }

        decision, errors = validate_judge_decision_dict(d)

        self.assertIsNotNone(decision)
        self.assertEqual(len(errors), 0)
        self.assertTrue(decision.is_final)

    def test_validate_judge_tool_call_missing_tool_name(self):
        """Test that TOOL_CALL without tool_name is caught."""
        d = {
            "decision_type": "TOOL_CALL",
            "tool_name": None,  # Should be a string
            "tool_args": {"query": "test"},
            "final_answer": None,
            "justification": "Need more info",
        }

        decision, errors = validate_judge_decision_dict(d)

        self.assertIsNone(decision)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("tool_name" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
