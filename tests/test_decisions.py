"""
Tests for decision classes (JudgeDecision, ReasonerDecision, ReflectionResult).

These tests cover:
- Empty candidates handling
- Decision creation and properties
- ReflectionResult property methods
"""
from __future__ import annotations

import unittest

from react_usc import (
    JudgeDecision,
    ReasonerDecision,
    ReasonerResult,
    ReflectionResult,
)


# =============================================================================
# Test: Empty Candidates Handling (Integration)
# =============================================================================


class TestEmptyCandidatesHandling(unittest.TestCase):
    """Test the _handle_all_reasoners_failed method in the agent."""

    def test_judge_decision_create_final(self):
        """Test that JudgeDecision.create_final works correctly."""
        decision = JudgeDecision.create_final(
            answer="Test answer",
            justification="Test justification",
        )

        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Test answer")
        self.assertEqual(decision.justification, "Test justification")
        self.assertIsNone(decision.tool_name)
        self.assertIsNone(decision.tool_args)
        self.assertTrue(decision.is_final)
        self.assertFalse(decision.is_tool_call)

    def test_reasoner_result_invalid_when_no_decision(self):
        """Test that ReasonerResult reports invalid when decision is None."""
        result = ReasonerResult(
            path_id=0,
            decision=None,
            raw_output={},
            error="Test error",
        )

        self.assertFalse(result.is_valid)

    def test_reasoner_result_valid_when_has_decision(self):
        """Test that ReasonerResult reports valid when decision exists."""
        decision = ReasonerDecision(
            decision_type="FINAL",
            tool_name=None,
            tool_args=None,
            final_answer="Test",
            brief_rationale="Test rationale",
        )
        result = ReasonerResult(
            path_id=0,
            decision=decision,
            raw_output={},
        )

        self.assertTrue(result.is_valid)


# =============================================================================
# Test: ReflectionResult Properties
# =============================================================================


class TestReflectionResultProperties(unittest.TestCase):
    """Test ReflectionResult property methods."""

    def test_should_retry(self):
        """Test should_retry property."""
        result = ReflectionResult(
            verdict="RETRY",
            retry_args={"key": "value"},
        )
        self.assertTrue(result.should_retry)

        # RETRY without args should not retry
        result_no_args = ReflectionResult(verdict="RETRY")
        self.assertFalse(result_no_args.should_retry)

    def test_should_wait(self):
        """Test should_wait property."""
        result = ReflectionResult(verdict="WAIT")
        self.assertTrue(result.should_wait)

        result_other = ReflectionResult(verdict="ABORT")
        self.assertFalse(result_other.should_wait)

    def test_should_abort(self):
        """Test should_abort property."""
        result = ReflectionResult(verdict="ABORT")
        self.assertTrue(result.should_abort)

        result_other = ReflectionResult(verdict="RETRY", retry_args={})
        self.assertFalse(result_other.should_abort)


if __name__ == "__main__":
    unittest.main()
