"""
Tests for executor classes (JudgeExecutor, ReasonerExecutor).

These tests cover:
- Non-JSON salvage logic in executors
- execute_best_effort_final behavior
- ReasonerExecutor.execute() parallel execution
- JudgeExecutor.execute() candidate selection
- ReasonerResult dataclass
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from react_usc import (
    JudgeExecutor,
    ReasonerDecision,
    ReasonerExecutor,
    ToolRegistry,
)
from react_usc.executors import ReasonerResult
from react_usc._internal.validation import (
    validate_judge_decision_dict,
    validate_reasoner_decision_dict,
)

from tests.conftest import make_mock_model, make_test_config, make_test_tool


# =============================================================================
# Test: JudgeExecutor salvage logic
# =============================================================================


class TestJudgeSalvageNonJsonFinal(unittest.TestCase):
    """Test the non-JSON salvage logic in JudgeExecutor."""

    def setUp(self):
        self.config = make_test_config(accept_non_json_final=True)
        self.tools = ToolRegistry([make_test_tool()])
        self.model = make_mock_model()
        self.executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )

    def test_salvage_single_line_final_answer(self):
        """Test salvaging a single-line final answer."""
        from react_usc._internal.salvage import salvage_judge_final

        text = '''decision_type: FINAL
final_answer: "The answer is 42"
justification: This is the computed result.'''

        result = salvage_judge_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertEqual(result["final_answer"], "The answer is 42")
        self.assertIsNone(result["tool_name"])
        self.assertIsNone(result["tool_args"])

    def test_salvage_multiline_final_answer(self):
        """Test salvaging a multi-line final answer."""
        from react_usc._internal.salvage import salvage_judge_final

        text = '''decision_type: FINAL
final_answer:
The answer spans multiple lines.
Here is more content.
And even more.
justification: Computed result.'''

        result = salvage_judge_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertIn("multiple lines", result["final_answer"])
        self.assertIn("more content", result["final_answer"])

    def test_no_salvage_for_tool_call(self):
        """Test that TOOL_CALL is not salvaged."""
        from react_usc._internal.salvage import salvage_judge_final

        text = '''decision_type: TOOL_CALL
tool_name: test_tool
tool_args: {"query": "hello"}'''

        result = salvage_judge_final(text)

        self.assertIsNone(result)

    def test_no_salvage_for_empty_text(self):
        """Test that empty text is not salvaged."""
        from react_usc._internal.salvage import salvage_judge_final

        result = salvage_judge_final("")
        self.assertIsNone(result)

        result = salvage_judge_final("   \n  ")
        self.assertIsNone(result)

    def test_no_salvage_without_final_answer_key(self):
        """Test that text without final_answer key is not salvaged."""
        from react_usc._internal.salvage import salvage_judge_final

        text = '''decision_type: FINAL
answer: The answer is 42'''

        result = salvage_judge_final(text)

        self.assertIsNone(result)

    def test_salvage_with_quoted_answer(self):
        """Test salvaging with various quote styles."""
        from react_usc._internal.salvage import salvage_judge_final

        # Double quotes
        text1 = '''final_answer: "Quoted answer"'''
        result1 = salvage_judge_final(text1)
        self.assertIsNotNone(result1)
        self.assertEqual(result1["final_answer"], "Quoted answer")

        # Single quotes
        text2 = '''final_answer: 'Single quoted'
justification: test'''
        result2 = salvage_judge_final(text2)
        self.assertIsNotNone(result2)
        self.assertEqual(result2["final_answer"], "Single quoted")

    def test_salvaged_result_validates(self):
        """Test that salvaged results pass validation."""
        from react_usc._internal.salvage import salvage_judge_final

        text = '''decision_type: FINAL
final_answer: "The validated answer"
justification: This should validate.'''

        result = salvage_judge_final(text)
        self.assertIsNotNone(result)

        decision, errors = validate_judge_decision_dict(result)
        self.assertIsNotNone(decision)
        self.assertEqual(len(errors), 0)
        self.assertEqual(decision.final_answer, "The validated answer")


# =============================================================================
# Test: ReasonerExecutor salvage logic
# =============================================================================


class TestReasonerSalvageNonJsonFinal(unittest.TestCase):
    """Test the non-JSON salvage logic in ReasonerExecutor."""

    def setUp(self):
        self.config = make_test_config(accept_non_json_final=True)
        self.tools = ToolRegistry([make_test_tool()])
        self.model = make_mock_model()
        self.executor = ReasonerExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )

    def test_salvage_single_line_final_answer(self):
        """Test salvaging a single-line final answer."""
        from react_usc._internal.salvage import salvage_reasoner_final

        text = '''decision_type: FINAL
final_answer: "The answer is 42"
brief_rationale: This is the computed result.'''

        result = salvage_reasoner_final(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertEqual(result["final_answer"], "The answer is 42")
        self.assertIsNone(result["tool_name"])
        self.assertIsNone(result["tool_args"])

    def test_salvaged_result_validates(self):
        """Test that salvaged results pass validation."""
        from react_usc._internal.salvage import salvage_reasoner_final

        text = '''decision_type: FINAL
final_answer: "The validated answer"
brief_rationale: This should validate.'''

        result = salvage_reasoner_final(text)
        self.assertIsNotNone(result)

        decision, errors = validate_reasoner_decision_dict(result)
        self.assertIsNotNone(decision)
        self.assertEqual(len(errors), 0)
        self.assertEqual(decision.final_answer, "The validated answer")


# =============================================================================
# Test: Integration - execute_best_effort_final with salvage
# =============================================================================


class TestExecuteBestEffortFinal(unittest.TestCase):
    """Test execute_best_effort_final with non-JSON salvage."""

    def setUp(self):
        self.config = make_test_config(accept_non_json_final=True)
        self.tools = ToolRegistry([make_test_tool()])
        self.model = make_mock_model()

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    @patch('react_usc._internal.llm_io.invoke_chat_structured_obj')
    def test_best_effort_returns_decision_on_valid_json(self, mock_structured, mock_text):
        """Test that valid JSON returns a proper decision."""
        # Mock structured output to fail (to test text fallback)
        mock_structured.side_effect = TypeError("No structured output")
        # Mock invoke_chat_text to return valid JSON string
        mock_text.return_value = '{"decision_type": "FINAL", "final_answer": "Test answer", "justification": "Best effort"}'

        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )

        decision = executor.execute_best_effort_final(
            user_query="What is 2+2?",
            observations=["calculator => 4"],
            max_steps=5,
        )

        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Test answer")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    @patch('react_usc._internal.llm_io.invoke_chat_structured_obj')
    def test_best_effort_fallback_on_parse_failure(self, mock_structured, mock_text):
        """Test that parse failure returns fallback decision."""
        # Mock structured output to fail
        mock_structured.side_effect = TypeError("No structured output")
        # Mock invoke_chat_text to return invalid content
        mock_text.return_value = "This is not JSON at all"

        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )

        decision = executor.execute_best_effort_final(
            user_query="What is 2+2?",
            observations=[],
            max_steps=5,
        )

        # Should return fallback decision
        self.assertEqual(decision.decision_type, "FINAL")
        self.assertIn("Step limit exceeded", decision.final_answer)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    @patch('react_usc._internal.llm_io.invoke_chat_structured_obj')
    def test_best_effort_salvages_non_json_final(self, mock_structured, mock_text):
        """Test that non-JSON final answer is salvaged when enabled."""
        # Mock structured output to fail
        mock_structured.side_effect = TypeError("No structured output")
        # Mock invoke_chat_text to return non-JSON with final_answer
        mock_text.return_value = '''decision_type: FINAL
final_answer: "Salvaged answer from non-JSON"
justification: Because observations indicate the result.'''

        executor = JudgeExecutor(
            model=self.model,
            config=self.config,  # accept_non_json_final=True
            tools=self.tools,
        )

        decision = executor.execute_best_effort_final(
            user_query="What is 2+2?",
            observations=["calculator => 4"],
            max_steps=5,
        )

        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Salvaged answer from non-JSON")


# =============================================================================
# Test: ReasonerResult
# =============================================================================


class TestReasonerResult(unittest.TestCase):
    """Test ReasonerResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid ReasonerResult."""
        decision = ReasonerDecision(
            decision_type="FINAL",
            tool_name=None,
            tool_args=None,
            final_answer="Test answer",
            brief_rationale="Test rationale",
        )
        result = ReasonerResult(
            path_id=0,
            decision=decision,
            raw_output={"decision_type": "FINAL"},
        )
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.path_id, 0)
        self.assertIsNotNone(result.decision)
        self.assertIsNone(result.error)

    def test_invalid_result_no_decision(self):
        """Test ReasonerResult without decision."""
        result = ReasonerResult(
            path_id=1,
            decision=None,
            raw_output={},
            error="Parse error",
        )
        
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.decision)
        self.assertEqual(result.error, "Parse error")

    def test_result_with_error(self):
        """Test ReasonerResult with error message."""
        result = ReasonerResult(
            path_id=2,
            decision=None,
            raw_output={},
            error="Timeout after 20s",
        )
        
        self.assertFalse(result.is_valid)
        self.assertIn("Timeout", result.error)


# =============================================================================
# Test: ReasonerExecutor.execute()
# =============================================================================


class TestReasonerExecutorExecute(unittest.TestCase):
    """Test ReasonerExecutor.execute() method."""

    def setUp(self):
        self.config = make_test_config(use_structured_output=False)
        self.tools = ToolRegistry([make_test_tool()])
        self.model = make_mock_model()

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_returns_valid_candidates(self, mock_invoke):
        """Test that execute returns valid candidates."""
        # Mock to return valid FINAL decision JSON
        mock_invoke.return_value = json.dumps({
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "The answer is 42",
            "brief_rationale": "Computed from query",
        })
        
        executor = ReasonerExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates, invalid = executor.execute(
            user_query="What is 2+2?",
            state_summary="No observations yet.",
        )
        
        # Should have k_paths valid candidates (all return same valid JSON)
        self.assertEqual(len(candidates), self.config.k_paths)
        self.assertEqual(len(invalid), 0)
        
        # Each candidate should be a FINAL decision
        for candidate in candidates:
            self.assertEqual(candidate.decision_type, "FINAL")
            self.assertEqual(candidate.final_answer, "The answer is 42")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_returns_tool_call_candidates(self, mock_invoke):
        """Test that execute returns TOOL_CALL candidates."""
        mock_invoke.return_value = json.dumps({
            "decision_type": "TOOL_CALL",
            "tool_name": "test_tool",
            "tool_args": {"query": "hello"},
            "final_answer": None,
            "brief_rationale": "Need to search",
        })
        
        executor = ReasonerExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates, invalid = executor.execute(
            user_query="Search for something",
            state_summary="No observations yet.",
        )
        
        self.assertEqual(len(candidates), self.config.k_paths)
        for candidate in candidates:
            self.assertEqual(candidate.decision_type, "TOOL_CALL")
            self.assertEqual(candidate.tool_name, "test_tool")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_filters_invalid_candidates(self, mock_invoke):
        """Test that execute filters invalid candidates."""
        # Return invalid JSON
        mock_invoke.return_value = "This is not valid JSON"
        
        executor = ReasonerExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates, invalid = executor.execute(
            user_query="Test query",
            state_summary="No observations.",
        )
        
        # All candidates should be invalid
        self.assertEqual(len(candidates), 0)
        self.assertEqual(len(invalid), self.config.k_paths)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_filters_unknown_tool(self, mock_invoke):
        """Test that execute filters candidates with unknown tools."""
        mock_invoke.return_value = json.dumps({
            "decision_type": "TOOL_CALL",
            "tool_name": "nonexistent_tool",
            "tool_args": {"query": "test"},
            "final_answer": None,
            "brief_rationale": "Call unknown tool",
        })
        
        executor = ReasonerExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates, invalid = executor.execute(
            user_query="Test query",
            state_summary="No observations.",
        )
        
        # All should be invalid due to unknown tool
        self.assertEqual(len(candidates), 0)
        self.assertGreater(len(invalid), 0)


# =============================================================================
# Test: JudgeExecutor.execute()
# =============================================================================


class TestJudgeExecutorExecute(unittest.TestCase):
    """Test JudgeExecutor.execute() method."""

    def setUp(self):
        self.config = make_test_config(use_structured_output=False)
        self.tools = ToolRegistry([make_test_tool()])
        self.model = make_mock_model()

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_selects_final_candidate(self, mock_invoke):
        """Test that execute selects a FINAL candidate."""
        mock_invoke.return_value = json.dumps({
            "decision_type": "FINAL",
            "selected_index": 0,
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Selected answer",
            "justification": "Best answer from candidates",
        })
        
        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates = [
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="Candidate answer",
                brief_rationale="Rationale",
            ),
        ]
        
        decision = executor.execute(
            user_query="What is the answer?",
            state_summary="Observations: none",
            candidates=candidates,
        )
        
        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Selected answer")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_selects_tool_call(self, mock_invoke):
        """Test that execute selects a TOOL_CALL."""
        mock_invoke.return_value = json.dumps({
            "decision_type": "TOOL_CALL",
            "selected_index": 0,
            "tool_name": "test_tool",
            "tool_args": {"query": "search term"},
            "final_answer": None,
            "justification": "Need to search first",
        })
        
        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates = [
            ReasonerDecision(
                decision_type="TOOL_CALL",
                tool_name="test_tool",
                tool_args={"query": "search term"},
                final_answer=None,
                brief_rationale="Search for info",
            ),
        ]
        
        decision = executor.execute(
            user_query="Find information",
            state_summary="No observations",
            candidates=candidates,
        )
        
        self.assertEqual(decision.decision_type, "TOOL_CALL")
        self.assertEqual(decision.tool_name, "test_tool")
        self.assertEqual(decision.tool_args, {"query": "search term"})

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_returns_final_on_invalid_output(self, mock_invoke):
        """Test that execute falls back to candidate on invalid judge output."""
        mock_invoke.side_effect = [
            "This is not valid JSON",
            "Still not valid JSON",
        ]
        
        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates = [
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="Candidate",
                brief_rationale="Rationale",
            ),
        ]
        
        decision = executor.execute(
            user_query="Test",
            state_summary="Summary",
            candidates=candidates,
        )
        
        # Should return a FINAL decision from fallback selection
        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Candidate")
        self.assertIn("fallback", decision.justification.lower())
        self.assertEqual(mock_invoke.call_count, 2)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_validates_tool_call(self, mock_invoke):
        """Test that execute validates tool calls."""
        # First call returns invalid tool, second call (reprompt) returns valid tool
        mock_invoke.side_effect = [
            json.dumps({
                "decision_type": "TOOL_CALL",
                "selected_index": 0,
                "tool_name": "unknown_tool",
                "tool_args": {},
                "final_answer": None,
                "justification": "Use unknown tool",
            }),
            json.dumps({
                "decision_type": "TOOL_CALL",
                "selected_index": 0,
                "tool_name": "test_tool",
                "tool_args": {"query": "test"},
                "final_answer": None,
                "justification": "Use valid tool",
            }),
        ]
        
        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates = [
            ReasonerDecision(
                decision_type="TOOL_CALL",
                tool_name="test_tool",
                tool_args={"query": "test"},
                final_answer=None,
                brief_rationale="Rationale",
            ),
        ]
        
        decision = executor.execute(
            user_query="Test",
            state_summary="Summary",
            candidates=candidates,
        )
        
        # With unified fallback, an invalid tool call goes to fallback_from_candidates
        # since the unified function already validated and the validator doesn't check tool existence
        # The tool validation happens after unified fallback returns
        self.assertEqual(decision.decision_type, "TOOL_CALL")
        # Should fallback to candidate since judge selected unknown tool
        self.assertEqual(decision.tool_name, "test_tool")

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_reprompts_and_recovers(self, mock_invoke):
        """Test that execute reprompts and recovers from invalid output."""
        mock_invoke.side_effect = [
            "This is not valid JSON",
            json.dumps({
                "decision_type": "FINAL",
                "selected_index": 0,
                "tool_name": None,
                "tool_args": None,
                "final_answer": "Recovered answer",
                "justification": "Recovered after reprompt",
            }),
        ]

        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )

        candidates = [
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="Candidate",
                brief_rationale="Rationale",
            ),
        ]

        decision = executor.execute(
            user_query="Test",
            state_summary="Summary",
            candidates=candidates,
        )

        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.final_answer, "Recovered answer")
        self.assertEqual(mock_invoke.call_count, 2)

    @patch('react_usc._internal.llm_io.invoke_chat_text')
    def test_execute_with_multiple_candidates(self, mock_invoke):
        """Test execute with multiple candidates."""
        mock_invoke.return_value = json.dumps({
            "decision_type": "FINAL",
            "selected_index": 1,
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Best answer",
            "justification": "Second candidate was better",
        })
        
        executor = JudgeExecutor(
            model=self.model,
            config=self.config,
            tools=self.tools,
        )
        
        candidates = [
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="First answer",
                brief_rationale="First rationale",
            ),
            ReasonerDecision(
                decision_type="FINAL",
                tool_name=None,
                tool_args=None,
                final_answer="Second answer",
                brief_rationale="Second rationale",
            ),
            ReasonerDecision(
                decision_type="TOOL_CALL",
                tool_name="test_tool",
                tool_args={"query": "test"},
                final_answer=None,
                brief_rationale="Third rationale",
            ),
        ]
        
        decision = executor.execute(
            user_query="Choose best",
            state_summary="Summary",
            candidates=candidates,
        )
        
        self.assertEqual(decision.decision_type, "FINAL")
        self.assertEqual(decision.selected_index, 1)


if __name__ == "__main__":
    unittest.main()
