"""
Tests for the unified LLM invocation function.

This module tests the invoke_with_unified_fallback function which provides
a consistent fallback chain for all LLM calls:
1. Try structured output (if enabled)
2. Fall back to text parsing
3. Try salvage (if enabled)
4. Reprompt with error context
5. Raise JSONParseError if all fail
"""
from __future__ import annotations

import json
import unittest
from typing import Dict, Optional
from unittest.mock import MagicMock

from react_usc._internal.llm_io import (
    _build_repair_prompt,
    invoke_with_unified_fallback,
)
from react_usc._internal.normalizers import normalize_reasoner_decision_obj
from react_usc._internal.validation import validate_reasoner_decision_dict
from react_usc._internal.salvage import salvage_reasoner_final
from react_usc._internal.schema import get_reasoner_decision_schema
from react_usc.exceptions import JSONParseError


class TestBuildRepairPrompt(unittest.TestCase):
    """Tests for the _build_repair_prompt helper function."""
    
    def test_builds_prompt_with_error_and_schema(self):
        """Test that repair prompt includes error reason and schema."""
        original_user = "Original prompt"
        error_reason = "Missing required field 'foo'"
        schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        
        result = _build_repair_prompt(original_user, error_reason, schema)
        
        self.assertIn(original_user, result)
        self.assertIn(error_reason, result)
        self.assertIn("foo", result)
        self.assertIn("INVALID", result)
        self.assertIn("REQUIRED SCHEMA", result)


class TestInvokeWithUnifiedFallback(unittest.TestCase):
    """Tests for the unified fallback function."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.schema = get_reasoner_decision_schema([])
        self.valid_final_response = {
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Test answer",
            "brief_rationale": "Test rationale",
        }
    
    def _make_mock_model(self, text_response: str = "", structured_response: Optional[Dict] = None):
        """Create a mock model with configurable responses."""
        model = MagicMock()
        model.invoke = MagicMock(return_value=MagicMock(content=text_response))
        
        if structured_response is not None:
            model.with_structured_output = MagicMock(return_value=MagicMock(
                invoke=MagicMock(return_value=structured_response)
            ))
        else:
            model.with_structured_output = MagicMock(side_effect=TypeError("No structured output"))
        
        return model
    
    def test_structured_output_success(self):
        """Test successful path through structured output."""
        model = self._make_mock_model(structured_response=self.valid_final_response)
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=True,
            max_reprompts=0,
        )
        
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertEqual(result["final_answer"], "Test answer")
    
    def test_structured_output_disabled_uses_text(self):
        """Test that text parsing is used when structured output is disabled."""
        text_response = json.dumps(self.valid_final_response)
        model = self._make_mock_model(text_response=text_response)
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=False,
            max_reprompts=0,
        )
        
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertEqual(result["final_answer"], "Test answer")
    
    def test_structured_output_fails_falls_back_to_text(self):
        """Test fallback to text parsing when structured output fails."""
        text_response = json.dumps(self.valid_final_response)
        model = self._make_mock_model(text_response=text_response, structured_response=None)
        
        # Make structured output fail
        model.with_structured_output.side_effect = TypeError("Model does not support structured output")
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=True,
            max_reprompts=0,
        )
        
        self.assertEqual(result["decision_type"], "FINAL")
    
    def test_salvage_on_non_json_output(self):
        """Test salvage is attempted when JSON parsing fails."""
        non_json_output = """
        decision_type: FINAL
        final_answer: "Salvaged answer"
        brief_rationale: "Salvaged rationale"
        """
        model = self._make_mock_model(text_response=non_json_output)
        model.with_structured_output.side_effect = TypeError()
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            salvage_fn=salvage_reasoner_final,
            use_structured_output=False,
            accept_non_json_final=True,
            max_reprompts=0,
        )
        
        self.assertEqual(result["decision_type"], "FINAL")
        self.assertEqual(result["final_answer"], "Salvaged answer")
    
    def test_reprompt_on_invalid_output(self):
        """Test reprompt is attempted when output is invalid."""
        # First response is invalid (missing required field)
        invalid_response = json.dumps({
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "",  # Empty - invalid
            "brief_rationale": "Test",
        })
        
        # Second response (after reprompt) is valid
        valid_response = json.dumps(self.valid_final_response)
        
        model = MagicMock()
        model.invoke = MagicMock(side_effect=[
            MagicMock(content=invalid_response),
            MagicMock(content=valid_response),
        ])
        model.with_structured_output.side_effect = TypeError()
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=False,
            max_reprompts=1,
        )
        
        # Should have called invoke twice (original + reprompt)
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(result["final_answer"], "Test answer")
    
    def test_max_reprompts_zero_disables_reprompt(self):
        """Test that max_reprompts=0 disables reprompting."""
        invalid_response = json.dumps({
            "decision_type": "INVALID",  # Invalid
            "tool_name": None,
            "tool_args": None,
            "final_answer": None,
            "brief_rationale": "Test",
        })
        
        model = self._make_mock_model(text_response=invalid_response)
        model.with_structured_output.side_effect = TypeError()
        
        with self.assertRaises(JSONParseError):
            invoke_with_unified_fallback(
                model,
                system="test system",
                user="test user",
                schema=self.schema,
                normalizer=normalize_reasoner_decision_obj,
                validator=validate_reasoner_decision_dict,
                use_structured_output=False,
                max_reprompts=0,
            )
        
        # Should have only called invoke once
        self.assertEqual(model.invoke.call_count, 1)
    
    def test_all_fallbacks_exhausted_raises_error(self):
        """Test that JSONParseError is raised when all fallbacks fail."""
        invalid_response = "This is not JSON at all"
        
        model = self._make_mock_model(text_response=invalid_response)
        model.with_structured_output.side_effect = TypeError()
        
        with self.assertRaises(JSONParseError) as ctx:
            invoke_with_unified_fallback(
                model,
                system="test system",
                user="test user",
                schema=self.schema,
                normalizer=normalize_reasoner_decision_obj,
                validator=validate_reasoner_decision_dict,
                use_structured_output=False,
                accept_non_json_final=False,
                max_reprompts=0,
            )
        
        self.assertIn("fallback", str(ctx.exception).lower())
    
    def test_handles_markdown_wrapped_json(self):
        """Test that JSON wrapped in markdown fences is handled."""
        markdown_wrapped = f"```json\n{json.dumps(self.valid_final_response)}\n```"
        
        model = self._make_mock_model(text_response=markdown_wrapped)
        model.with_structured_output.side_effect = TypeError()
        
        result = invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=False,
            max_reprompts=0,
        )
        
        self.assertEqual(result["decision_type"], "FINAL")
    
    def test_logger_receives_appropriate_calls(self):
        """Test that logger methods are called appropriately."""
        text_response = json.dumps(self.valid_final_response)
        model = self._make_mock_model(text_response=text_response)
        model.with_structured_output.side_effect = TypeError()
        
        mock_logger = MagicMock()
        mock_logger.structured_output_attempt = MagicMock()
        mock_logger.structured_output_fallback = MagicMock()
        
        invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=True,
            max_reprompts=0,
            logger=mock_logger,
        )
        
        mock_logger.structured_output_attempt.assert_called_once()
        mock_logger.structured_output_fallback.assert_called_once()


class TestRepromptBehavior(unittest.TestCase):
    """Tests for the reprompt mechanism in the unified fallback."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.schema = get_reasoner_decision_schema([])
        self.valid_final_response = {
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "Test answer",
            "brief_rationale": "Test rationale",
        }
    
    def test_reprompt_includes_error_in_message(self):
        """Test that reprompt message includes the error reason."""
        # First response fails validation
        invalid_response = json.dumps({
            "decision_type": "FINAL",
            "tool_name": None,
            "tool_args": None,
            "final_answer": "",  # Invalid - empty
            "brief_rationale": "Test",
        })
        
        # Second response succeeds
        valid_response = json.dumps(self.valid_final_response)
        
        model = MagicMock()
        invoke_calls = []
        
        def capture_invoke(messages):
            invoke_calls.append(messages)
            if len(invoke_calls) == 1:
                return MagicMock(content=invalid_response)
            return MagicMock(content=valid_response)
        
        model.invoke = MagicMock(side_effect=capture_invoke)
        model.with_structured_output.side_effect = TypeError()
        
        invoke_with_unified_fallback(
            model,
            system="test system",
            user="test user",
            schema=self.schema,
            normalizer=normalize_reasoner_decision_obj,
            validator=validate_reasoner_decision_dict,
            use_structured_output=False,
            max_reprompts=1,
        )
        
        # Check that second call includes error information
        self.assertEqual(len(invoke_calls), 2)
        second_call_user = invoke_calls[1][1].content  # HumanMessage content
        self.assertIn("INVALID", second_call_user)
        self.assertIn("REQUIRED SCHEMA", second_call_user)
    
    def test_multiple_reprompts_decrement_counter(self):
        """Test that max_reprompts decrements correctly."""
        invalid_response = json.dumps({
            "decision_type": "INVALID",
            "tool_name": None,
            "tool_args": None,
            "final_answer": None,
            "brief_rationale": "Test",
        })
        
        model = MagicMock()
        model.invoke = MagicMock(return_value=MagicMock(content=invalid_response))
        model.with_structured_output.side_effect = TypeError()
        
        with self.assertRaises(JSONParseError):
            invoke_with_unified_fallback(
                model,
                system="test system",
                user="test user",
                schema=self.schema,
                normalizer=normalize_reasoner_decision_obj,
                validator=validate_reasoner_decision_dict,
                use_structured_output=False,
                max_reprompts=3,
            )
        
        # Should call original + 3 reprompts = 4 total
        self.assertEqual(model.invoke.call_count, 4)


if __name__ == "__main__":
    unittest.main()
