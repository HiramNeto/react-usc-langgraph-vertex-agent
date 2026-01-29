"""
Tests for configuration classes (ModelConfig, RetryConfig, AgentConfig).

These tests cover:
- Field validation and boundary conditions
- Factory methods (default, none)
- Immutability (frozen dataclasses)
"""
from __future__ import annotations

import unittest

from react_usc import AgentConfig, ModelConfig, RetryConfig
from react_usc.types import AgentConstants


# =============================================================================
# Test: ModelConfig
# =============================================================================


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig validation and behavior."""

    def test_valid_model_config(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(name="gemini-1.5-pro", temperature=0.7)
        
        self.assertEqual(config.name, "gemini-1.5-pro")
        self.assertEqual(config.temperature, 0.7)
        self.assertIsNone(config.max_tokens)

    def test_valid_model_config_with_max_tokens(self):
        """Test creating a ModelConfig with max_tokens."""
        config = ModelConfig(name="gpt-4", temperature=0.5, max_tokens=1000)
        
        self.assertEqual(config.name, "gpt-4")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1000)

    def test_empty_name_rejected(self):
        """Test that empty model name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="", temperature=0.5)
        
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_name_rejected(self):
        """Test that whitespace-only model name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="   ", temperature=0.5)
        
        self.assertIn("empty", str(ctx.exception).lower())

    def test_temperature_below_minimum(self):
        """Test that temperature below MIN_TEMPERATURE raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="model", temperature=-0.1)
        
        self.assertIn("temperature", str(ctx.exception).lower())

    def test_temperature_above_maximum(self):
        """Test that temperature above MAX_TEMPERATURE raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="model", temperature=2.5)
        
        self.assertIn("temperature", str(ctx.exception).lower())

    def test_temperature_at_boundaries(self):
        """Test that temperature at exact boundaries is valid."""
        # Minimum temperature
        config_min = ModelConfig(
            name="model",
            temperature=AgentConstants.MIN_TEMPERATURE,
        )
        self.assertEqual(config_min.temperature, AgentConstants.MIN_TEMPERATURE)
        
        # Maximum temperature
        config_max = ModelConfig(
            name="model",
            temperature=AgentConstants.MAX_TEMPERATURE,
        )
        self.assertEqual(config_max.temperature, AgentConstants.MAX_TEMPERATURE)

    def test_invalid_max_tokens_zero(self):
        """Test that zero max_tokens raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="model", temperature=0.5, max_tokens=0)
        
        self.assertIn("max_tokens", str(ctx.exception).lower())

    def test_invalid_max_tokens_negative(self):
        """Test that negative max_tokens raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(name="model", temperature=0.5, max_tokens=-100)
        
        self.assertIn("max_tokens", str(ctx.exception).lower())

    def test_frozen_dataclass(self):
        """Test that ModelConfig is immutable (frozen)."""
        config = ModelConfig(name="model", temperature=0.5)
        
        with self.assertRaises(AttributeError):
            config.name = "new_name"  # type: ignore


# =============================================================================
# Test: RetryConfig
# =============================================================================


class TestRetryConfig(unittest.TestCase):
    """Test RetryConfig validation and factory methods."""

    def test_valid_retry_config(self):
        """Test creating a valid RetryConfig."""
        config = RetryConfig(max_retries=3, backoff_seconds=1.5)
        
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.backoff_seconds, 1.5)

    def test_zero_retries_valid(self):
        """Test that zero retries is valid (no retries)."""
        config = RetryConfig(max_retries=0, backoff_seconds=0.0)
        
        self.assertEqual(config.max_retries, 0)
        self.assertEqual(config.backoff_seconds, 0.0)

    def test_negative_max_retries_rejected(self):
        """Test that negative max_retries raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RetryConfig(max_retries=-1, backoff_seconds=1.0)
        
        self.assertIn("max_retries", str(ctx.exception).lower())

    def test_negative_backoff_seconds_rejected(self):
        """Test that negative backoff_seconds raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RetryConfig(max_retries=2, backoff_seconds=-0.5)
        
        self.assertIn("backoff_seconds", str(ctx.exception).lower())

    def test_default_factory(self):
        """Test RetryConfig.default() factory method."""
        config = RetryConfig.default()
        
        self.assertEqual(config.max_retries, AgentConstants.DEFAULT_MAX_RETRIES)
        self.assertEqual(config.backoff_seconds, AgentConstants.DEFAULT_BACKOFF_SECONDS)

    def test_none_factory(self):
        """Test RetryConfig.none() factory method."""
        config = RetryConfig.none()
        
        self.assertEqual(config.max_retries, 0)
        self.assertEqual(config.backoff_seconds, 0.0)

    def test_frozen_dataclass(self):
        """Test that RetryConfig is immutable (frozen)."""
        config = RetryConfig.default()
        
        with self.assertRaises(AttributeError):
            config.max_retries = 10  # type: ignore


# =============================================================================
# Test: AgentConfig
# =============================================================================


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig validation and factory methods."""

    def setUp(self):
        """Create common test fixtures."""
        self.model_config = ModelConfig(name="test-model", temperature=0.7)

    def test_valid_agent_config(self):
        """Test creating a valid AgentConfig."""
        config = AgentConfig(
            system_prompt="You are a helpful assistant.",
            k_paths=3,
            max_steps=10,
            reasoner_model=self.model_config,
            judge_model=self.model_config,
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.default(),
            trace=False,
            tool_result_max_chars=4000,
        )
        
        self.assertEqual(config.k_paths, 3)
        self.assertEqual(config.max_steps, 10)
        self.assertEqual(config.selection_strategy, "select_one")

    def test_k_paths_below_minimum(self):
        """Test that k_paths below MIN_K_PATHS raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=0,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
        
        self.assertIn("k_paths", str(ctx.exception).lower())

    def test_k_paths_above_maximum(self):
        """Test that k_paths above MAX_K_PATHS raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=101,  # MAX_K_PATHS is 100
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
        
        self.assertIn("k_paths", str(ctx.exception).lower())

    def test_k_paths_at_boundaries(self):
        """Test that k_paths at exact boundaries is valid."""
        # Minimum
        config_min = AgentConfig(
            system_prompt="test",
            k_paths=AgentConstants.MIN_K_PATHS,
            max_steps=10,
            reasoner_model=self.model_config,
            judge_model=self.model_config,
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=4000,
        )
        self.assertEqual(config_min.k_paths, AgentConstants.MIN_K_PATHS)
        
        # Maximum
        config_max = AgentConfig(
            system_prompt="test",
            k_paths=AgentConstants.MAX_K_PATHS,
            max_steps=10,
            reasoner_model=self.model_config,
            judge_model=self.model_config,
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=4000,
        )
        self.assertEqual(config_max.k_paths, AgentConstants.MAX_K_PATHS)

    def test_max_steps_below_minimum(self):
        """Test that max_steps below MIN_MAX_STEPS raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=0,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
        
        self.assertIn("max_steps", str(ctx.exception).lower())

    def test_max_steps_above_maximum(self):
        """Test that max_steps above MAX_MAX_STEPS raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=51,  # MAX_MAX_STEPS is 50
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
        
        self.assertIn("max_steps", str(ctx.exception).lower())

    def test_invalid_timeout_seconds_zero(self):
        """Test that zero timeout_seconds raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
                timeout_seconds=0.0,
            )
        
        self.assertIn("timeout_seconds", str(ctx.exception).lower())

    def test_invalid_timeout_seconds_negative(self):
        """Test that negative timeout_seconds raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
                timeout_seconds=-5.0,
            )
        
        self.assertIn("timeout_seconds", str(ctx.exception).lower())

    def test_invalid_tool_result_max_chars_negative(self):
        """Test that negative tool_result_max_chars raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="select_one",
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=-100,
            )
        
        self.assertIn("tool_result_max_chars", str(ctx.exception).lower())

    def test_invalid_selection_strategy(self):
        """Test that invalid selection_strategy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy="invalid_strategy",  # type: ignore
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
        
        self.assertIn("selection_strategy", str(ctx.exception).lower())

    def test_valid_selection_strategies(self):
        """Test both valid selection strategies."""
        for strategy in ("select_one", "synthesize_one"):
            config = AgentConfig(
                system_prompt="test",
                k_paths=3,
                max_steps=10,
                reasoner_model=self.model_config,
                judge_model=self.model_config,
                selection_strategy=strategy,  # type: ignore
                allow_tool_synthesis=False,
                llm_retry=RetryConfig.none(),
                trace=False,
                tool_result_max_chars=4000,
            )
            self.assertEqual(config.selection_strategy, strategy)

    def test_default_factory(self):
        """Test AgentConfig.default() factory method."""
        config = AgentConfig.default()
        
        self.assertEqual(config.k_paths, 3)
        self.assertEqual(config.max_steps, 10)
        self.assertEqual(config.selection_strategy, "select_one")
        self.assertFalse(config.allow_tool_synthesis)
        self.assertFalse(config.trace)
        self.assertEqual(
            config.tool_result_max_chars,
            AgentConstants.DEFAULT_TOOL_RESULT_MAX_CHARS,
        )

    def test_default_factory_with_custom_values(self):
        """Test AgentConfig.default() with custom parameters."""
        config = AgentConfig.default(
            system_prompt="Custom prompt",
            k_paths=5,
            max_steps=15,
            model_name="custom-model",
        )
        
        self.assertEqual(config.system_prompt, "Custom prompt")
        self.assertEqual(config.k_paths, 5)
        self.assertEqual(config.max_steps, 15)
        self.assertEqual(config.reasoner_model.name, "custom-model")
        self.assertEqual(config.judge_model.name, "custom-model")

    def test_frozen_dataclass(self):
        """Test that AgentConfig is immutable (frozen)."""
        config = AgentConfig.default()
        
        with self.assertRaises(AttributeError):
            config.k_paths = 10  # type: ignore

    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        config = AgentConfig(
            system_prompt="test",
            k_paths=3,
            max_steps=10,
            reasoner_model=self.model_config,
            judge_model=self.model_config,
            selection_strategy="select_one",
            allow_tool_synthesis=False,
            llm_retry=RetryConfig.none(),
            trace=False,
            tool_result_max_chars=4000,
        )
        
        self.assertFalse(config.truncate_agent_observations)
        self.assertEqual(config.timeout_seconds, AgentConstants.DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(config.use_structured_output)
        self.assertFalse(config.log_structured_output)
        self.assertFalse(config.accept_non_json_final)


if __name__ == "__main__":
    unittest.main()
