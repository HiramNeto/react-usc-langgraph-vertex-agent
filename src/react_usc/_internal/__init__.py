"""
Internal implementation details for the ReAct USC Agent.

WARNING: This module contains private implementation details that may change
without notice between versions. Do not import from this module directly
in production code.

For public APIs, use the main `react_usc` package exports.
"""
from __future__ import annotations

# LLM I/O utilities
from .llm_io import (
    invoke_chat_structured_obj,
    invoke_chat_text,
    invoke_with_unified_fallback,
    json_loads_object,
)

# Decision normalizers
from .normalizers import (
    normalize_judge_decision_obj,
    normalize_reasoner_decision_obj,
    normalize_reflection_decision_obj,
)

# Prompt builders
from .prompts import (
    build_judge_prompt,
    build_reasoner_prompt,
    build_reflection_prompt,
    build_tools_block,
)

# Pydantic schemas for structured output
from .schema import (
    JUDGE_DECISION_SCHEMA,
    REASONER_DECISION_SCHEMA,
    REFLECTION_DECISION_SCHEMA,
    get_judge_decision_schema,
    get_reasoner_decision_schema,
    get_reflection_decision_schema,
)

# Non-JSON salvage utilities
from .salvage import (
    salvage_judge_final,
    salvage_non_json_final_answer,
    salvage_reasoner_final,
    salvage_reflection_final,
)

# Validation utilities
from .validation import (
    validate_json_obj,
    validate_judge_decision_dict,
    validate_reasoner_decision_dict,
    validate_reflection_decision_dict,
)

# Common utilities
from .utils import (
    build_state_summary,
    extract_json_block,
    format_error,
    is_json_like,
    safe_json_dumps,
    simple_word_hits,
    truncate,
)

__all__ = [
    # LLM I/O
    "invoke_chat_text",
    "invoke_chat_structured_obj",
    "invoke_with_unified_fallback",
    "json_loads_object",
    # Normalizers
    "normalize_reasoner_decision_obj",
    "normalize_judge_decision_obj",
    "normalize_reflection_decision_obj",
    # Prompts
    "build_tools_block",
    "build_reasoner_prompt",
    "build_judge_prompt",
    "build_reflection_prompt",
    # Schema
    "get_reasoner_decision_schema",
    "get_judge_decision_schema",
    "get_reflection_decision_schema",
    "REASONER_DECISION_SCHEMA",
    "JUDGE_DECISION_SCHEMA",
    "REFLECTION_DECISION_SCHEMA",
    # Salvage
    "salvage_non_json_final_answer",
    "salvage_reasoner_final",
    "salvage_judge_final",
    "salvage_reflection_final",
    # Validation
    "validate_json_obj",
    "validate_reasoner_decision_dict",
    "validate_judge_decision_dict",
    "validate_reflection_decision_dict",
    # Utils
    "truncate",
    "build_state_summary",
    "safe_json_dumps",
    "simple_word_hits",
    "format_error",
    "is_json_like",
    "extract_json_block",
]
