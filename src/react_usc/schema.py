from __future__ import annotations

from typing import Any, Dict, Optional

try:
    # LangChain structured output works best with Pydantic schemas.
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[misc,assignment]

try:
    # Pydantic v2
    from pydantic import ConfigDict  # type: ignore

    _HAS_CONFIGDICT = True
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]
    _HAS_CONFIGDICT = False

try:
    from typing import Literal
except Exception:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore


class ReasonerDecisionStructured(BaseModel):  # type: ignore[misc,valid-type]
    """
    Pydantic schema used with LangChain `with_structured_output(...)`.

    Cross-field constraints (e.g. tool_name required for TOOL_CALL) are still enforced
    by our existing validators in `validation.py` so we keep this lightweight.
    """

    decision_type: Literal["TOOL_CALL", "FINAL"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    brief_rationale: str
    expected_signal: Optional[str] = None

    if _HAS_CONFIGDICT:
        model_config = ConfigDict(extra="forbid")  # type: ignore[assignment]
    else:

        class Config:  # pragma: no cover
            extra = "forbid"


class JudgeDecisionStructured(BaseModel):  # type: ignore[misc,valid-type]
    """
    Pydantic schema used with LangChain `with_structured_output(...)`.
    """

    decision_type: Literal["TOOL_CALL", "FINAL"]
    selected_index: Optional[int] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    justification: str

    if _HAS_CONFIGDICT:
        model_config = ConfigDict(extra="forbid")  # type: ignore[assignment]
    else:

        class Config:  # pragma: no cover
            extra = "forbid"


