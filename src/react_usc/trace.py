from __future__ import annotations

from typing import Sequence

from .models import JudgeDecision, ReasonerDecision
from .utils import safe_json_dumps, truncate


def trace_candidates(*, step: int, k: int, valid: Sequence[ReasonerDecision], invalid: Sequence[str]) -> None:
    print(f"\nStep {step}: reasoner candidates (K={k})")
    if invalid:
        print("  Invalid candidates:")
        for r in invalid[:8]:
            print(f"   - {truncate(r, 260)}")
        if len(invalid) > 8:
            print(f"   - ... ({len(invalid) - 8} more)")
    if not valid:
        print("  Valid candidates: (none)")
        return
    print("  Valid candidates:")
    for i, c in enumerate(valid):
        if c.decision_type == "TOOL_CALL":
            print(
                f"   [{i}] TOOL_CALL tool={c.tool_name} "
                f"args={truncate(safe_json_dumps(c.tool_args), 140)} "
                f"| rationale={truncate(c.brief_rationale, 120)}"
            )
        else:
            print(f"   [{i}] FINAL | rationale={truncate(c.brief_rationale, 120)}")


def trace_judge(*, step: int, decision: JudgeDecision) -> None:
    if decision.decision_type == "FINAL":
        print(
            f"Step {step}: judge => FINAL (selected_index={decision.selected_index}) "
            f"because: {truncate(decision.justification, 220)}"
        )
    else:
        print(
            f"Step {step}: judge => TOOL_CALL {decision.tool_name} "
            f"(selected_index={decision.selected_index}) because: {truncate(decision.justification, 220)}"
        )


