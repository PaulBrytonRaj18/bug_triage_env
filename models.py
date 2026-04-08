# bug_triage_env/models.py
#
# Defines the typed Pydantic models for the Bug Triage Environment.
# These are the core contracts between the environment and any agent.
#
# BugAction  — what the agent sends into step()
# BugObservation — what the agent receives back
#
# Both inherit from openenv-core base types so they comply with
# the OpenEnv spec and pass openenv validate.

from __future__ import annotations
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action — what the agent does each step
# ---------------------------------------------------------------------------


class BugAction(BaseModel):
    """
    One triage decision made by the agent for a single issue.

    The agent must always provide action_type, severity, and issue_id.
    duplicate_of is only required when action_type is 'label_duplicate'.
    reasoning is optional and never scored — it exists for logging/debugging.
    """

    action_type: str = Field(
        ...,
        description=(
            "The label to assign to this issue. Must be one of:\n"
            "  label_bug       — a confirmed software defect\n"
            "  label_feature   — a request for new functionality\n"
            "  label_duplicate — same problem as an earlier issue\n"
            "  label_invalid   — not actionable, spam, test, or nonsense\n"
            "  label_question  — user asking a support question"
        ),
    )

    severity: str = Field(
        ...,
        description=(
            "Priority level assigned to this issue. Must be one of:\n"
            "  P0 — Critical: system down, data loss, security breach\n"
            "  P1 — High: major feature broken, many users affected\n"
            "  P2 — Medium: feature broken but workaround exists\n"
            "  P3 — Low: cosmetic issue, minor inconvenience, docs"
        ),
    )

    issue_id: str = Field(
        ...,
        description="The exact issue_id from the current observation. Must match.",
    )

    duplicate_of: Optional[str] = Field(
        default=None,
        description=(
            "Required if action_type is 'label_duplicate'. "
            "Provide the issue_id of the original issue this duplicates."
        ),
    )

    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "One-sentence explanation of your decision. "
            "Not scored. Used for logging and debugging only."
        ),
    )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    VALID_LABELS: ClassVar[FrozenSet[str]] = frozenset(
        {
            "label_bug",
            "label_feature",
            "label_duplicate",
            "label_invalid",
            "label_question",
        }
    )

    VALID_SEVERITIES: ClassVar[FrozenSet[str]] = frozenset({"P0", "P1", "P2", "P3"})

    def is_valid_label(self) -> bool:
        return self.action_type in self.VALID_LABELS

    def is_valid_severity(self) -> bool:
        return self.severity in self.VALID_SEVERITIES


# ---------------------------------------------------------------------------
# Observation — what the agent receives after reset() or step()
# ---------------------------------------------------------------------------


class BugObservation(BaseModel):
    """
    Everything the agent can see about the current state of the environment.

    After reset()  — contains the first issue in the inbox.
    After step()   — contains the NEXT issue to triage (or the last issue
                     repeated with done=True when the inbox is exhausted).

    The agent should use:
      - title + body            to understand the problem
      - has_stack_trace         as a strong signal for label_bug
      - mentioned_components    to help assess severity
      - issues_remaining        to track episode progress
      - last_action_result      to understand how the last step was scored
      - cumulative_score        to track running performance
      - done                    to know when the episode is over
    """

    # --- Current issue fields ---
    issue_id: str = Field(..., description="Unique identifier for this issue (e.g. ISS-1042)")
    title: str = Field(..., description="Short title of the issue as filed by the reporter")
    body: str = Field(..., description="Full description, reproduction steps, and any stack traces")
    reporter: str = Field(..., description="Username of the person who filed the issue")
    created_at: str = Field(..., description="Date the issue was created (YYYY-MM-DD)")
    comments_count: int = Field(..., description="Number of follow-up comments on the issue")
    has_stack_trace: bool = Field(
        ...,
        description="True if the issue body contains a stack trace. Strong signal for label_bug.",
    )
    mentioned_components: List[str] = Field(
        default_factory=list,
        description=(
            "System components referenced in the issue body. "
            "Possible values: auth, payment, dashboard, api, database, "
            "notifications, search, profile"
        ),
    )

    # --- Episode progress fields ---
    issues_remaining: int = Field(
        ...,
        description="Number of issues left to triage after the current one (0 means this is the last).",
    )
    last_action_result: str = Field(
        ...,
        description=(
            "Human-readable feedback on the previous action. "
            "On reset() this is 'Environment reset. Start triaging.'"
        ),
    )
    cumulative_score: float = Field(
        ...,
        description="Running total score for this episode. Normalized at episode end.",
        ge=0.0,
    )
    done: bool = Field(
        ...,
        description="True when all issues in the inbox have been triaged. Episode is over.",
    )


# ---------------------------------------------------------------------------
# Step Result — return value from step(), per OpenEnv spec
# ---------------------------------------------------------------------------


class BugStepResult(BaseModel):
    """
    Return value from step(), complying with OpenEnv spec.

    OpenEnv requires: step(action) -> (observation, reward, done, info)
    This model wraps all four components.

    Attribute delegation: properties on BugObservation are accessible directly
    on BugStepResult (e.g. result.cumulative_score reads result.observation.cumulative_score).
    This ensures backward compatibility with existing code that treats the step()
    return value as an observation object.
    """

    observation: BugObservation
    reward: float = Field(..., ge=0.0, le=1.0, description="Step reward in [0.0, 1.0]")
    done: bool = Field(..., description="True when episode is over")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug metadata (last_action_result, step_count, etc.)",
    )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to observation for backward compatibility."""
        if name.startswith("_") or name in ("observation", "reward", "done", "info"):
            raise AttributeError(name)
        return getattr(self.observation, name)
