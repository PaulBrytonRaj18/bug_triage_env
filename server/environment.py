# bug_triage_env/server/environment.py
from __future__ import annotations
import sys, os, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import BugAction, BugObservation, BugStepResult
from issue_generator import generate_inbox, strip_ground_truth

# Reward constants
REWARD_CORRECT_LABEL = 0.60
REWARD_CORRECT_SEVERITY = 0.30
REWARD_SEVERITY_OFF_ONE = 0.15
REWARD_CORRECT_DUPLICATE = 0.10
PENALTY_MISSED_P0 = -0.30
PENALTY_INVALID_ACTION = -0.20
PENALTY_INVALID_SEVERITY = -0.10
PENALTY_WRONG_ISSUE_ID = -0.10

SEVERITY_MAP: Dict[str, int] = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
VALID_LABELS = frozenset(
    {"label_bug", "label_feature", "label_duplicate", "label_invalid", "label_question"}
)
VALID_SEVERITIES = frozenset(SEVERITY_MAP.keys())
VALID_TASK_IDS = frozenset({"easy", "medium", "hard"})


@dataclass
class EpisodeState:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    task_id: str = "easy"

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
        }


class BugTriageEnvironment:
    """
    Stateful bug triage environment.
    Holds a single inbox per episode and evaluates agent triage decisions.
    Thread-safe for single-worker Uvicorn deployments.
    """

    def __init__(self) -> None:
        self._state = EpisodeState()
        self._inbox: List[Dict] = []
        self._current_index: int = 0
        self._cumulative_score: float = 0.0
        self._triaged: List[Dict] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> BugObservation:
        """Start a new episode. Returns first issue as observation."""
        if task_id not in VALID_TASK_IDS:
            task_id = "easy"

        self._state = EpisodeState(task_id=task_id)
        self._inbox = generate_inbox(task_id)
        self._current_index = 0
        self._cumulative_score = 0.0
        self._triaged = []
        self._initialized = True

        return self._build_observation(
            last_action_result=(
                f"Episode started. Task: {task_id}. "
                f"Inbox contains {len(self._inbox)} issues. Start triaging."
            ),
            done=False,
        )

    def step(self, action: BugAction) -> BugStepResult:
        """Process one triage action, compute reward, advance to next issue.

        Returns BugStepResult with (observation, reward, done, info) per OpenEnv spec.
        """
        if not self._initialized:
            self.reset("easy")

        self._state.step_count += 1
        step_reward: float = 0.0
        info: Dict = {}
        last_result: str = ""

        # Validate action_type
        if action.action_type not in VALID_LABELS:
            last_result = (
                f"Invalid action_type '{action.action_type}'. "
                f"Must be one of: {sorted(VALID_LABELS)}. Penalized."
            )
            self._cumulative_score = max(0.0, self._cumulative_score + PENALTY_INVALID_ACTION)
            step_reward = max(0.0, PENALTY_INVALID_ACTION)
            info = {"last_action_result": last_result, "step_count": self._state.step_count}
            return BugStepResult(
                observation=self._build_observation(last_action_result=last_result, done=False),
                reward=step_reward,
                done=False,
                info=info,
            )

        # Validate severity
        if action.severity not in VALID_SEVERITIES:
            last_result = (
                f"Invalid severity '{action.severity}'. Must be one of: P0, P1, P2, P3. Penalized."
            )
            self._cumulative_score = max(0.0, self._cumulative_score + PENALTY_INVALID_SEVERITY)
            step_reward = max(0.0, PENALTY_INVALID_SEVERITY)
            info = {"last_action_result": last_result, "step_count": self._state.step_count}
            return BugStepResult(
                observation=self._build_observation(last_action_result=last_result, done=False),
                reward=step_reward,
                done=False,
                info=info,
            )

        # Validate issue_id matches current issue
        current = self._current_issue()
        if action.issue_id != current["issue_id"]:
            last_result = (
                f"Wrong issue_id '{action.issue_id}'. "
                f"Current issue is '{current['issue_id']}'. Penalized."
            )
            self._cumulative_score = max(0.0, self._cumulative_score + PENALTY_WRONG_ISSUE_ID)
            step_reward = max(0.0, PENALTY_WRONG_ISSUE_ID)
            info = {"last_action_result": last_result, "step_count": self._state.step_count}
            return BugStepResult(
                observation=self._build_observation(last_action_result=last_result, done=False),
                reward=step_reward,
                done=False,
                info=info,
            )

        # Evaluate action against ground truth
        reward, result = self._evaluate_action(action, current)

        # Record for graders
        self._triaged.append(
            {
                "issue_id": action.issue_id,
                "action": action.action_type,
                "severity": action.severity,
                "duplicate_of": action.duplicate_of,
                "reasoning": action.reasoning,
                "reward": reward,
                "correct_label": current["_correct_label"],
                "correct_severity": current["_correct_severity"],
            }
        )

        self._cumulative_score = max(0.0, self._cumulative_score + reward)
        self._current_index += 1

        done = self._current_index >= len(self._inbox)
        step_reward = round(max(0.0, min(1.0, reward)), 3)
        info = {"last_action_result": result, "step_count": self._state.step_count}

        return BugStepResult(
            observation=self._build_observation(last_action_result=result, done=done),
            reward=step_reward,
            done=done,
            info=info,
        )

    @property
    def state(self) -> EpisodeState:
        return self._state

    @property
    def triaged(self) -> List[Dict]:
        """Full list of triage decisions this episode (for graders)."""
        return list(self._triaged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_penalty(self, penalty: float, message: str) -> BugObservation:
        """Apply a penalty without advancing the inbox pointer."""
        self._cumulative_score = max(0.0, self._cumulative_score + penalty)
        return self._build_observation(last_action_result=message, done=False)

    def _current_issue(self) -> Dict:
        idx = min(self._current_index, len(self._inbox) - 1)
        return self._inbox[idx]

    def _evaluate_action(self, action: BugAction, issue: Dict) -> Tuple[float, str]:
        """Compare agent action to ground truth. Returns (reward, feedback)."""
        correct_label = issue["_correct_label"]
        correct_severity = issue["_correct_severity"]
        correct_dup = issue.get("_duplicate_of")

        reward = 0.0
        notes: List[str] = []

        # Label scoring (60%)
        if action.action_type == correct_label:
            reward += REWARD_CORRECT_LABEL
            notes.append(f"Correct label ({correct_label})")
        else:
            notes.append(f"Wrong label: got '{action.action_type}', expected '{correct_label}'")

        # Severity scoring (30% exact, 15% off-by-one)
        a_sev = SEVERITY_MAP.get(action.severity, 3)
        c_sev = SEVERITY_MAP.get(correct_severity, 3)
        diff = abs(a_sev - c_sev)
        if diff == 0:
            reward += REWARD_CORRECT_SEVERITY
            notes.append(f"Correct severity ({correct_severity})")
        elif diff == 1:
            reward += REWARD_SEVERITY_OFF_ONE
            notes.append(
                f"Severity off by 1: got '{action.severity}', "
                f"expected '{correct_severity}' (+0.15 partial)"
            )
        else:
            notes.append(
                f"Severity far off: got '{action.severity}', expected '{correct_severity}'"
            )

        # Duplicate reference bonus (10%)
        if correct_label == "label_duplicate" and action.action_type == "label_duplicate":
            if action.duplicate_of and action.duplicate_of == correct_dup:
                reward += REWARD_CORRECT_DUPLICATE
                notes.append(f"Correct duplicate reference ({correct_dup})")
            elif action.duplicate_of:
                notes.append(
                    f"Wrong duplicate reference: got '{action.duplicate_of}', expected '{correct_dup}'"
                )
            else:
                notes.append("Duplicate labeled but 'duplicate_of' not provided")

        # P0 miss penalty
        if correct_label == "label_bug" and correct_severity == "P0":
            if action.action_type != "label_bug":
                reward += PENALTY_MISSED_P0
                notes.append(
                    f"PENALTY: Missed P0 critical bug (labeled '{action.action_type}') "
                    f"{PENALTY_MISSED_P0}"
                )

        reward = round(max(0.0, min(1.0, reward)), 3)
        result = " | ".join(notes) + f" => step_reward: {reward:.3f}"
        return reward, result

    def _build_observation(self, last_action_result: str, done: bool) -> BugObservation:
        """Build BugObservation, stripping all ground truth fields."""
        issue = self._current_issue()
        clean = strip_ground_truth(issue)

        issues_remaining = max(0, len(self._inbox) - self._current_index - 1)
        if done:
            issues_remaining = 0

        return BugObservation(
            issue_id=clean["issue_id"],
            title=clean["title"],
            body=clean["body"],
            reporter=clean["reporter"],
            created_at=clean["created_at"],
            comments_count=clean["comments_count"],
            has_stack_trace=clean["has_stack_trace"],
            mentioned_components=clean["mentioned_components"],
            issues_remaining=issues_remaining,
            last_action_result=last_action_result,
            cumulative_score=round(self._cumulative_score, 4),
            done=done,
        )
