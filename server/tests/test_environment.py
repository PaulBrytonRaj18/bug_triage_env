# bug_triage_env/server/tests/test_environment.py
#
# Full test suite for Phase 2 — environment logic and graders.
#
# Run with:
#   cd bug_triage_env
#   pip install pytest
#   pytest server/tests/test_environment.py -v
#
# All tests are pure Python — no server needs to be running.
# Tests cover: reset, step, reward calculation, episode completion,
# grader accuracy, grader variance, and edge cases.

from __future__ import annotations

import sys
import os
import pytest

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from models import BugAction, BugObservation, BugStepResult
from issue_generator import generate_inbox, strip_ground_truth
from server.environment import BugTriageEnvironment, EpisodeState
from graders import grade_easy, grade_medium, grade_hard, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    return BugTriageEnvironment()


@pytest.fixture
def easy_inbox():
    return generate_inbox("easy")


@pytest.fixture
def medium_inbox():
    return generate_inbox("medium")


@pytest.fixture
def hard_inbox():
    return generate_inbox("hard")


def make_action(
    issue_id: str, label: str = "label_bug", severity: str = "P1", duplicate_of: str = None
) -> BugAction:
    """Helper to build a BugAction quickly."""
    return BugAction(
        action_type=label,
        severity=severity,
        issue_id=issue_id,
        duplicate_of=duplicate_of,
        reasoning="test action",
    )


def perfect_action(issue: dict) -> BugAction:
    """Build a perfectly correct action for a given issue dict."""
    return BugAction(
        action_type=issue["_correct_label"],
        severity=issue["_correct_severity"],
        issue_id=issue["issue_id"],
        duplicate_of=issue.get("_duplicate_of"),
        reasoning="perfect answer",
    )


# ===========================================================================
# RESET TESTS
# ===========================================================================


class TestReset:
    def test_reset_easy_returns_observation(self, env):
        obs = env.reset("easy")
        assert isinstance(obs, BugObservation)
        assert obs.issue_id.startswith("ISS-")
        assert len(obs.title) > 0
        assert len(obs.body) > 0
        assert obs.done is False

    def test_reset_medium_inbox_size(self, env):
        obs = env.reset("medium")
        # 15 issues total, first one shown, 14 remaining
        assert obs.issues_remaining == 14

    def test_reset_hard_inbox_size(self, env):
        obs = env.reset("hard")
        assert obs.issues_remaining == 19

    def test_reset_easy_inbox_size(self, env):
        obs = env.reset("easy")
        assert obs.issues_remaining == 7

    def test_reset_cumulative_score_starts_at_zero(self, env):
        obs = env.reset("easy")
        assert obs.cumulative_score == 0.0

    def test_reset_result_message_contains_task(self, env):
        obs = env.reset("medium")
        assert "medium" in obs.last_action_result.lower()

    def test_reset_invalid_task_falls_back_to_easy(self, env):
        obs = env.reset("impossible_task")
        assert obs.issues_remaining == 7  # easy inbox = 8 issues

    def test_double_reset_gives_fresh_state(self, env):
        obs1 = env.reset("easy")
        env.step(make_action(obs1.issue_id, "label_bug", "P0"))
        obs2 = env.reset("easy")
        assert obs2.cumulative_score == 0.0
        assert obs2.done is False
        assert env.state.step_count == 0

    def test_reset_strips_ground_truth_from_observation(self, env):
        obs = env.reset("easy")
        obs_dict = obs.model_dump()
        for key in obs_dict:
            assert not key.startswith("_"), f"Ground truth field leaked: {key}"


# ===========================================================================
# STEP TESTS
# ===========================================================================


class TestStep:
    def test_step_correct_label_and_severity_gives_full_reward(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        first = inbox[0]

        action = perfect_action(first)
        obs = env.step(action)

        # Perfect label (0.6) + perfect severity (0.3) = 0.9
        assert obs.cumulative_score >= 0.9

    def test_step_correct_label_wrong_severity_gives_partial(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        first = inbox[0]

        # Correct label, severity exactly 1 off
        correct_sev_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        correct_sev = first["_correct_severity"]
        sev_num = correct_sev_order[correct_sev]
        off_by_one_sev = ["P0", "P1", "P2", "P3"][min(sev_num + 1, 3)]

        action = make_action(first["issue_id"], first["_correct_label"], off_by_one_sev)
        obs = env.step(action)

        # 0.6 (label) + 0.15 (partial severity) = 0.75
        assert obs.cumulative_score >= 0.74

    def test_step_wrong_label_gives_lower_reward(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        first = inbox[0]

        # Wrong label
        wrong_label = "label_feature" if first["_correct_label"] != "label_feature" else "label_bug"
        action = make_action(first["issue_id"], wrong_label, first["_correct_severity"])
        obs = env.step(action)

        # Only severity correct (0.3)
        assert obs.cumulative_score <= 0.35

    def test_step_invalid_action_type_penalized(self, env):
        obs = env.reset("easy")
        bad_action = make_action(obs.issue_id, "label_gibberish", "P1")
        result = env.step(bad_action)
        assert result.cumulative_score == 0.0  # penalty applied, clipped to 0
        assert "Invalid action_type" in result.last_action_result

    def test_step_invalid_severity_penalized(self, env):
        obs = env.reset("easy")
        bad_action = make_action(obs.issue_id, "label_bug", "P9")
        result = env.step(bad_action)
        assert "Invalid severity" in result.last_action_result

    def test_step_wrong_issue_id_penalized(self, env):
        env.reset("easy")
        bad_action = make_action("ISS-9999", "label_bug", "P1")
        result = env.step(bad_action)
        assert "Wrong issue_id" in result.last_action_result

    def test_step_does_not_advance_on_invalid_action(self, env):
        obs1 = env.reset("easy")
        issue_id_before = obs1.issue_id

        bad_action = make_action(obs1.issue_id, "GARBAGE", "P1")
        obs2 = env.step(bad_action)

        # Issue should NOT advance on invalid action
        assert obs2.issue_id == issue_id_before

    def test_step_advances_to_next_issue(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")

        obs1_id = inbox[0]["issue_id"]
        obs2_id = inbox[1]["issue_id"]

        action = perfect_action(inbox[0])
        obs_after = env.step(action)

        assert obs_after.issue_id == obs2_id

    def test_step_done_after_last_issue(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")

        for issue in inbox:
            obs = env.step(perfect_action(issue))

        assert obs.done is True
        assert obs.issues_remaining == 0

    def test_step_p0_miss_incurs_penalty(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")

        # Find first P0 bug
        p0_issue = next(
            (
                i
                for i in inbox
                if i["_correct_severity"] == "P0" and i["_correct_label"] == "label_bug"
            ),
            None,
        )
        if p0_issue is None:
            pytest.skip("No P0 bug in easy inbox")

        # Fast-forward to P0 issue
        for issue in inbox:
            if issue["issue_id"] == p0_issue["issue_id"]:
                break
            env.step(perfect_action(issue))

        score_before = env._cumulative_score

        # Mis-label as feature
        wrong = make_action(p0_issue["issue_id"], "label_feature", "P3")
        obs = env.step(wrong)

        # Score should not have increased (penalty applied)
        assert obs.cumulative_score <= score_before

    def test_step_state_increments(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")

        assert env.state.step_count == 0
        env.step(perfect_action(inbox[0]))
        assert env.state.step_count == 1
        env.step(perfect_action(inbox[1]))
        assert env.state.step_count == 2

    def test_step_without_reset_auto_initializes(self, env):
        # Step before reset should not crash
        action = make_action("ISS-1000", "label_bug", "P1")
        obs = env.step(action)
        assert obs is not None

    def test_step_returns_bugstepresult(self, env):
        """step() must return BugStepResult (not just BugObservation)."""
        env.reset("easy")
        action = make_action("ISS-1000", "label_bug", "P1")
        result = env.step(action)
        assert isinstance(result, BugStepResult), f"Expected BugStepResult, got {type(result)}"

    def test_step_bugstepresult_has_all_required_fields(self, env):
        """BugStepResult must have observation, reward, done, info."""
        env.reset("easy")
        action = make_action("ISS-1000", "label_bug", "P1")
        result = env.step(action)
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")
        assert isinstance(result.observation, BugObservation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_reward_in_valid_range(self, env):
        """Step reward must be in [0.0, 1.0]."""
        env.reset("easy")
        action = make_action("ISS-1000", "label_bug", "P1")
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0

    def test_step_done_true_only_at_end(self, env):
        """done=True only when all issues are triaged."""
        inbox = generate_inbox("easy")
        env.reset("easy")
        for i, issue in enumerate(inbox[:-1]):
            result = env.step(perfect_action(issue))
            assert result.done is False, f"done=True at step {i + 1}, expected False"
        last_result = env.step(perfect_action(inbox[-1]))
        assert last_result.done is True

    def test_step_info_contains_debug_metadata(self, env):
        """info dict should contain last_action_result and step_count."""
        env.reset("easy")
        inbox = generate_inbox("easy")
        result = env.step(perfect_action(inbox[0]))
        assert "last_action_result" in result.info
        assert "step_count" in result.info
        assert result.info["step_count"] == 1

    def test_step_observation_matches_old_interface(self, env):
        """BugStepResult supports obs.issue_id style access for backward compat.

        After stepping, the observation shows the NEXT issue to be triaged.
        result.issue_id should delegate to result.observation.issue_id.
        """
        inbox = generate_inbox("easy")
        env.reset("easy")
        result = env.step(perfect_action(inbox[0]))
        assert result.issue_id == inbox[1]["issue_id"]
        assert result.title == inbox[1]["title"]
        assert result.cumulative_score >= 0.0
        assert result.done is False


# ===========================================================================
# GROUND TRUTH ISOLATION TESTS
# ===========================================================================


class TestGroundTruthIsolation:
    def test_observation_has_no_private_fields(self, env):
        obs = env.reset("hard")
        d = obs.model_dump()
        assert "_correct_label" not in d
        assert "_correct_severity" not in d
        assert "_duplicate_of" not in d

    def test_triaged_list_contains_ground_truth_for_graders(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        env.step(perfect_action(inbox[0]))

        triaged = env.triaged
        assert len(triaged) == 1
        assert "correct_label" in triaged[0]
        assert "correct_severity" in triaged[0]

    def test_triaged_list_is_copy(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        env.step(perfect_action(inbox[0]))

        t1 = env.triaged
        t1.append({"fake": True})

        t2 = env.triaged
        assert len(t2) == 1  # original not mutated


# ===========================================================================
# FULL EPISODE TESTS
# ===========================================================================


class TestFullEpisode:
    def _run_perfect_episode(self, env, task_id):
        inbox = generate_inbox(task_id)
        env.reset(task_id)
        last_obs = None
        for issue in inbox:
            last_obs = env.step(perfect_action(issue))
        return last_obs, env.triaged

    def test_perfect_episode_easy_high_score(self, env):
        obs, triaged = self._run_perfect_episode(env, "easy")
        assert obs.done is True
        assert obs.cumulative_score >= 6.0  # 8 issues * 0.9 per issue

    def test_perfect_episode_medium_completes(self, env):
        obs, triaged = self._run_perfect_episode(env, "medium")
        assert obs.done is True
        assert obs.issues_remaining == 0

    def test_perfect_episode_hard_completes(self, env):
        obs, triaged = self._run_perfect_episode(env, "hard")
        assert obs.done is True

    def test_triaged_count_matches_inbox_size(self, env):
        inbox = generate_inbox("easy")
        env.reset("easy")
        for issue in inbox:
            env.step(perfect_action(issue))
        assert len(env.triaged) == len(inbox)


# ===========================================================================
# GRADER TESTS
# ===========================================================================


class TestGraders:
    def _perfect_actions(self, task_id: str):
        inbox = generate_inbox(task_id)
        return [
            {
                "issue_id": i["issue_id"],
                "action": i["_correct_label"],
                "severity": i["_correct_severity"],
                "duplicate_of": i.get("_duplicate_of"),
            }
            for i in inbox
        ]

    def _all_wrong_actions(self, task_id: str):
        inbox = generate_inbox(task_id)
        wrong_label = "label_invalid"
        return [
            {
                "issue_id": i["issue_id"],
                "action": wrong_label,
                "severity": "P3",
                "duplicate_of": None,
            }
            for i in inbox
        ]

    # --- grade_easy ---

    def test_grade_easy_perfect_score_is_one(self):
        score = grade_easy(self._perfect_actions("easy"))
        assert score == 1.0

    def test_grade_easy_all_wrong_score_is_zero(self):
        score = grade_easy(self._all_wrong_actions("easy"))
        assert score <= 0.15  # easy inbox has 1 label_invalid, so "all wrong" gets 1/8 = 0.125

    def test_grade_easy_empty_actions_is_zero(self):
        assert grade_easy([]) == 0.0

    def test_grade_easy_half_correct_is_half(self):
        inbox = generate_inbox("easy")
        actions = []
        for i, issue in enumerate(inbox):
            if i % 2 == 0:
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": issue["_correct_label"],
                        "severity": issue["_correct_severity"],
                        "duplicate_of": issue.get("_duplicate_of"),
                    }
                )
            else:
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": "label_invalid",
                        "severity": "P3",
                        "duplicate_of": None,
                    }
                )
        score = grade_easy(actions)
        assert 0.35 <= score <= 0.70  # alternating correct/wrong on 8 issues

    # --- grade_medium ---

    def test_grade_medium_perfect_score_is_one(self):
        score = grade_medium(self._perfect_actions("medium"))
        assert score == 1.0

    def test_grade_medium_all_wrong_is_low(self):
        score = grade_medium(self._all_wrong_actions("medium"))
        assert score <= 0.15  # easy inbox has 1 label_invalid, so "all wrong" gets 1/8 = 0.125

    def test_grade_medium_empty_is_zero(self):
        assert grade_medium([]) == 0.0

    def test_grade_medium_correct_labels_wrong_dup_ref(self):
        inbox = generate_inbox("medium")
        actions = []
        for issue in inbox:
            if issue["_correct_label"] == "label_duplicate":
                # Correct label but wrong reference
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": "label_duplicate",
                        "severity": issue["_correct_severity"],
                        "duplicate_of": "ISS-9999",  # wrong reference
                    }
                )
            else:
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": issue["_correct_label"],
                        "severity": issue["_correct_severity"],
                        "duplicate_of": None,
                    }
                )
        score = grade_medium(actions)
        # All labels correct (1.0 * 0.7) + dup detection 0 (0 * 0.3) = 0.7
        assert 0.65 <= score <= 0.75

    # --- grade_hard ---

    def test_grade_hard_perfect_score_is_one(self):
        score = grade_hard(self._perfect_actions("hard"))
        assert score == 1.0

    def test_grade_hard_all_wrong_is_zero_or_below(self):
        score = grade_hard(self._all_wrong_actions("hard"))
        assert score <= 0.15  # could be 0 or small positive from random severity

    def test_grade_hard_empty_is_zero(self):
        assert grade_hard([]) == 0.0

    def test_grade_hard_p0_miss_reduces_score(self):
        inbox = generate_inbox("hard")
        # Perfect actions but deliberately mis-label all P0 bugs
        actions = []
        for issue in inbox:
            if issue["_correct_label"] == "label_bug" and issue["_correct_severity"] == "P0":
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": "label_feature",  # miss the P0 bug
                        "severity": "P3",
                        "duplicate_of": None,
                    }
                )
            else:
                actions.append(
                    {
                        "issue_id": issue["issue_id"],
                        "action": issue["_correct_label"],
                        "severity": issue["_correct_severity"],
                        "duplicate_of": issue.get("_duplicate_of"),
                    }
                )
        score_with_misses = grade_hard(actions)
        perfect_score = grade_hard(self._perfect_actions("hard"))
        assert score_with_misses < perfect_score

    # --- Grader variance (disqualification prevention) ---

    def test_all_graders_produce_varied_scores(self):
        """
        Hackathon rule: graders that always return the same score = DQ.
        This test generates 5 action sets with different qualities
        and confirms all graders produce at least 3 distinct values.
        """
        for task_id, grader_fn in [
            ("easy", grade_easy),
            ("medium", grade_medium),
            ("hard", grade_hard),
        ]:
            inbox = generate_inbox(task_id)
            scores = set()

            # Empty
            scores.add(grader_fn([]))

            # All wrong
            wrong = [
                {
                    "issue_id": i["issue_id"],
                    "action": "label_invalid",
                    "severity": "P3",
                    "duplicate_of": None,
                }
                for i in inbox
            ]
            scores.add(grader_fn(wrong))

            # First half correct
            half = []
            for j, issue in enumerate(inbox):
                if j < len(inbox) // 2:
                    half.append(
                        {
                            "issue_id": issue["issue_id"],
                            "action": issue["_correct_label"],
                            "severity": issue["_correct_severity"],
                            "duplicate_of": issue.get("_duplicate_of"),
                        }
                    )
                else:
                    half.append(
                        {
                            "issue_id": issue["issue_id"],
                            "action": "label_invalid",
                            "severity": "P3",
                            "duplicate_of": None,
                        }
                    )
            scores.add(grader_fn(half))

            # All perfect
            perfect = [
                {
                    "issue_id": i["issue_id"],
                    "action": i["_correct_label"],
                    "severity": i["_correct_severity"],
                    "duplicate_of": i.get("_duplicate_of"),
                }
                for i in inbox
            ]
            scores.add(grader_fn(perfect))

            assert len(scores) >= 3, (
                f"Grader '{task_id}' only produced {len(scores)} distinct values: {scores}. "
                f"This will cause disqualification."
            )

    # --- Score bounds ---

    def test_all_grader_scores_in_valid_range(self):
        for task_id, grader_fn in [
            ("easy", grade_easy),
            ("medium", grade_medium),
            ("hard", grade_hard),
        ]:
            inbox = generate_inbox(task_id)
            perfect = [
                {
                    "issue_id": i["issue_id"],
                    "action": i["_correct_label"],
                    "severity": i["_correct_severity"],
                    "duplicate_of": i.get("_duplicate_of"),
                }
                for i in inbox
            ]
            score = grader_fn(perfect)
            assert 0.0 <= score <= 1.0, (
                f"Grader '{task_id}' returned {score} which is outside [0.0, 1.0]"
            )

    def test_task_registry_has_all_tasks(self):
        assert "easy" in TASK_REGISTRY
        assert "medium" in TASK_REGISTRY
        assert "hard" in TASK_REGISTRY

    def test_task_registry_grader_callables(self):
        for task_id, info in TASK_REGISTRY.items():
            assert callable(info["grader"]), f"Grader for '{task_id}' is not callable"


# ===========================================================================
# STATE TESTS
# ===========================================================================


class TestEpisodeState:
    def test_state_has_episode_id(self, env):
        env.reset("easy")
        assert len(env.state.episode_id) > 0

    def test_state_episode_id_changes_on_reset(self, env):
        env.reset("easy")
        id1 = env.state.episode_id
        env.reset("easy")
        id2 = env.state.episode_id
        assert id1 != id2

    def test_state_task_id_reflects_reset(self, env):
        env.reset("hard")
        assert env.state.task_id == "hard"

    def test_state_to_dict(self, env):
        env.reset("medium")
        d = env.state.to_dict()
        assert "episode_id" in d
        assert "step_count" in d
        assert "task_id" in d
        assert d["task_id"] == "medium"
