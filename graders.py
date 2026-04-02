# bug_triage_env/graders.py
#
# Three deterministic task graders for the Bug Triage Environment.
#
# Each grader takes a list of action dicts (the agent's decisions during
# an episode) and returns a float in [0.0, 1.0].
#
# Graders are intentionally STATELESS and DETERMINISTIC:
#   - They do not depend on the live environment state.
#   - They load a fresh deterministic inbox via generate_inbox().
#   - The same actions always produce the same score.
#   - Different actions always produce different scores (no flat returns).
#
# This satisfies the hackathon requirement:
#   "Graders must have clear, deterministic success/failure criteria."
#   "Graders that always return the same score = disqualification."
#
# Grader scoring logic:
#
#   grade_easy   -- label accuracy only (label match / total issues)
#
#   grade_medium -- 70% label accuracy + 30% duplicate detection accuracy
#
#   grade_hard   -- 50% label accuracy + 30% severity accuracy
#                   + 20% duplicate detection, minus P0 miss penalties
#
# Action dict format (what env.triaged returns):
#   {
#     "issue_id":      str,
#     "action":        str,   -- the label the agent chose
#     "severity":      str,   -- the severity the agent chose
#     "duplicate_of":  str | None,
#     "reasoning":     str | None,
#     "reward":        float, -- not used by graders (live reward, not grader score)
#     "correct_label": str,   -- ground truth (recorded by environment)
#     "correct_severity": str,
#   }
#
# Usage:
#   from graders import grade_easy, grade_medium, grade_hard, TASK_REGISTRY
#   score = grade_easy(env.triaged)

from __future__ import annotations

import sys
import os
from typing import Dict, List, Any

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from issue_generator import generate_inbox


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: Dict[str, int] = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _build_ground_truth_map(task_id: str) -> Dict[str, Dict]:
    """
    Load a deterministic inbox and return a dict keyed by issue_id.
    Used by all graders to look up correct labels without depending on
    the live environment state.
    """
    inbox = generate_inbox(task_id)
    return {issue["issue_id"]: issue for issue in inbox}


def _safe_score(value: float) -> float:
    """Clip and round to 3 decimal places."""
    return round(max(0.0, min(1.0, value)), 3)


# ---------------------------------------------------------------------------
# Grade Easy
# ---------------------------------------------------------------------------

def grade_easy(triaged_actions: List[Dict[str, Any]]) -> float:
    """
    Easy task grader: pure label accuracy.

    Score = (number of correctly labeled issues) / (total issues in easy inbox)

    An issue is "correctly labeled" if the agent's action matches the
    ground truth label exactly.

    Args:
        triaged_actions: List of action dicts from env.triaged

    Returns:
        Float in [0.0, 1.0]
    """
    gt_map = _build_ground_truth_map("easy")
    total = len(gt_map)

    if total == 0:
        return 0.0

    if not triaged_actions:
        return 0.0

    correct = 0
    seen_ids = set()

    for action in triaged_actions:
        issue_id = action.get("issue_id")
        if issue_id not in gt_map:
            continue
        if issue_id in seen_ids:
            continue  # don't double-count if agent retried same issue
        seen_ids.add(issue_id)

        ground_truth_label = gt_map[issue_id]["_correct_label"]
        agent_label = action.get("action", "")

        if agent_label == ground_truth_label:
            correct += 1

    return _safe_score(correct / total)


# ---------------------------------------------------------------------------
# Grade Medium
# ---------------------------------------------------------------------------

def grade_medium(triaged_actions: List[Dict[str, Any]]) -> float:
    """
    Medium task grader: label accuracy + duplicate detection.

    Score = 0.70 * label_accuracy + 0.30 * duplicate_detection_accuracy

    label_accuracy:
        Fraction of all 15 issues with correct labels.

    duplicate_detection_accuracy:
        Among issues that ARE duplicates, what fraction did the agent:
        (a) correctly label as 'label_duplicate', AND
        (b) correctly identify the original issue (duplicate_of field)?

    Full duplicate credit requires BOTH the label AND the correct reference.

    Args:
        triaged_actions: List of action dicts from env.triaged

    Returns:
        Float in [0.0, 1.0]
    """
    gt_map = _build_ground_truth_map("medium")
    total = len(gt_map)

    if total == 0 or not triaged_actions:
        return 0.0

    # Find all ground-truth duplicate issues
    duplicate_ids = {
        iid: issue
        for iid, issue in gt_map.items()
        if issue["_correct_label"] == "label_duplicate"
    }
    total_duplicates = len(duplicate_ids)

    label_correct = 0
    dup_fully_correct = 0
    seen_ids: set = set()

    for action in triaged_actions:
        issue_id = action.get("issue_id")
        if issue_id not in gt_map or issue_id in seen_ids:
            continue
        seen_ids.add(issue_id)

        ground_truth = gt_map[issue_id]
        correct_label = ground_truth["_correct_label"]
        agent_label = action.get("action", "")

        # Label accuracy
        if agent_label == correct_label:
            label_correct += 1

        # Duplicate detection: need correct label AND correct reference
        if correct_label == "label_duplicate":
            if agent_label == "label_duplicate":
                correct_dup_ref = ground_truth.get("_duplicate_of")
                agent_dup_ref = action.get("duplicate_of")
                if agent_dup_ref and agent_dup_ref == correct_dup_ref:
                    dup_fully_correct += 1

    label_score = label_correct / total
    dup_score = dup_fully_correct / max(total_duplicates, 1)

    combined = 0.70 * label_score + 0.30 * dup_score
    return _safe_score(combined)


# ---------------------------------------------------------------------------
# Grade Hard
# ---------------------------------------------------------------------------

def grade_hard(triaged_actions: List[Dict[str, Any]]) -> float:
    """
    Hard task grader: label accuracy + severity accuracy + duplicate detection.
    Penalizes missed P0 critical bugs.

    Score = (
        0.50 * label_accuracy
      + 0.30 * severity_accuracy
      + 0.20 * duplicate_detection_accuracy
      - sum(p0_miss_penalties)
    )

    label_accuracy:
        Fraction of all 20 issues with correct labels.

    severity_accuracy:
        1.0 point for exact severity match.
        0.5 points for severity off by exactly 1 level.
        0.0 otherwise.
        Average across all issues.

    duplicate_detection_accuracy:
        Full credit requires correct label AND correct reference.

    p0_miss_penalty:
        For each P0 critical bug that the agent labeled as anything
        other than 'label_bug', subtract 0.15 from the final score.
        (Same penalty as environment step, but applied globally here.)

    Args:
        triaged_actions: List of action dicts from env.triaged

    Returns:
        Float in [0.0, 1.0]
    """
    gt_map = _build_ground_truth_map("hard")
    total = len(gt_map)

    if total == 0 or not triaged_actions:
        return 0.0

    duplicate_ids = {
        iid: issue
        for iid, issue in gt_map.items()
        if issue["_correct_label"] == "label_duplicate"
    }
    total_duplicates = len(duplicate_ids)

    label_correct = 0
    severity_score_sum = 0.0
    dup_fully_correct = 0
    p0_penalty = 0.0
    seen_ids: set = set()

    for action in triaged_actions:
        issue_id = action.get("issue_id")
        if issue_id not in gt_map or issue_id in seen_ids:
            continue
        seen_ids.add(issue_id)

        ground_truth = gt_map[issue_id]
        correct_label = ground_truth["_correct_label"]
        correct_severity = ground_truth["_correct_severity"]
        agent_label = action.get("action", "")
        agent_severity = action.get("severity", "P3")

        # Label accuracy
        if agent_label == correct_label:
            label_correct += 1

        # Severity accuracy (with partial credit)
        a_sev = _SEVERITY_ORDER.get(agent_severity, 3)
        c_sev = _SEVERITY_ORDER.get(correct_severity, 3)
        diff = abs(a_sev - c_sev)
        if diff == 0:
            severity_score_sum += 1.0
        elif diff == 1:
            severity_score_sum += 0.5
        # else: 0 points

        # Duplicate detection
        if correct_label == "label_duplicate":
            if agent_label == "label_duplicate":
                correct_dup_ref = ground_truth.get("_duplicate_of")
                agent_dup_ref = action.get("duplicate_of")
                if agent_dup_ref and agent_dup_ref == correct_dup_ref:
                    dup_fully_correct += 1

        # P0 miss penalty
        if correct_label == "label_bug" and correct_severity == "P0":
            if agent_label != "label_bug":
                p0_penalty += 0.15

    label_score = label_correct / total
    severity_score = severity_score_sum / total
    dup_score = dup_fully_correct / max(total_duplicates, 1)

    raw = (
        0.50 * label_score
        + 0.30 * severity_score
        + 0.20 * dup_score
        - p0_penalty
    )
    return _safe_score(raw)


# ---------------------------------------------------------------------------
# Task registry — used by app.py /tasks endpoint and inference.py
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict] = {
    "easy": {
        "name": "Basic Label Classification",
        "difficulty": "easy",
        "description": (
            "Triage 8 clearly defined issues. One of each label type. "
            "No duplicates. Stack traces present in all bugs. "
            "Expected frontier model score: ~0.80"
        ),
        "grader": grade_easy,
        "inbox_size": 8,
    },
    "medium": {
        "name": "Triage with Duplicate Detection",
        "difficulty": "medium",
        "description": (
            "15 issues including 2 hidden duplicate pairs. "
            "Duplicates are paraphrased versions of earlier issues. "
            "Graded: 70% label accuracy + 30% duplicate detection. "
            "Expected frontier model score: ~0.55"
        ),
        "grader": grade_medium,
        "inbox_size": 15,
    },
    "hard": {
        "name": "Full Severity Triage with Subtle Duplicates",
        "difficulty": "hard",
        "description": (
            "20 issues with subtle wording, ambiguous labels, "
            "and subtle duplicate pairs. Severity is scored. "
            "Missing a P0 critical bug incurs a -0.15 penalty per miss. "
            "Graded: 50% label + 30% severity + 20% duplicate detection. "
            "Expected frontier model score: ~0.30"
        ),
        "grader": grade_hard,
        "inbox_size": 20,
    },
}


# ---------------------------------------------------------------------------
# Smoke test (run directly: python graders.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from issue_generator import generate_inbox

    print("=== Grader smoke test ===\n")

    for task_id, info in TASK_REGISTRY.items():
        grader = info["grader"]
        inbox = generate_inbox(task_id)

        # Build perfect actions (all correct)
        perfect = [
            {
                "issue_id": issue["issue_id"],
                "action": issue["_correct_label"],
                "severity": issue["_correct_severity"],
                "duplicate_of": issue.get("_duplicate_of"),
            }
            for issue in inbox
        ]

        # Build random wrong actions
        import random
        wrong_labels = ["label_bug", "label_feature", "label_invalid"]
        wrong_sevs = ["P0", "P1", "P2", "P3"]
        wrong = [
            {
                "issue_id": issue["issue_id"],
                "action": random.choice(wrong_labels),
                "severity": random.choice(wrong_sevs),
                "duplicate_of": None,
            }
            for issue in inbox
        ]

        # Build mixed actions (alternate correct/wrong)
        mixed = []
        for i, issue in enumerate(inbox):
            if i % 2 == 0:
                mixed.append({
                    "issue_id": issue["issue_id"],
                    "action": issue["_correct_label"],
                    "severity": issue["_correct_severity"],
                    "duplicate_of": issue.get("_duplicate_of"),
                })
            else:
                mixed.append({
                    "issue_id": issue["issue_id"],
                    "action": "label_invalid",
                    "severity": "P3",
                    "duplicate_of": None,
                })

        perfect_score = grader(perfect)
        wrong_score = grader(wrong)
        mixed_score = grader(mixed)
        empty_score = grader([])

        print(f"Task: {task_id} ({info['inbox_size']} issues)")
        print(f"  Perfect actions : {perfect_score:.3f} (expected ~1.0)")
        print(f"  Mixed actions   : {mixed_score:.3f} (expected ~0.3–0.6)")
        print(f"  All wrong       : {wrong_score:.3f} (expected ~0.0–0.2)")
        print(f"  Empty           : {empty_score:.3f} (expected 0.0)")
        print(f"  Scores vary     : {len({perfect_score, mixed_score, wrong_score, empty_score}) > 2}")
        print()

    print("All grader checks complete.")
