"""
Pseudo-test for inference.py - tests scoring logic without API key.
Mimics the fallback behavior to get baseline scores.

Run with: python3 test_inference_pseudo.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from server.environment import BugTriageEnvironment
from models import BugAction


def run_task_with_fallback(task_id: str) -> dict:
    """Run a task using fallback P2 bug labels to get baseline score."""

    TASK_INBOX_SIZES = {"easy": 8, "medium": 15, "hard": 20}
    inbox_size = TASK_INBOX_SIZES.get(task_id, 8)

    print(f"\n[START] task={task_id} env=bug-triage model=fallback")

    env = BugTriageEnvironment()
    obs = env.reset(task_id=task_id)

    step_num = 0
    step_rewards = []

    while not obs.done and step_num < 25:
        step_num += 1

        fallback_action = BugAction(
            action_type="label_bug",
            severity="P2",
            issue_id=obs.issue_id,
            duplicate_of=None,
            reasoning="Fallback decision — no LLM call",
        )

        result = env.step(fallback_action)
        step_rewards.append(result.reward)

        print(
            f"[STEP] step={step_num} "
            f'action={{"action_type":"label_bug","severity":"P2",'
            f'"issue_id":"{obs.issue_id}","duplicate_of":null}} '
            f"reward={result.reward:.2f} done={str(result.done).lower()}",
            flush=True,
        )

        obs = result.observation

    final_score = round(obs.cumulative_score / inbox_size, 3)
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)

    print(f"[END] success=true steps={step_num} score={final_score:.3f} rewards={rewards_str}")

    return {
        "task": task_id,
        "steps": step_num,
        "score": max(0.0, min(1.0, final_score)),
        "rewards": step_rewards,
        "total": obs.cumulative_score,
    }


def main():
    print("=" * 60)
    print("  PSEUDO-TEST: Fallback Baseline Scores")
    print("  (No API key required - uses P2 bug fallback)")
    print("=" * 60)

    results = {}

    for task in ["easy", "medium", "hard"]:
        result = run_task_with_fallback(task)
        results[task] = result
        print(f"  Task '{task}' completed in 0.0s")

    print(f"\n{'=' * 60}")
    print("  BASELINE RESULTS (Fallback)")
    print(f"{'=' * 60}")
    print(f"  {'Task':<10} {'Score':>8} {'Total R':>10} {'Steps':>6}")
    print(f"  {'-' * 40}")

    for task_id, r in results.items():
        bar = "=" * int(r["score"] * 20)
        print(f"  {task_id:<10} {r['score']:>8.3f} {r['total']:>10.3f} {r['steps']:>6}  {bar}")

    print(f"  {'-' * 40}")
    avg = sum(r["score"] for r in results.values()) / len(results)
    print(f"  {'average':<10} {avg:>8.3f}")
    print(f"{'=' * 60}")
    print(f"  Total time: 0.0s")


if __name__ == "__main__":
    main()
