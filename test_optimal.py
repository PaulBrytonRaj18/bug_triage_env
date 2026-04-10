"""
Test the fixed inference.py logic with mock LLM responses.
This tests the BugStepResult handling and scoring logic.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from server.bug_triage_env_environment import BugTriageEnvEnvironment
from models import BugAction, BugObservation, BugStepResult


def extract_observation(result):
    if isinstance(result, BugStepResult):
        return result.observation
    elif isinstance(result, BugObservation):
        return result
    elif isinstance(result, dict):
        if "observation" in result:
            return result["observation"]
        return BugObservation(**result)
    return result


def extract_reward(result):
    if isinstance(result, BugStepResult):
        return result.reward
    elif isinstance(result, dict):
        return result.get("reward", 0.0)
    return 0.0


def get_ground_truth(env, task_id):
    """Get ground truth from the environment's inbox."""
    from issue_generator import generate_inbox

    inbox = generate_inbox(task_id)
    return inbox


def run_optimal_test(task_id: str) -> dict:
    """Run with optimal decisions based on ground truth."""
    TASK_INBOX_SIZES = {"easy": 8, "medium": 15, "hard": 20}
    inbox_size = TASK_INBOX_SIZES.get(task_id, 8)

    print(f"\n[START] task={task_id} env=bug-triage model=optimal")

    env = BugTriageEnvEnvironment()
    result = env.reset(task_id=task_id)
    obs = extract_observation(result)

    ground_truth = get_ground_truth(env, task_id)
    gt_map = {g["issue_id"]: g for g in ground_truth}

    step_num = 0
    step_rewards = []

    while not obs.done and step_num < 25:
        step_num += 1
        issue_id = obs.issue_id

        gt = gt_map.get(issue_id, {})
        action = BugAction(
            action_type=gt.get("_correct_label", "label_bug"),
            severity=gt.get("_correct_severity", "P2"),
            issue_id=issue_id,
            duplicate_of=gt.get("_duplicate_of"),
            reasoning="Optimal decision based on ground truth",
        )

        step_result = env.step(action)
        step_reward = extract_reward(step_result)
        obs = extract_observation(step_result)
        step_rewards.append(step_reward)

        action_dict = {
            "action_type": action.action_type,
            "severity": action.severity,
            "issue_id": action.issue_id,
            "duplicate_of": action.duplicate_of,
        }
        import json

        print(
            f"[STEP] step={step_num} action={json.dumps(action_dict, separators=(',', ':'))} "
            f"reward={step_reward:.2f} done={str(obs.done).lower()}",
            flush=True,
        )

    final_score = round(obs.cumulative_score / inbox_size, 3)
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    print(f"[END] success=true steps={step_num} score={final_score:.3f} rewards={rewards_str}")

    return {
        "task": task_id,
        "steps": step_num,
        "score": final_score,
        "rewards": step_rewards,
        "total": obs.cumulative_score,
    }


def main():
    print("=" * 60)
    print("  OPTIMAL TEST: Ground Truth Decisions")
    print("  (Shows maximum achievable score)")
    print("=" * 60)

    results = {}
    for task in ["easy", "medium", "hard"]:
        result = run_optimal_test(task)
        results[task] = result

    print(f"\n{'=' * 60}")
    print("  OPTIMAL RESULTS (Ground Truth)")
    print(f"{'=' * 60}")
    print(f"  {'Task':<10} {'Score':>8} {'Total R':>10} {'Steps':>6}")
    print(f"  {'-' * 40}")

    for task_id, r in results.items():
        bar = "=" * int(r["score"] * 20)
        print(f"  {task_id:<10} {r['score']:>8.3f} {r['total']:>10.3f} {r['steps']:>6}  {bar}")

    print(f"  {'-' * 40}")
    avg = sum(r["score"] for r in results.values()) / len(results)
    print(f"  {'average':<10} {avg:>8.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
