# inference.py

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

API_BASE_URL: str = os.environ.get(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models",
)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
TASK_NAME: str = os.environ.get("BUG_TRIAGE_TASK", "bug-triage")
BENCHMARK: str = os.environ.get("BUG_TRIAGE_BENCHMARK", "bug-triage")
MAX_STEPS: int = 25
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 500

from server.bug_triage_env_environment import BugTriageEnvEnvironment
from models import BugAction, BugObservation, BugStepResult

SYSTEM_PROMPT = """You are an expert software engineer specializing in bug triage and issue classification.

Analyze each GitHub issue and return ONLY a JSON object with your triage decision.

SCHEMA:
{
  "action_type": "<label>",
  "severity": "<priority>",
  "issue_id": "<exact issue_id>",
  "duplicate_of": "<original issue_id or null>",
  "reasoning": "<one sentence>"
}

LABELS (choose exactly one):
- label_bug: Software defect causing incorrect behavior, crashes, or errors
- label_feature: Request for new functionality that does not exist
- label_duplicate: Same underlying problem as an earlier issue in this session
- label_invalid: Spam, test, gibberish, or vague complaint with no detail
- label_question: User asking how to do something

SEVERITY (choose exactly one):
- P0: System down, all users affected, data loss, security breach, login broken
- P1: Major feature broken for many users, no workaround available
- P2: Feature broken with workaround, or minor issue affecting some users
- P3: Cosmetic, typo, documentation, low-impact single-user issue

DUPLICATE DETECTION:
- Only mark as duplicate if the ROOT CAUSE is clearly the same
- Different wording/reporter/partial overlap does NOT make it a duplicate
- When in doubt, use the original label (bug/feature/question) instead

CRITICAL RULES:
- Stack trace present = label_bug (almost certainly)
- "would love", "please add", "could you implement" = label_feature
- "how do I", "what is the best way to" = label_question
- One-word body, "test", gibberish = label_invalid
- Always use the EXACT issue_id from the observation"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs: BugObservation, seen_issues: list) -> str:
    seen_summary = ""
    if seen_issues:
        lines = ["Previously triaged issues:"]
        for s in seen_issues[-10:]:
            dup_note = f" [duplicate of {s['duplicate_of']}]" if s.get("duplicate_of") else ""
            lines.append(f"  {s['issue_id']}: [{s['action_type']}] {s['title'][:60]}{dup_note}")
        seen_summary = "\n".join(lines) + "\n\n"

    has_trace = "YES - strong signal for label_bug" if obs.has_stack_trace else "No"

    return f"""{seen_summary}ISSUE TO TRIAGE:

Issue ID:    {obs.issue_id}
Title:       {obs.title}
Reporter:    {obs.reporter}
Filed:       {obs.created_at}
Comments:    {obs.comments_count}
Stack trace: {has_trace}
Components:  {", ".join(obs.mentioned_components) or "none"}
Remaining:   {obs.issues_remaining}

Body:
{obs.body}

Previous feedback: {obs.last_action_result}

Return JSON now."""


def extract_observation(result) -> BugObservation:
    """Extract BugObservation from various return types."""
    if isinstance(result, BugStepResult):
        return result.observation
    elif isinstance(result, BugObservation):
        return result
    elif isinstance(result, dict):
        if "observation" in result:
            return result["observation"]
        return BugObservation(**result)
    else:
        raise ValueError(f"Unknown result type: {type(result)}")


def extract_reward(result) -> float:
    """Extract reward from various return types."""
    if isinstance(result, BugStepResult):
        return result.reward
    elif isinstance(result, dict):
        return result.get("reward", 0.0)
    return 0.0


def call_llm(
    client: OpenAI,
    user_prompt: str,
    issue_id: str,
    retries: int = 3,
    retry_delay: float = 1.5,
) -> Optional[Dict[str, Any]]:
    for attempt in range(1, retries + 1):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()

            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

            parsed = json.loads(raw)

            required = {"action_type", "severity", "issue_id"}
            if not required.issubset(parsed.keys()):
                raise ValueError(f"Missing required fields. Got: {list(parsed.keys())}")

            if parsed["action_type"] not in {
                "label_bug",
                "label_feature",
                "label_duplicate",
                "label_invalid",
                "label_question",
            }:
                raise ValueError(f"Invalid action_type: {parsed['action_type']}")

            if parsed["severity"] not in {"P0", "P1", "P2", "P3"}:
                raise ValueError(f"Invalid severity: {parsed['severity']}")

            return parsed

        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}] JSON parse error: {e}. Raw: {raw[:80]!r}", flush=True)
        except ValueError as e:
            print(f"    [attempt {attempt}] Validation error: {e}", flush=True)
        except Exception as e:
            print(f"    [attempt {attempt}] API error: {type(e).__name__}: {e}", flush=True)

        if attempt < retries:
            time.sleep(retry_delay)

    return None


def fallback_action(issue_id: str, obs: BugObservation) -> Dict[str, Any]:
    """Smart fallback based on issue content."""
    action_type = "label_bug"
    severity = "P2"

    if obs.has_stack_trace:
        action_type = "label_bug"
        severity = "P1"

    title_lower = obs.title.lower()
    body_lower = obs.body.lower()

    if any(q in body_lower for q in ["how do i", "how can i", "what is", "how to"]):
        action_type = "label_question"
    elif any(
        f in body_lower
        for f in ["would love", "please add", "could you", "feature request", "implement"]
    ):
        action_type = "label_feature"
    elif len(obs.body.strip()) < 10 or obs.body.lower() in ["test", "bug", "error"]:
        action_type = "label_invalid"

    return {
        "action_type": action_type,
        "severity": severity,
        "issue_id": issue_id,
        "duplicate_of": None,
        "reasoning": "Fallback triage decision",
    }


async def run_task(client: OpenAI, task_id: str) -> float:
    TASK_INBOX_SIZES = {"easy": 8, "medium": 15, "hard": 20}
    inbox_size = TASK_INBOX_SIZES.get(task_id, 8)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = BugTriageEnvEnvironment()

    try:
        result = env.reset(task_id=task_id)
    except Exception as e:
        print(f"[ERROR] env.reset() failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    obs = extract_observation(result)
    seen_issues: list = []
    step_num: int = 0
    step_rewards: List[float] = []
    error: Optional[str] = None
    current_action: Optional[Dict[str, Any]] = None

    try:
        while not obs.done and step_num < MAX_STEPS:
            step_num += 1
            issue_id = obs.issue_id

            user_prompt = build_user_prompt(obs, seen_issues)
            parsed = call_llm(client, user_prompt, issue_id)

            if parsed is None:
                error = "LLM call failed after 3 retries"
                current_action = fallback_action(issue_id, obs)
            else:
                error = None
                current_action = parsed

            current_action["issue_id"] = issue_id

            seen_issues.append(
                {
                    "issue_id": issue_id,
                    "title": obs.title,
                    "action_type": current_action["action_type"],
                    "duplicate_of": current_action.get("duplicate_of"),
                }
            )

            try:
                action = BugAction(
                    action_type=current_action["action_type"],
                    severity=current_action["severity"],
                    issue_id=current_action["issue_id"],
                    duplicate_of=current_action.get("duplicate_of"),
                    reasoning=current_action.get("reasoning"),
                )
                step_result = env.step(action)
            except Exception as e:
                print(f"[ERROR] env.step() failed at step {step_num}: {e}", flush=True)
                step_rewards.append(0.0)
                log_step(step_num, current_action, 0.0, True, f"step failed: {e}")
                break

            step_reward = extract_reward(step_result)
            obs = extract_observation(step_result)
            step_rewards.append(step_reward)

            log_step(step_num, current_action, step_reward, obs.done, error)
            time.sleep(0.2)

    except Exception as e:
        error = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"[ERROR] {error}", flush=True)
        if current_action:
            log_step(step_num, current_action, 0.0, True, error)

    final_cumulative = obs.cumulative_score
    normalized = round(final_cumulative / inbox_size, 3)
    normalized = max(0.0, min(1.0, normalized))

    success = normalized > 0.0 or step_num > 0
    log_end(success=success, steps=step_num, score=normalized, rewards=step_rewards)

    return normalized


async def main() -> None:
    print(f"Bug Triage Environment - Inference Script", flush=True)
    print(f"  API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(
        f"  HF_TOKEN     : {'set' if HF_TOKEN else 'NOT SET - check your environment'}", flush=True
    )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

    scores: Dict[str, float] = {}
    start_time = time.time()

    for task_id in ["easy", "medium", "hard"]:
        task_start = time.time()
        scores[task_id] = await run_task(client, task_id)
        elapsed = time.time() - task_start
        print(f"  Task '{task_id}' completed in {elapsed:.1f}s", flush=True)

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 60}", flush=True)
    print("  BASELINE RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  {'Task':<10} {'Score':>8}", flush=True)
    print(f"  {'-' * 20}", flush=True)
    for task_id, score in scores.items():
        bar = "=" * int(score * 20)
        print(f"  {task_id:<10} {score:>8.3f}  {bar}", flush=True)
    print(f"  {'-' * 20}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<10} {avg:>8.3f}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total time: {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
