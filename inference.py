# inference.py
#
# =============================================================
# HACKATHON REQUIREMENT — this file MUST:
#   - be named exactly inference.py
#   - live in the ROOT of the project (not inside server/)
#   - use the OpenAI client (not requests directly for LLM calls)
#   - read credentials from environment variables only
#   - run all 3 tasks and print a score for each
#   - complete in under 20 minutes on vcpu=2 memory=8gb
# =============================================================
#
# Run locally:
#   ENV_URL=http://localhost:7860 \
#   API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/models \
#   MODEL_NAME=gemini-1.5-flash \
#   HF_TOKEN=your_gemini_api_key \
#   python inference.py
#
# The judges will set API_BASE_URL, MODEL_NAME, HF_TOKEN in their
# environment before running this file. ENV_URL defaults to the
# HuggingFace Space URL set in environment config.

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
MAX_TOKENS: int = 300

from server.bug_triage_env_environment import BugTriageEnvEnvironment
from models import BugAction

SYSTEM_PROMPT = """You are a senior software engineer triaging incoming GitHub issue reports.

For each issue you receive, you must return a JSON object with your triage decision.
Return ONLY the JSON object — no preamble, no explanation, no markdown fences.

Your decision must follow this exact schema:
{
  "action_type": "<label>",
  "severity":    "<priority>",
  "issue_id":    "<exact issue_id from the observation>",
  "duplicate_of": "<original issue_id if duplicate, otherwise null>",
  "reasoning":   "<one sentence explaining your decision>"
}

action_type must be exactly one of:
  label_bug       — a confirmed software defect causing incorrect behavior
  label_feature   — a request for new functionality that does not exist yet
  label_duplicate — same underlying problem as an issue already seen this session
  label_invalid   — spam, test, vague complaint with no actionable detail
  label_question  — user asking how to do something, not reporting a bug

severity must be exactly one of:
  P0 — Critical: system down, all users affected, data loss, or security breach
  P1 — High: major feature broken for many users, no workaround
  P2 — Medium: feature broken but workaround exists, or minor data issue
  P3 — Low: cosmetic issue, typo, docs, or low-impact inconvenience

Rules for duplicate detection:
  - If an issue describes the SAME underlying problem as an earlier issue in this session,
    label it label_duplicate and set duplicate_of to the issue_id of the ORIGINAL issue.
  - Different wording, different reporter, or partial overlap does not make it a duplicate
    unless the root cause is clearly the same.
  - If in doubt, label it as a bug/feature/question rather than a duplicate.

Rules for severity:
  - P0: login broken for ALL users, payment failing, data deleted, security hole
  - P1: important feature broken for many users, no workaround
  - P2: workaround exists OR only some users affected
  - P3: single user, cosmetic, docs, minor inconvenience

Strong signals:
  - Stack trace present = almost certainly label_bug
  - "would love", "please add", "could you" = label_feature
  - "how do I", "what is the" = label_question
  - One-word body, gibberish, or "test" = label_invalid"""


def log_start(task: str, env: str, model: str) -> None:
    """Print [START] line. One per episode, at the very beginning."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """
    Print [STEP] line. One per step, immediately after /step returns.

    Rules:
      - reward formatted to 2 decimal places
      - done is lowercase: true or false
      - error is raw string or null
      - action is a compact JSON string on single line
    """
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
    """
    Print [END] line. Always emitted, even on exception.

    Rules:
      - success is lowercase: true or false
      - score formatted to 3 decimal places
      - rewards formatted to 2 decimal places, comma-separated
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs: Dict[str, Any], seen_issues: list) -> str:
    """
    Build the user-turn prompt from the current observation.
    Includes a summary of previously seen issues to help with duplicate detection.
    """
    seen_summary = ""
    if seen_issues:
        lines = ["Issues already triaged this session (for duplicate detection):"]
        for s in seen_issues[-10:]:
            dup_note = (
                f" [you labeled as duplicate of {s['duplicate_of']}]"
                if s.get("duplicate_of")
                else ""
            )
            lines.append(f"  {s['issue_id']}: [{s['action_type']}] {s['title'][:60]}{dup_note}")
        seen_summary = "\n".join(lines) + "\n\n"

    has_trace = "YES — strong signal this is a bug" if obs.get("has_stack_trace") else "No"
    components = ", ".join(obs.get("mentioned_components", [])) or "none mentioned"

    return f"""{seen_summary}Current issue to triage:

Issue ID:    {obs["issue_id"]}
Title:       {obs["title"]}
Reporter:    {obs["reporter"]}
Filed on:    {obs["created_at"]}
Comments:    {obs["comments_count"]}
Stack trace: {has_trace}
Components:  {components}
Issues remaining after this one: {obs.get("issues_remaining", "?")}

Body:
{obs["body"]}

Previous action feedback: {obs.get("last_action_result", "none")}

Return your triage decision as a JSON object now."""


def call_llm(
    client: OpenAI,
    user_prompt: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Call the LLM and parse JSON from the response.

    Returns the parsed action dict or None on failure.
    Retries up to `retries` times on JSON parse failure or API error.
    """
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

            return parsed

        except json.JSONDecodeError as e:
            print(
                f"    [attempt {attempt}] JSON parse error: {e}. Raw: {raw[:80]!r}",
                flush=True,
            )
        except ValueError as e:
            print(f"    [attempt {attempt}] Validation error: {e}", flush=True)
        except Exception as e:
            print(
                f"    [attempt {attempt}] API error: {type(e).__name__}: {e}",
                flush=True,
            )

        if attempt < retries:
            time.sleep(retry_delay)

    return None


def fallback_action(issue_id: str) -> Dict[str, Any]:
    """
    Safe fallback when LLM fails. Labels as bug P2 — never triggers P0 penalty
    but also never scores full marks. Better than crashing.
    """
    return {
        "action_type": "label_bug",
        "severity": "P2",
        "issue_id": issue_id,
        "duplicate_of": None,
        "reasoning": "Fallback decision — LLM call failed.",
    }


async def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one full episode for the given task_id.

    Returns the normalized final score as a float in [0.0, 1.0].
    Returns 0.0 on environment connection failure.
    """
    TASK_INBOX_SIZES = {"easy": 8, "medium": 15, "hard": 20}
    inbox_size = TASK_INBOX_SIZES.get(task_id, 8)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = BugTriageEnvEnvironment()

    try:
        obs = env.reset(task_id=task_id)
    except Exception as e:
        print(f"[ERROR] env.reset() failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    if isinstance(obs, dict):
        obs_dict = obs
    else:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)

    seen_issues: list = []
    step_num: int = 0
    step_rewards: List[float] = []
    error: Optional[str] = None
    current_action: Optional[Dict[str, Any]] = None

    try:
        done = obs_dict.get("done", False)
        while not done and step_num < MAX_STEPS:
            step_num += 1
            issue_id = obs_dict["issue_id"]

            user_prompt = build_user_prompt(obs_dict, seen_issues)
            current_action = call_llm(client, user_prompt)

            if current_action is None:
                error = "LLM call failed after 3 retries"
                current_action = fallback_action(issue_id)
            else:
                error = None

            current_action["issue_id"] = issue_id

            seen_issues.append(
                {
                    "issue_id": issue_id,
                    "title": obs_dict["title"],
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
                next_obs = env.step(action)
            except Exception as e:
                print(f"[ERROR] env.step() failed at step {step_num}: {e}", flush=True)
                step_rewards.append(0.0)
                log_step(step_num, current_action, 0.0, True, f"step failed: {e}")
                break

            if isinstance(next_obs, dict):
                obs_dict = next_obs
            else:
                obs_dict = (
                    next_obs.model_dump() if hasattr(next_obs, "model_dump") else vars(next_obs)
                )

            step_reward = obs_dict.get("last_action_result", "0.0")
            if isinstance(step_reward, str):
                try:
                    step_reward = float(step_reward.split("step_reward:")[-1].strip().split()[0])
                except:
                    step_reward = 0.0

            step_rewards.append(step_reward)

            done = obs_dict.get("done", False)
            log_step(step_num, current_action, step_reward, done, error)

            time.sleep(0.3)

    except Exception as e:
        error = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"[ERROR] {error}", flush=True)
        if current_action:
            log_step(step_num, current_action, 0.0, True, error)

    final_cumulative = obs_dict.get("cumulative_score", 0.0)
    normalized = round(final_cumulative / inbox_size, 3)
    normalized = max(0.0, min(1.0, normalized))

    success = normalized > 0.0 or step_num > 0
    log_end(success=success, steps=step_num, score=normalized, rewards=step_rewards)

    return normalized


async def main() -> None:
    print(f"Bug Triage Environment — Inference Script", flush=True)
    print(f"  API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"  MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(
        f"  HF_TOKEN     : {'set' if HF_TOKEN else 'NOT SET — check your environment'}",
        flush=True,
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
