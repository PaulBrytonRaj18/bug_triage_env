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
#   API_BASE_URL=https://api.openai.com/v1 \
#   MODEL_NAME=gpt-4o-mini \
#   HF_TOKEN=your_key_here \
#   python inference.py
#
# The judges will set API_BASE_URL, MODEL_NAME, HF_TOKEN in their
# environment before running this file. ENV_URL defaults to the
# HuggingFace Space URL set in environment config.

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — all required by hackathon spec
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")
ENV_URL:      str = os.environ.get("ENV_URL",       "http://localhost:7860").rstrip("/")

# ---------------------------------------------------------------------------
# OpenAI-compatible client (required by hackathon spec)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",  # some endpoints don't need a key
)

# ---------------------------------------------------------------------------
# System prompt — gives the LLM everything it needs to triage correctly
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Prompt builder — formats the current observation for the LLM
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict[str, Any], seen_issues: list) -> str:
    """
    Build the user-turn prompt from the current observation.
    Includes a summary of previously seen issues to help with duplicate detection.
    """
    seen_summary = ""
    if seen_issues:
        lines = ["Issues already triaged this session (for duplicate detection):"]
        for s in seen_issues[-10:]:  # last 10 to stay within context
            dup_note = f" [you labeled as duplicate of {s['duplicate_of']}]" if s.get("duplicate_of") else ""
            lines.append(f"  {s['issue_id']}: [{s['action_type']}] {s['title'][:60]}{dup_note}")
        seen_summary = "\n".join(lines) + "\n\n"

    has_trace = "YES — strong signal this is a bug" if obs.get("has_stack_trace") else "No"
    components = ", ".join(obs.get("mentioned_components", [])) or "none mentioned"

    return f"""{seen_summary}Current issue to triage:

Issue ID:    {obs['issue_id']}
Title:       {obs['title']}
Reporter:    {obs['reporter']}
Filed on:    {obs['created_at']}
Comments:    {obs['comments_count']}
Stack trace: {has_trace}
Components:  {components}
Issues remaining after this one: {obs.get('issues_remaining', '?')}

Body:
{obs['body']}

Previous action feedback: {obs.get('last_action_result', 'none')}

Return your triage decision as a JSON object now."""


# ---------------------------------------------------------------------------
# LLM call with retry and fallback
# ---------------------------------------------------------------------------

def call_llm(
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
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=300,
                temperature=0.0,  # deterministic — same issue always same answer
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if the model added them anyway
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            # Validate required fields exist
            required = {"action_type", "severity", "issue_id"}
            if not required.issubset(parsed.keys()):
                raise ValueError(f"Missing required fields. Got: {list(parsed.keys())}")

            return parsed

        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}] JSON parse error: {e}. Raw: {raw[:80]!r}")
        except ValueError as e:
            print(f"    [attempt {attempt}] Validation error: {e}")
        except Exception as e:
            print(f"    [attempt {attempt}] API error: {type(e).__name__}: {e}")

        if attempt < retries:
            time.sleep(retry_delay)

    return None


# ---------------------------------------------------------------------------
# Fallback action — used when LLM fails completely
# ---------------------------------------------------------------------------

def fallback_action(issue_id: str) -> Dict[str, Any]:
    """
    Safe fallback when LLM fails. Labels as bug P2 — never triggers P0 penalty
    but also never scores full marks. Better than crashing.
    """
    return {
        "action_type":  "label_bug",
        "severity":     "P2",
        "issue_id":     issue_id,
        "duplicate_of": None,
        "reasoning":    "Fallback decision — LLM call failed.",
    }


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Optional[Dict[str, Any]]:
    """Call POST /reset and return the observation dict."""
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] /reset failed: {e}")
        return None


def env_step(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call POST /step and return the observation dict."""
    try:
        resp = requests.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] /step failed: {e}")
        return None


def env_health() -> bool:
    """Check if the environment server is reachable."""
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    """
    Run one full episode for the given task_id.

    Returns the normalized final score as a float in [0.0, 1.0].
    Returns 0.0 on environment connection failure.
    """
    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset environment
    obs = env_reset(task_id)
    if obs is None:
        print(f"  [FAIL] Could not reset environment for task '{task_id}'")
        return 0.0

    seen_issues: list = []  # track triaged issues for duplicate detection
    step_num:    int   = 0
    max_steps:   int   = 25  # hard ceiling — no task has more than 20 issues

    while not obs.get("done", False) and step_num < max_steps:
        step_num += 1
        issue_id = obs["issue_id"]

        print(f"  Step {step_num:02d} | {issue_id} | {obs['title'][:50]!r}")

        # Build prompt and call LLM
        user_prompt = build_user_prompt(obs, seen_issues)
        action      = call_llm(user_prompt)

        if action is None:
            print(f"    [WARN] LLM failed — using fallback action")
            action = fallback_action(issue_id)

        # Always ensure issue_id matches — the server rejects mismatches
        action["issue_id"] = issue_id

        # Record for duplicate detection context
        seen_issues.append({
            "issue_id":    issue_id,
            "title":       obs["title"],
            "action_type": action["action_type"],
            "duplicate_of": action.get("duplicate_of"),
        })

        print(
            f"    -> {action['action_type']:18s} | {action['severity']} "
            f"| dup_of={action.get('duplicate_of')}"
        )

        # Submit action
        obs = env_step(action)
        if obs is None:
            print(f"    [FAIL] /step call failed at step {step_num}")
            break

        feedback = obs.get("last_action_result", "")
        score    = obs.get("cumulative_score", 0.0)
        print(f"    <- score={score:.4f} | {feedback[:70]}")

        # Small delay to avoid rate limits on shared API endpoints
        time.sleep(0.3)

    # Final score from last observation
    final_cumulative = obs.get("cumulative_score", 0.0) if obs else 0.0

    # Normalize: divide cumulative by number of steps taken
    # (each step max reward = 1.0, so max cumulative = step_num)
    normalized = round(final_cumulative / max(step_num, 1), 4)

    print(f"\n  Final | steps={step_num} | cumulative={final_cumulative:.4f} | normalized={normalized:.4f}")
    return normalized


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nBug Triage Environment — Baseline Inference")
    print(f"  ENV_URL      : {ENV_URL}")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'set' if HF_TOKEN else 'NOT SET — check your environment'}")

    # Health check before starting
    print(f"\nChecking environment server at {ENV_URL} ...")
    if not env_health():
        print(f"[ERROR] Environment server not reachable at {ENV_URL}")
        print("  Make sure the server is running:")
        print("    uvicorn server.app:app --port 7860")
        print("  Or set ENV_URL to your HuggingFace Space URL.")
        sys.exit(1)
    print("  Server is healthy.\n")

    # Run all 3 tasks
    scores: Dict[str, float] = {}
    start_time = time.time()

    for task_id in ["easy", "medium", "hard"]:
        task_start = time.time()
        scores[task_id] = run_task(task_id)
        elapsed = time.time() - task_start
        print(f"  Task '{task_id}' completed in {elapsed:.1f}s")

    total_elapsed = time.time() - start_time

    # Print final results table
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>8}")
    print(f"  {'-'*20}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<10} {score:>8.4f}  {bar}")
    print(f"  {'-'*20}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<10} {avg:>8.4f}")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
