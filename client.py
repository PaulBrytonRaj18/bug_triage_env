# bug_triage_env/client.py
#
# HTTP client for the Bug Triage Environment.
# Agents use this to connect to a running instance of the environment
# (locally via Docker or remotely via HuggingFace Spaces).
#
# Phase 1: Basic HTTP client using requests.
# Phase 3: Will be upgraded to use openenv-core's HTTPEnvClient base class.
#
# Usage:
#   from bug_triage_env.client import BugTriageClient
#   client = BugTriageClient("http://localhost:7860")
#   obs = client.reset("easy")
#   result = client.step({...})
#   state = client.state()

from __future__ import annotations
import requests
from typing import Dict, Any


class BugTriageClient:
    """
    Minimal HTTP client for interacting with the Bug Triage environment.
    Connects to a running FastAPI server at base_url.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        """Reset the environment and return the first observation."""
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a triage action and return the next observation.

        action must contain:
          action_type, severity, issue_id
        optionally:
          duplicate_of, reasoning
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Return the current episode state (episode_id, step_count)."""
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Quick health check. Returns True if server is responding."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
