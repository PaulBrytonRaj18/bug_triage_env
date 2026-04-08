# bug_triage_env/tests/test_inference.py
"""
Tests for inference.py - the baseline inference script.
"""

import pytest
import asyncio
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    log_start,
    log_step,
    log_end,
    build_user_prompt,
    call_llm,
    fallback_action,
    run_task,
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_STEPS,
)


class TestLoggingFunctions:
    """Tests for logging functions."""

    def test_log_start_format(self, capsys):
        """log_start should print in correct format."""
        log_start(task="easy", env="bug-triage", model="gemini-1.5-flash")
        captured = capsys.readouterr()
        assert "[START]" in captured.out
        assert "task=easy" in captured.out
        assert "env=bug-triage" in captured.out
        assert "model=gemini-1.5-flash" in captured.out

    def test_log_step_format_with_reward(self, capsys):
        """log_step should print reward with 2 decimal places."""
        action = {"action_type": "label_bug", "severity": "P0", "issue_id": "ISS-1000"}
        log_step(step=1, action=action, reward=0.90, done=False, error=None)
        captured = capsys.readouterr()
        assert "[STEP]" in captured.out
        assert "step=1" in captured.out
        assert "reward=0.90" in captured.out
        assert "done=false" in captured.out
        assert "error=null" in captured.out

    def test_log_step_with_error(self, capsys):
        """log_step should print error when provided."""
        action = {"action_type": "label_bug", "severity": "P0", "issue_id": "ISS-1000"}
        log_step(step=1, action=action, reward=0.0, done=False, error="LLM failed")
        captured = capsys.readouterr()
        assert "error=LLM failed" in captured.out

    def test_log_step_done_true(self, capsys):
        """log_step should print done=true as lowercase."""
        action = {"action_type": "label_bug", "severity": "P0", "issue_id": "ISS-1000"}
        log_step(step=5, action=action, reward=1.0, done=True, error=None)
        captured = capsys.readouterr()
        assert "done=true" in captured.out

    def test_log_end_format(self, capsys):
        """log_end should print in correct format."""
        rewards = [0.5, 0.75, 1.0]
        log_end(success=True, steps=3, score=0.75, rewards=rewards)
        captured = capsys.readouterr()
        assert "[END]" in captured.out
        assert "success=true" in captured.out
        assert "steps=3" in captured.out
        assert "score=0.750" in captured.out
        assert "0.50,0.75,1.00" in captured.out

    def test_log_end_failure(self, capsys):
        """log_end should print success=false on failure."""
        log_end(success=False, steps=0, score=0.0, rewards=[])
        captured = capsys.readouterr()
        assert "success=false" in captured.out


class TestBuildUserPrompt:
    """Tests for build_user_prompt function."""

    def test_prompt_contains_issue_details(self):
        """Prompt should contain issue details from observation."""
        obs = {
            "issue_id": "ISS-1234",
            "title": "Login fails",
            "body": "Cannot login on mobile",
            "reporter": "alice",
            "created_at": "2025-01-01",
            "comments_count": 3,
            "has_stack_trace": True,
            "mentioned_components": ["auth", "api"],
            "issues_remaining": 5,
            "last_action_result": "Correct label +0.6",
        }
        prompt = build_user_prompt(obs, [])
        assert "ISS-1234" in prompt
        assert "Login fails" in prompt
        assert "alice" in prompt
        assert "auth, api" in prompt

    def test_prompt_shows_stack_trace_signal(self):
        """Prompt should indicate when stack trace is present."""
        obs_with_trace = {
            "issue_id": "ISS-1",
            "title": "Bug",
            "body": "Error occurred",
            "reporter": "bob",
            "created_at": "2025-01-01",
            "comments_count": 0,
            "has_stack_trace": True,
            "mentioned_components": [],
            "issues_remaining": 1,
            "last_action_result": "",
        }
        prompt = build_user_prompt(obs_with_trace, [])
        assert "YES" in prompt

    def test_prompt_includes_seen_issues(self):
        """Prompt should include previously seen issues for duplicate detection."""
        obs = {
            "issue_id": "ISS-2",
            "title": "New bug",
            "body": "New issue",
            "reporter": "bob",
            "created_at": "2025-01-01",
            "comments_count": 0,
            "has_stack_trace": False,
            "mentioned_components": [],
            "issues_remaining": 1,
            "last_action_result": "",
        }
        seen_issues = [
            {
                "issue_id": "ISS-1",
                "title": "Login bug",
                "action_type": "label_bug",
                "duplicate_of": None,
            }
        ]
        prompt = build_user_prompt(obs, seen_issues)
        assert "ISS-1" in prompt
        assert "Login bug" in prompt

    def test_prompt_shows_duplicate_reference(self):
        """Prompt should show duplicate reference when issue was marked as duplicate."""
        obs = {
            "issue_id": "ISS-3",
            "title": "Dup bug",
            "body": "Same as before",
            "reporter": "carol",
            "created_at": "2025-01-01",
            "comments_count": 0,
            "has_stack_trace": False,
            "mentioned_components": [],
            "issues_remaining": 1,
            "last_action_result": "",
        }
        seen_issues = [
            {
                "issue_id": "ISS-1",
                "title": "Original",
                "action_type": "label_duplicate",
                "duplicate_of": "ISS-1",
            }
        ]
        prompt = build_user_prompt(obs, seen_issues)
        assert "ISS-1" in prompt
        assert "duplicate of ISS-1" in prompt


class TestFallbackAction:
    """Tests for fallback_action function."""

    def test_fallback_returns_safe_action(self):
        """Fallback should return a safe default action."""
        result = fallback_action("ISS-1234")
        assert result["issue_id"] == "ISS-1234"
        assert result["action_type"] == "label_bug"
        assert result["severity"] == "P2"
        assert result["duplicate_of"] is None

    def test_fallback_never_uses_p0(self):
        """Fallback should never use P0 to avoid penalty."""
        result = fallback_action("ISS-9999")
        assert result["severity"] != "P0"


class TestCallLlm:
    """Tests for call_llm function."""

    def test_call_llm_parses_valid_json_response(self):
        """call_llm should parse valid JSON from LLM response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"action_type": "label_bug", "severity": "P1", "issue_id": "ISS-1000"}
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        result = call_llm(mock_client, "Test prompt")

        assert result is not None
        assert result["action_type"] == "label_bug"
        assert result["severity"] == "P1"
        assert result["issue_id"] == "ISS-1000"

    def test_call_llm_strips_markdown_fences(self):
        """call_llm should strip markdown code fences from response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = (
            '```json\n{"action_type": "label_feature", "severity": "P3", "issue_id": "ISS-2"}\n```'
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        result = call_llm(mock_client, "Test prompt")

        assert result is not None
        assert result["action_type"] == "label_feature"

    def test_call_llm_retries_on_parse_error(self):
        """call_llm should retry when JSON parsing fails."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "not valid json"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        result = call_llm(mock_client, "Test prompt", retries=2, retry_delay=0.1)

        assert result is None
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_validates_required_fields(self):
        """call_llm should reject responses missing required fields."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({"action_type": "label_bug"})

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        result = call_llm(mock_client, "Test prompt")

        assert result is None


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT content."""

    def test_system_prompt_defines_action_types(self):
        """System prompt should define all action types."""
        assert "label_bug" in SYSTEM_PROMPT
        assert "label_feature" in SYSTEM_PROMPT
        assert "label_duplicate" in SYSTEM_PROMPT
        assert "label_invalid" in SYSTEM_PROMPT
        assert "label_question" in SYSTEM_PROMPT

    def test_system_prompt_defines_severities(self):
        """System prompt should define all severity levels."""
        assert "P0" in SYSTEM_PROMPT
        assert "P1" in SYSTEM_PROMPT
        assert "P2" in SYSTEM_PROMPT
        assert "P3" in SYSTEM_PROMPT

    def test_system_prompt_requires_json_format(self):
        """System prompt should require JSON output."""
        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT

    def test_system_prompt_explains_duplicate_rules(self):
        """System prompt should explain duplicate detection rules."""
        assert "duplicate" in SYSTEM_PROMPT.lower()


class TestConstants:
    """Tests for module constants."""

    def test_max_steps_is_reasonable(self):
        """MAX_STEPS should be reasonable for the task."""
        assert MAX_STEPS >= 8
        assert MAX_STEPS <= 50

    def test_model_name_has_default(self):
        """MODEL_NAME should have a default value."""
        assert MODEL_NAME is not None
        assert len(MODEL_NAME) > 0


class TestIntegration:
    """Integration tests for run_task (mocked LLM)."""

    @pytest.mark.asyncio
    async def test_run_task_easy_completes(self):
        """run_task should complete an easy episode."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "action_type": "label_bug",
                "severity": "P0",
                "issue_id": "ISS-1000",
                "duplicate_of": None,
                "reasoning": "Test",
            }
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        class ResetResult:
            def __init__(self):
                self.issue_id = "ISS-1000"
                self.title = "Test"
                self.body = "Body"
                self.reporter = "test"
                self.created_at = "2025-01-01"
                self.comments_count = 0
                self.has_stack_trace = True
                self.mentioned_components = ["auth"]
                self.issues_remaining = 7
                self.last_action_result = ""
                self.cumulative_score = 0.0
                self.done = False

            def model_dump(self):
                return {
                    "issue_id": self.issue_id,
                    "title": self.title,
                    "body": self.body,
                    "reporter": self.reporter,
                    "created_at": self.created_at,
                    "comments_count": self.comments_count,
                    "has_stack_trace": self.has_stack_trace,
                    "mentioned_components": self.mentioned_components,
                    "issues_remaining": self.issues_remaining,
                    "last_action_result": self.last_action_result,
                    "cumulative_score": self.cumulative_score,
                    "done": self.done,
                }

        class StepResult:
            def __init__(self):
                self.issue_id = "ISS-1001"
                self.title = "Test"
                self.body = "Body"
                self.reporter = "test"
                self.created_at = "2025-01-01"
                self.comments_count = 0
                self.has_stack_trace = False
                self.mentioned_components = []
                self.issues_remaining = 0
                self.last_action_result = "Correct"
                self.cumulative_score = 0.9
                self.done = True

            def model_dump(self):
                return {
                    "issue_id": self.issue_id,
                    "title": self.title,
                    "body": self.body,
                    "reporter": self.reporter,
                    "created_at": self.created_at,
                    "comments_count": self.comments_count,
                    "has_stack_trace": self.has_stack_trace,
                    "mentioned_components": self.mentioned_components,
                    "issues_remaining": self.issues_remaining,
                    "last_action_result": self.last_action_result,
                    "cumulative_score": self.cumulative_score,
                    "done": self.done,
                }

        def mock_reset(**kw):
            return ResetResult()

        def mock_step(**kw):
            return StepResult()

        with patch("inference.BugTriageEnvEnvironment") as MockEnv:
            mock_env = Mock()
            mock_env.reset = mock_reset
            mock_env.step = mock_step
            MockEnv.return_value = mock_env

            score = await run_task(mock_client, "easy")

            assert score >= 0.0
            assert score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
