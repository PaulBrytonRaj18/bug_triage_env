# bug_triage_env/server/tests/test_app.py
"""
Tests for the FastAPI application (app.py).
"""

import pytest
from fastapi.testclient import TestClient

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        """Health endpoint should return status 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_version(self):
        """Health endpoint should return version info."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_returns_tasks(self):
        """Health endpoint should list available tasks."""
        response = client.get("/health")
        data = response.json()
        assert "tasks_available" in data
        assert set(data["tasks_available"]) == {"easy", "medium", "hard"}


class TestTasksEndpoint:
    """Tests for /tasks endpoint."""

    def test_tasks_returns_200(self):
        """Tasks endpoint should return 200 status."""
        response = client.get("/tasks")
        assert response.status_code == 200

    def test_tasks_returns_three_tasks(self):
        """Tasks endpoint should return exactly 3 tasks."""
        response = client.get("/tasks")
        data = response.json()
        assert len(data["tasks"]) == 3

    def test_tasks_have_required_fields(self):
        """Each task should have id, name, difficulty, inbox_size, description."""
        response = client.get("/tasks")
        data = response.json()
        required_fields = {"id", "name", "difficulty", "inbox_size", "description"}
        for task in data["tasks"]:
            assert required_fields.issubset(task.keys())

    def test_tasks_have_correct_difficulties(self):
        """Tasks should have correct difficulty levels."""
        response = client.get("/tasks")
        data = response.json()
        difficulties = {task["difficulty"] for task in data["tasks"]}
        assert difficulties == {"easy", "medium", "hard"}

    def test_tasks_have_correct_ids(self):
        """Tasks should have correct IDs."""
        response = client.get("/tasks")
        data = response.json()
        ids = {task["id"] for task in data["tasks"]}
        assert ids == {"easy", "medium", "hard"}


class TestResetEndpoint:
    """Tests for /reset endpoint."""

    def test_reset_default_task(self):
        """Reset with no body should default to 'easy' task."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "issue_id" in data

    def test_reset_easy_task(self):
        """Reset with easy task should return 8 issues in inbox."""
        response = client.post("/reset", json={"task_id": "easy"})
        assert response.status_code == 200
        data = response.json()
        assert data["issues_remaining"] == 7
        assert data["done"] is False

    def test_reset_medium_task(self):
        """Reset with medium task should return 15 issues."""
        response = client.post("/reset", json={"task_id": "medium"})
        assert response.status_code == 200
        data = response.json()
        assert data["issues_remaining"] == 14
        assert data["done"] is False

    def test_reset_hard_task(self):
        """Reset with hard task should return 20 issues."""
        response = client.post("/reset", json={"task_id": "hard"})
        assert response.status_code == 200
        data = response.json()
        assert data["issues_remaining"] == 19
        assert data["done"] is False

    def test_reset_invalid_task_defaults_to_easy(self):
        """Reset with invalid task should default to easy."""
        response = client.post("/reset", json={"task_id": "invalid"})
        assert response.status_code == 200
        data = response.json()
        assert "issue_id" in data

    def test_reset_returns_observation_fields(self):
        """Reset should return all required observation fields."""
        response = client.post("/reset", json={"task_id": "easy"})
        data = response.json()
        required_fields = {
            "issue_id",
            "title",
            "body",
            "reporter",
            "created_at",
            "comments_count",
            "has_stack_trace",
            "mentioned_components",
            "issues_remaining",
            "last_action_result",
            "cumulative_score",
            "done",
        }
        assert required_fields.issubset(data.keys())

    def test_reset_cumulative_score_starts_at_zero(self):
        """Reset should set cumulative_score to 0."""
        response = client.post("/reset", json={"task_id": "easy"})
        data = response.json()
        assert data["cumulative_score"] == 0.0


class TestStepEndpoint:
    """Tests for /step endpoint."""

    def test_step_returns_observation_reward_done_info(self):
        """Step should return observation, reward, done, and info."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": "ISS-1000"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_for_correct_action(self):
        """Step should give positive reward for correct label."""
        reset_resp = client.post("/reset", json={"task_id": "easy"})
        issue_id = reset_resp.json()["issue_id"]

        response = client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": issue_id}
        )
        data = response.json()
        assert data["reward"] > 0

    def test_step_penalty_for_invalid_action_type(self):
        """Step should penalize invalid action_type."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.post(
            "/step",
            json={"action_type": "invalid_action", "severity": "P0", "issue_id": "ISS-1000"},
        )
        data = response.json()
        assert data["reward"] == 0.0

    def test_step_penalty_for_invalid_severity(self):
        """Step should penalize invalid severity."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.post(
            "/step", json={"action_type": "label_bug", "severity": "P5", "issue_id": "ISS-1000"}
        )
        data = response.json()
        assert data["reward"] == 0.0

    def test_step_penalty_for_wrong_issue_id(self):
        """Step should penalize wrong issue_id."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": "WRONG-ID"}
        )
        data = response.json()
        assert data["reward"] == 0.0

    def test_step_advances_to_next_issue(self):
        """Step should advance to the next issue in the inbox."""
        reset_resp = client.post("/reset", json={"task_id": "easy"})
        first_issue_id = reset_resp.json()["issue_id"]

        client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": first_issue_id}
        )

        state_resp = client.get("/state")
        state = state_resp.json()
        assert state["step_count"] == 1

    def test_step_episode_completes_after_all_issues(self):
        """Episode should complete after all issues are triaged."""
        client.post("/reset", json={"task_id": "easy"})

        for _ in range(8):
            state_resp = client.get("/state")
            issue_id = state_resp.json().get("issue_id", "ISS-1000")

            resp = client.post(
                "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": issue_id}
            )
            if resp.json().get("done"):
                break

    def test_step_info_contains_last_action_result(self):
        """Step info should contain last_action_result."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": "ISS-1000"}
        )
        data = response.json()
        assert "last_action_result" in data["info"]


class TestStateEndpoint:
    """Tests for /state endpoint."""

    def test_state_returns_200(self):
        """State endpoint should return 200."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.get("/state")
        assert response.status_code == 200

    def test_state_returns_episode_metadata(self):
        """State should return episode_id, step_count, task_id."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.get("/state")
        data = response.json()
        assert "episode_id" in data
        assert "step_count" in data
        assert "task_id" in data

    def test_state_step_count_starts_at_zero(self):
        """Step count should start at 0 after reset."""
        client.post("/reset", json={"task_id": "easy"})
        response = client.get("/state")
        data = response.json()
        assert data["step_count"] == 0

    def test_state_increments_after_step(self):
        """Step count should increment after each step."""
        client.post("/reset", json={"task_id": "easy"})
        response1 = client.get("/state")
        step_count_before = response1.json()["step_count"]

        reset_resp = client.post("/reset", json={"task_id": "easy"})
        issue_id = reset_resp.json()["issue_id"]
        client.post(
            "/step", json={"action_type": "label_bug", "severity": "P0", "issue_id": issue_id}
        )

        response2 = client.get("/state")
        step_count_after = response2.json()["step_count"]
        assert step_count_after == step_count_before + 1


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_redirects_to_docs(self):
        """Root should redirect to /docs."""
        response = client.get("/")
        assert response.status_code == 200
