# bug_triage_env/server/app.py
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from models import BugAction, BugObservation
from server.environment import BugTriageEnvironment, VALID_TASK_IDS

app = FastAPI(
    title="Bug Triage Environment",
    description="OpenEnv RL environment for GitHub issue triage. 3 difficulty tasks.",
    version="1.0.0",
)

env = BugTriageEnvironment()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="easy | medium | hard")


class StepRequest(BaseModel):
    action_type:  str            = Field(..., description="label_bug|label_feature|label_duplicate|label_invalid|label_question")
    severity:     str            = Field(..., description="P0|P1|P2|P3")
    issue_id:     str            = Field(..., description="Must match current observation's issue_id")
    duplicate_of: Optional[str]  = Field(default=None)
    reasoning:    Optional[str]  = Field(default=None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/health")


@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.0.0",
        "tasks_available": sorted(VALID_TASK_IDS),
    }


@app.get("/tasks", tags=["System"])
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Basic Label Classification",
                "difficulty": "easy",
                "inbox_size": 8,
                "description": "8 issues, one of each type, no duplicates. Expected score: ~0.80",
            },
            {
                "id": "medium",
                "name": "Triage with Duplicate Detection",
                "difficulty": "medium",
                "inbox_size": 15,
                "description": "15 issues, 2 hidden duplicate pairs. Expected score: ~0.55",
            },
            {
                "id": "hard",
                "name": "Full Severity Triage with Subtle Duplicates",
                "difficulty": "hard",
                "inbox_size": 20,
                "description": "20 issues, subtle duplicates, P0 penalty. Expected score: ~0.30",
            },
        ]
    }


@app.post("/reset", response_model=BugObservation, tags=["OpenEnv"])
async def reset(body: ResetRequest) -> BugObservation:
    """Start a new episode. Returns the first issue."""
    try:
        return env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/step", response_model=BugObservation, tags=["OpenEnv"])
async def step(body: StepRequest) -> BugObservation:
    """Submit a triage action. Returns the next issue."""
    action = BugAction(
        action_type  = body.action_type,
        severity     = body.severity,
        issue_id     = body.issue_id,
        duplicate_of = body.duplicate_of,
        reasoning    = body.reasoning,
    )
    return env.step(action)


@app.get("/state", tags=["OpenEnv"])
async def state() -> Dict[str, Any]:
    """Return current episode metadata."""
    return env.state
