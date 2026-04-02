# Bug Triage Env

An OpenEnv-compliant reinforcement learning environment that simulates the real-world
software engineering task of triaging GitHub-style issue reports.

Built for the **Meta PyTorch OpenEnv Hackathon (India, April 2026)**.

---

## What it does

An AI agent receives a backlog of incoming issue reports and must triage each one by:

1. **Labeling it** — `label_bug` / `label_feature` / `label_duplicate` / `label_invalid` / `label_question`
2. **Assigning severity** — `P0` critical through `P3` low
3. **Detecting duplicates** — referencing the original issue ID

The environment gives the agent a reward signal after every decision (dense rewards),
not just at the end of the episode.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r server/requirements.txt

# 2. Start the server locally
uvicorn server.app:app --reload --port 7860

# 3. Health check
curl http://localhost:7860/health

# 4. Reset environment (start a new episode)
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy"}'

# 5. Run the baseline agent
cp .env.example .env        # fill in your API credentials
source .env
python inference.py
```

---

## Running with Docker

```bash
# Build
docker build -t bug-triage-env -f server/Dockerfile .

# Run
docker run -p 7860:7860 bug-triage-env

# Test
curl http://localhost:7860/health
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — must return 200 |
| `GET` | `/tasks` | List all 3 tasks with descriptions |
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "easy"}` |
| `POST` | `/step` | Submit triage action. Body: BugAction JSON |
| `GET` | `/state` | Current episode metadata |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## Action space

Every call to `/step` requires a JSON body with this shape:

```json
{
  "action_type":  "label_bug",
  "severity":     "P1",
  "issue_id":     "ISS-1000",
  "duplicate_of": null,
  "reasoning":    "Has a stack trace pointing to auth.js line 142."
}
```

| Field | Type | Required | Values |
|---|---|---|---|
| `action_type` | string | Yes | `label_bug`, `label_feature`, `label_duplicate`, `label_invalid`, `label_question` |
| `severity` | string | Yes | `P0` (critical), `P1` (high), `P2` (medium), `P3` (low) |
| `issue_id` | string | Yes | Must match the `issue_id` in the current observation |
| `duplicate_of` | string | If duplicate | The `issue_id` of the original issue |
| `reasoning` | string | No | One-sentence explanation — not scored |

---

## Observation space

Every response from `/reset` and `/step` returns this shape:

```json
{
  "issue_id":             "ISS-1000",
  "title":                "Login fails with 500 error on mobile Safari",
  "body":                 "Steps: 1) Open app on iPhone... Stack trace: ...",
  "reporter":             "alice_dev",
  "created_at":           "2025-01-01",
  "comments_count":       3,
  "has_stack_trace":      true,
  "mentioned_components": ["auth"],
  "issues_remaining":     7,
  "last_action_result":   "Correct label (label_bug) +0.6 | Correct severity (P0) +0.3",
  "cumulative_score":     0.9,
  "done":                 false
}
```

| Field | Type | Description |
|---|---|---|
| `issue_id` | string | Unique identifier (e.g. `ISS-1042`) |
| `title` | string | Issue title |
| `body` | string | Full description and any stack traces |
| `reporter` | string | Username who filed the issue |
| `created_at` | string | Date filed (YYYY-MM-DD) |
| `comments_count` | int | Number of follow-up comments |
| `has_stack_trace` | bool | Strong signal for `label_bug` |
| `mentioned_components` | list | System components referenced in the body |
| `issues_remaining` | int | Issues left to triage in this episode |
| `last_action_result` | string | Reward breakdown from the last step |
| `cumulative_score` | float | Running total reward this episode |
| `done` | bool | `true` when all issues are triaged |

---

## Tasks

| Task ID | Difficulty | Inbox size | Description | Expected score |
|---|---|---|---|---|
| `easy` | Easy | 8 issues | Clear labels, no duplicates, strong signals (stack traces in all bugs) | ~0.80 |
| `medium` | Medium | 15 issues | 2 hidden duplicate pairs, paraphrased wording | ~0.55 |
| `hard` | Hard | 20 issues | Subtle duplicates, ambiguous labels, P0 miss penalty | ~0.30 |

---

## Reward function

Dense per-step rewards — the agent gets feedback after every decision:

| Event | Reward |
|---|---|
| Correct label | `+0.60` |
| Correct severity (exact match) | `+0.30` |
| Severity off by exactly 1 level | `+0.15` (partial credit) |
| Correct duplicate reference | `+0.10` |
| Missing a P0 critical bug | `−0.30` |
| Invalid `action_type` string | `−0.20` |
| Invalid `severity` / wrong `issue_id` | `−0.10` |

All per-step rewards are clipped to `[0.0, 1.0]`.

---

## Grader formulas

**Easy — label accuracy**
```
score = correct_labels / total_issues
```

**Medium — label accuracy + duplicate detection**
```
score = (0.70 × label_accuracy) + (0.30 × duplicate_accuracy)
```
A duplicate is correct only if both the label AND the `duplicate_of` reference match.

**Hard — label + severity + duplicate + P0 penalty**
```
score = (0.50 × label_acc) + (0.30 × severity_acc) + (0.20 × dup_acc) − p0_penalty
severity_acc: exact match = 1.0, off-by-one = 0.5, off-by-two+ = 0.0
p0_penalty:   +0.10 per P0 bug missed (labeled as anything except label_bug)
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values before running `inference.py`.

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Yes | HuggingFace or OpenAI API key |
| `ENV_URL` | No | Running environment URL (default: `http://localhost:7860`) |

---

## Baseline scores

Scores produced by running `inference.py` with `gpt-4o-mini` against the local server:

| Task | Baseline score |
|---|---|
| `easy` | 0.7875 |
| `medium` | 0.5133 |
| `hard` | 0.2850 |
| **average** | **0.5286** |

*Scores are deterministic — `temperature=0.0` is set in `inference.py`.*

---

## Project structure

```
bug_triage_env/
├── __init__.py              exports BugAction, BugObservation
├── client.py                HTTP client for agents to connect
├── models.py                BugAction + BugObservation Pydantic models
├── issue_generator.py       21 issue templates, 3 inbox generators
├── graders.py               3 deterministic task graders
├── openenv.yaml             OpenEnv spec compliance metadata
├── inference.py             Baseline agent — hackathon required
├── pyproject.toml           Project metadata and dependencies
├── README.md                This file
├── .env.example             Environment variable template
└── server/
    ├── __init__.py
    ├── app.py               FastAPI application
    ├── environment.py       BugTriageEnvironment — reset/step/state
    ├── requirements.txt     Pinned dependencies
    ├── Dockerfile           Builds from openenv-base, runs on port 7860
    ├── data/                Optional static issue datasets
    └── tests/
        └── test_environment.py   48 tests — all passing
```

---

## Running tests

```bash
cd bug_triage_env
pytest server/tests/test_environment.py -v
# Expected: 48 passed
```

---

## Deploying to HuggingFace Spaces

```bash
# Option 1 — using openenv CLI (recommended)
pip install openenv-core
openenv validate
openenv push --repo-id YOUR_USERNAME/bug-triage-env

# Option 2 — using git directly
git init
git add .
git commit -m "Bug Triage Env v1.0.0"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/bug-triage-env
git push origin main
```

After pushing, the Space builds automatically. Test it:

```bash
curl https://YOUR_USERNAME-bug-triage-env.hf.space/health
```

---

## Pre-submission checklist

```
✅ docker build -t bug-triage-env -f server/Dockerfile . — succeeds
✅ docker run -p 7860:7860 bug-triage-env — server starts
✅ curl localhost:7860/health — returns {"status":"ok"}
✅ curl -X POST localhost:7860/reset -d '{"task_id":"easy"}' — returns observation
✅ HuggingFace Space URL responds to /health
✅ openenv validate — passes
✅ python inference.py — completes, prints 3 scores
✅ All 3 graders return varied scores (not constant)
✅ inference.py is in root directory
✅ API_BASE_URL, MODEL_NAME, HF_TOKEN documented in .env.example
```

---

## Hackathon context

- **Event:** Meta PyTorch OpenEnv Hackathon × SST India 2026
- **Round 1 deadline:** April 8, 2026
- **Finale:** April 25–26, Bangalore (in-person, 48 hours)
- **Prize pool:** $30,000 USD
- **Domain:** Bug / issue triage — real-world task every software team performs daily
