# bug_triage_env/server/Dockerfile
#
# Production Dockerfile for the Bug Triage Environment.
#
# IMPORTANT: Build context MUST be the bug_triage_env/ project root (not its parent):
#   docker build -t bug-triage-env -f server/Dockerfile .
#
# Run locally:
#   docker run -p 7860:7860 bug-triage-env
#
# Test:
#   curl http://localhost:7860/health
#   curl -X POST http://localhost:7860/reset \
#        -H "Content-Type: application/json" \
#        -d '{"task_id": "easy"}'
#
# HuggingFace Spaces automatically builds this on every push to main.
# The Space must be tagged with "openenv" for the validator to find it.
#
# Container layout:
#   /app/           <- WORKDIR
#     __init__.py
#     models.py
#     issue_generator.py
#     graders.py
#     client.py
#     inference.py
#     openenv.yaml
#     server/
#       __init__.py
#       app.py
#       environment.py
#       requirements.txt
#       Dockerfile
#       tests/
#
# IMPORTANT:
#   - Build context = project root (bug_triage_env/)
#   - Must build FROM openenv-base (required for openenv validate)
#   - Must expose port 7860 (HuggingFace Spaces default)
#   - PYTHONPATH=/app ensures imports resolve from /app
#   - uvicorn starts from WORKDIR=/app so relative imports work

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

ENV PYTHONPATH=/app

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]
