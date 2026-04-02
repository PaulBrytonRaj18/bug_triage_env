# bug_triage_env/server/Dockerfile
#
# Production Dockerfile for the Bug Triage Environment.
#
# Build:
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
# IMPORTANT:
#   - Must build FROM openenv-base (required for openenv validate)
#   - Must expose port 7860 (HuggingFace Spaces default)
#   - COPY . . must happen AFTER pip install to use Docker layer cache
#   - The CMD starts uvicorn from the project root so imports resolve

FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory to project root inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer as long as
# requirements.txt doesn't change, so rebuilds are fast.
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container.
# Structure inside container:
#   /app/models.py
#   /app/issue_generator.py
#   /app/graders.py
#   /app/client.py
#   /app/inference.py
#   /app/openenv.yaml
#   /app/server/__init__.py
#   /app/server/app.py
#   /app/server/environment.py
COPY . .

# HuggingFace Spaces and the openenv validator both expect port 7860
EXPOSE 7860

# Start the FastAPI server.
# --workers 1    keeps a single process so the shared env instance is safe
# --host 0.0.0.0 required for container networking
# --port 7860    matches EXPOSE above
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]
