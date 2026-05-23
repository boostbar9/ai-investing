# Temporal Worker

Runs the agent graph as a durable Temporal workflow (issue #3, §5).

## Run

```bash
# 1. Start Temporal (already in infra/docker/docker-compose.yml)
make up

# 2. Run worker
python -m apps.worker.main
```

Environment variables:

- `TEMPORAL_HOST` (default `localhost:7233`)
- `TEMPORAL_NAMESPACE` (default `default`)
- `TEMPORAL_TASK_QUEUE` (default `ai-investing`)
- `OLLAMA_HOST` (passed through to activities)

## Testing

Tests use `temporalio.testing.WorkflowEnvironment`'s in-memory dev server so
no external Temporal is required:

```bash
pytest apps/worker -q
```
