# ai-investing — one-command Makefile
# Acceptance criterion §16: clone -> first backtest in < 30 min

.PHONY: help setup setup-node setup-python setup-llms setup-db \
        dev up down logs ps \
        test test-node test-python lint format typecheck \
        backtest nightly \
        clean reset

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------
setup: setup-node setup-python setup-db setup-llms  ## Full one-command setup
	@echo "✅ Setup complete. Run 'make dev' to start."

setup-node:  ## Install JS deps (pnpm)
	corepack enable
	corepack prepare pnpm@9 --activate
	pnpm install

setup-python:  ## Install Python deps (uv)
	uv sync --all-packages

setup-db:  ## Bring up Postgres+Timescale and run migrations
	docker compose -f infra/docker/docker-compose.yml up -d postgres dragonfly
	@sleep 3
	uv run alembic -c packages/data/alembic.ini upgrade head || true

setup-llms:  ## Pull DeepSeek R1 and Qwen 2.5 via Ollama
	docker compose -f infra/docker/docker-compose.yml up -d ollama
	@sleep 5
	docker compose -f infra/docker/docker-compose.yml exec -T ollama ollama pull deepseek-r1:70b || true
	docker compose -f infra/docker/docker-compose.yml exec -T ollama ollama pull qwen2.5:72b || true

# ---------- Run ----------
dev: up  ## Start full local stack
	@echo "Cockpit:  http://localhost:3000"
	@echo "API:      http://localhost:8000"
	@echo "Grafana:  http://localhost:3001"
	@echo "Temporal: http://localhost:8233"

up:  ## docker compose up -d (all services)
	docker compose -f infra/docker/docker-compose.yml up -d

down:  ## docker compose down
	docker compose -f infra/docker/docker-compose.yml down

logs:  ## Tail all logs
	docker compose -f infra/docker/docker-compose.yml logs -f --tail=100

ps:  ## List running services
	docker compose -f infra/docker/docker-compose.yml ps

# ---------- Quality ----------
test: test-node test-python  ## Run all tests

test-node:
	pnpm -r test

test-python:
	uv run pytest -q

lint:
	pnpm -r lint
	uv run ruff check .

format:
	pnpm -r format
	uv run ruff format .

typecheck:
	pnpm -r typecheck
	uv run mypy packages apps/api apps/telegram-bot

# ---------- Backtests ----------
backtest:  ## Run the standard backtest matrix locally
	uv run python -m packages.backtests.run --matrix standard

nightly:  ## Mirror the GH Actions nightly job locally
	uv run python -m packages.backtests.run --matrix nightly --strategies all --regimes all

# ---------- House-keeping ----------
clean:
	rm -rf node_modules .turbo dist build .next .venv
	find . -type d -name __pycache__ -exec rm -rf {} +

reset: down clean  ## Nuke local state (DB volumes preserved)
	@echo "Reset complete."
