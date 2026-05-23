# ai-investing — one-command Makefile
# Acceptance criterion §16: clone -> first backtest in < 30 min

.PHONY: help setup setup-node setup-python setup-llms setup-db \
        setup-windows pull-models pull-models-7900xt \
        dev up down logs ps \
        test test-node test-python lint format typecheck \
        backtest nightly pretrain retune schedules doctor first-run \
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

setup-llms:  ## Pull DeepSeek R1 and Qwen 2.5 via Ollama (workstation profile)
	docker compose -f infra/docker/docker-compose.yml up -d ollama
	@sleep 5
	docker compose -f infra/docker/docker-compose.yml exec -T ollama ollama pull deepseek-r1:70b || true
	docker compose -f infra/docker/docker-compose.yml exec -T ollama ollama pull qwen2.5:72b || true

# ---------- Local (Windows / single-PC) setup ----------
# These targets assume Ollama is installed on the host (not in Docker) and
# work for Path A (Ollama on Windows) and Path B (WSL2 + ROCm).
# See docs/runbooks/local-setup-windows.md for the full guide.

setup-windows: setup-node setup-python setup-db  ## One-shot setup for a local Windows PC (Path A/B)
	@echo ""
	@echo "Next: set HARDWARE_PROFILE in your env (e.g. rx_7900_xt), then run:"
	@echo "  make pull-models"
	@echo ""
	@echo "See docs/runbooks/local-setup-windows.md for details."

pull-models:  ## Pull every Ollama model required by HARDWARE_PROFILE (defaults to active profile)
	@echo "Resolving models for active hardware profile..."
	@uv run python -c "from packages.agents.model_profiles import all_models, active_profile; p = active_profile(); print(f'Profile: {p.name} ({p.description})'); [print(f'  - {m}') for m in all_models()]"
	@echo ""
	@uv run python -c "from packages.agents.model_profiles import all_models; print('\n'.join(all_models()))" | while read model; do \
		echo ">>> ollama pull $$model"; \
		ollama pull $$model || exit 1; \
	done
	@echo ""
	@echo "All models pulled. Quick smoke test:"
	@echo "  ollama run $$(uv run python -c 'from packages.agents.model_profiles import all_models; print(all_models()[0])') 'Reply with OK.'"

pull-models-7900xt:  ## Convenience: force the rx_7900_xt profile and pull its models
	@HARDWARE_PROFILE=rx_7900_xt $(MAKE) pull-models

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

pretrain:  ## First-install bootstrap: pull 20yr daily + 90d intraday + FRED macro into data/parquet
	uv run python -m packages.data.pretrain

retune:  ## Walk-forward retune of strategy params (writes data/params/champion.json)
	uv run python -m packages.data.jobs.weekly_retune

schedules:  ## Install Temporal schedules for nightly_refresh + weekly_retune
	uv run python -m packages.data.jobs.scheduler

doctor:  ## Readiness check: which data sources work, cache status, champion params
	uv run python -m tools.doctor

first-run: doctor pretrain retune  ## End-to-end first install: doctor → pretrain → retune
	@echo ""
	@echo "✅ First-run complete. The bot is trained on real history."
	@echo "   Run 'make dev' to start the cockpit and start the nightly training loop."

# ---------- House-keeping ----------
clean:
	rm -rf node_modules .turbo dist build .next .venv
	find . -type d -name __pycache__ -exec rm -rf {} +

reset: down clean  ## Nuke local state (DB volumes preserved)
	@echo "Reset complete."
