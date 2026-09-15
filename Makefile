.PHONY: up down logs migrate revision fmt lint test ci shell eval-golden eval-run \
        perf-seed perf-bench

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f bot worker api

migrate:
	docker compose run --rm migrate

revision:
	docker compose run --rm migrate alembic revision --autogenerate -m "$(m)"

fmt:
	ruff format src tests evals && ruff check --fix src tests evals

lint:
	ruff check src tests evals && mypy src evals

test:
	pytest -q

ci: lint test   # GitHub Actions bilan bir xil

# Baholash (docs/architecture-v2.md, 10-bo'lim). Real DB/LLM'ga ulanadi —
# `pytest`/`ci` bularni yugurtirmaydi. Misol:
#   make eval-golden account=1 args="--n 400 --seed 1"
#   make eval-run args="--golden evals/golden.jsonl --config baseline --runs 3"
eval-golden:
	PYTHONPATH=src python -m evals.build_golden --account-id $(account) $(args)

eval-run:
	PYTHONPATH=src python -m evals.run_eval $(args)

shell:
	docker compose exec postgres psql -U $${POSTGRES_USER} -d $${POSTGRES_DB}

# SQL miqyosi o'lchovi (docs/architecture-v2.md, 11.1-bo'lim). Real LLM ham,
# Telegram ham chaqirilmaydi — embeddinglar sintetik birlik vektorlar.
# **Bir martalik** bazada yugurtiring: seeder jadvallarni TRUNCATE qiladi.
#
#   docker run -d --name tgai-perfpg -e POSTGRES_PASSWORD=perf -p 55433:5432 \
#     --shm-size=1g pgvector/pgvector:pg16 postgres -c shared_buffers=512MB \
#     -c work_mem=64MB -c maintenance_work_mem=512MB
#   export PERF_DATABASE_URL=postgresql+asyncpg://postgres:perf@localhost:55433/postgres
#   make perf-seed && make perf-bench
perf-seed:
	DATABASE_URL=$(PERF_DATABASE_URL) PYTHONPATH=src python -m alembic upgrade head
	PYTHONPATH=src python -m evals.perf.seed --dsn $(PERF_DATABASE_URL) $(args)

perf-bench:
	PYTHONPATH=src python -m evals.perf.bench --dsn $(PERF_DATABASE_URL) \
		--runs 5 --json perf-bench.json $(args)
