.PHONY: up down logs migrate revision fmt lint test ci shell eval-golden eval-run

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
