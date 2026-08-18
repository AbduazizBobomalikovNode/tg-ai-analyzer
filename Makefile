.PHONY: up down logs migrate revision fmt lint test shell

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
	ruff format src tests && ruff check --fix src tests

lint:
	ruff check src tests && mypy src

test:
	pytest -q

shell:
	docker compose exec postgres psql -U $${POSTGRES_USER} -d $${POSTGRES_DB}
