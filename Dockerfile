FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer, cached across `src` changes. `pip install -e .` only
# needs `pyproject.toml` to resolve/install the ~170 MB of third-party deps
# (lingua is the big one); hatchling's editable backend still needs the
# `src/app` package directory to *exist* to build the editable redirect, so
# a stub `__init__.py` stands in for the real source at this layer. Docker
# only invalidates a layer when the files it COPYs change, so this RUN stays
# CACHED for any `src`-only change — see docs/deployment.md.
COPY pyproject.toml README.md ./
RUN mkdir -p src/app && touch src/app/__init__.py \
    && pip install --upgrade pip && pip install -e .

# Real source, its own layer: replaces the stub. The editable install above
# already points here, but the deps are not re-resolved (`--no-deps`) — this
# rerun is just cheap bookkeeping (regenerates the editable redirect/metadata
# for whatever files actually exist now), not a dependency install.
COPY src ./src
RUN pip install --no-deps -e .

COPY alembic.ini ./
COPY migrations ./migrations

ENV PYTHONPATH=/app/src

# Root bo'lib yugurmaymiz: konteyner buzilsa ham image ichida yozish huquqi
# minimal bo'lsin. UID 1000 — host'dagi odatiy birinchi foydalanuvchi bilan
# bir xil, `./data` bind-mount'i shu tufayli yoziladigan bo'lib qoladi.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

# `api` uchun (compose'da bot/worker/migrate'da o'chirilgan): auth talab
# qilmaydigan, hech narsa oshkor qilmaydigan `/healthz`.
# `curl` emas, Python: image'ga yangi apt paketi qo'shilmaydi (yuza kichik
# qoladi) va urllib baribir mavjud.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "app.main", "bot"]
