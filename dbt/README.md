# Market Value Pulse dbt project

This dbt project models the PostgreSQL serving database without replacing
the Polars feature pipelines.

## Layers

- `staging`: typed views over the application tables.
- `intermediate`: latest-record and player-level aggregation logic.
- `marts`: player outlook, model monitoring, and pipeline health.

## Run locally

Start and load PostgreSQL first:

```bash
docker compose up -d postgres db-init
```

Install the optional dbt dependencies:

```bash
uv sync --extra dbt
```

Validate the connection and build all models and tests:

```bash
uv run --extra dbt dbt debug \
  --project-dir dbt \
  --profiles-dir dbt

uv run --extra dbt dbt build \
  --project-dir dbt \
  --profiles-dir dbt
```

The default profile matches the local Docker configuration. Override any
connection value with `DBT_POSTGRES_HOST`, `DBT_POSTGRES_PORT`,
`DBT_POSTGRES_USER`, `DBT_POSTGRES_PASSWORD`, or `DBT_POSTGRES_DB`.
