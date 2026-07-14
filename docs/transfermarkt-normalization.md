# Transfermarkt normalization

Run against a completed raw season directory:

```bash
mvp transfermarkt normalize \
  --run-directory data/raw/transfermarkt/run_date=2026-07-10/run_id=20260710T194135Z-f8d3644c/competition=GB1/season=2025
```

The command accepts only terminal raw manifests (`succeeded` or `partial`).
It computes a signature across the manifest, player rows and valuation
responses, writes through a staging directory and atomically promotes the
result. Repeating an unchanged run returns the existing normalized partition;
an interrupted attempt cannot block a retry.

Outputs:

```text
data/normalized/transfermarkt/
  competition=GB1/
    season=2025/
      run_id=<source-run-id>/
        players.parquet
        player_valuations.parquet
        data_quality_issues.parquet
        normalization_summary.json
```

The numeric source field `y` becomes `market_value_eur`.
Retirement rows with `mw="-"` and `y=0` retain the source zero in
`source_market_value_eur`, but set `market_value_eur` to null and
`is_valid_for_model` to false.

The valuation natural key is:

```text
transfermarkt_player_id + valuation_date
```
