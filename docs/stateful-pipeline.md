# Stateful incremental pipeline

## Match as the unit of work

The pipeline uses a stable match ID as its smallest processing unit. This makes the continuous path bounded and auditable:

```text
new completed match
→ normalized match partition
→ enriched feature partition
→ content hash comparison
→ player-match rating upserts
→ player form refresh
→ valuation rescore
→ PostgreSQL upserts
```

## Idempotency boundaries

| Stage | Completion/state boundary | Repeat behavior |
|---|---|---|
| WhoScored acquisition | immutable run manifest + normalized `_SUCCESS.json` | completed match is skipped |
| Transfermarkt acquisition | immutable run manifest | new run is additive |
| Enrichment | feature `_SUCCESS.json` + source/config/artifact signature | unchanged feature partition is skipped |
| Rating | `processed_matches.parquet` with feature SHA-256 | new hashes are scored; corrected historical hashes trigger a season refit |
| Form | `player_form_state.parquet` | rebuilt chronologically after changed ratings |
| PostgreSQL | primary keys + `processed_match_partitions` hash | unchanged features skipped; changed rows upserted |

Raw data is append-only. Normalized/features can be regenerated deterministically from their upstream layer. Operational state is kept outside both layers under `data/state/` so a completion marker is never mistaken for analytical data.

## First run

The first season run should be a representative backfill. It enriches all normalized matches, fits rating priors/z-score statistics, scores the full rating history and initializes form state. Fitting statistics from only one or two matches is technically possible but not defensible.

## Later run

A scheduled update fetches the newest missing completed matches with `--max-new-matches`. Enrichment skips only markers whose source, configuration and artifact signature still matches. The rating update compares feature hashes and applies the frozen rating artifacts. Database loading performs another hash check before serializing/upserting feature JSON.

If a source corrects an old match, its regenerated feature hash changes. Because season-position priors and z-scores depend on the historical population, the rating season is refitted and the form sequence is recalculated in match-time order. A genuinely new match continues to use the frozen artifacts and bounded append-only state update.

## Operational recovery

- Source unavailable: bounded retries; run/match failure remains visible and eligible for retry.
- Missing source payload/schema drift: no success marker; raw evidence and validation output remain for diagnosis.
- Partial season run: successful match partitions remain committed; failed partitions can be targeted later.
- Enrichment model missing: fail before `_SUCCESS.json`; `--prepare-only` can still validate the adapter.
- Ambiguous player identity: place the player in a review Parquet rather than invent a join.
- Database unavailable: Parquet outputs remain valid; rerun the idempotent load when PostgreSQL recovers.

## Database loading

The database stores normalized matches, JSON feature snapshots, ratings, current form, valuation histories, valuation estimates and pipeline runs. Stable primary keys protect against duplicates. `processed_match_partitions` stores the last feature hash loaded for each competition/season/match/stage.

The database is a serving projection, not the only copy of the data. Parquet partitions are the reproducible analytical handoff and can rebuild PostgreSQL.
