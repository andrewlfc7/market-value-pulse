# Feature enrichment and historical replay

## Contract

The feature engine reads one completed normalized WhoScored match partition:

```text
data/normalized/whoscored/competition=<key>/season=<season>/matches/match_id=<id>/
├── matches.parquet
├── player_matches.parquet
├── events.parquet
├── shots.parquet
└── _SUCCESS.json
```

`events.parquet` is the action source of truth. Passes are derived in memory,
so no normalized pass table is required. The feature output retains
`passes.parquet` alongside xG/xGOT, carries, xA, xT, xPV, defensive xPV and
player-match aggregates for reproducibility.

Source `persistent_id` becomes the canonical `event_uid`. Numeric `event_id`
is retained for source relations but is not assumed to be match-global;
relations use match and team context.

## Idempotency and provenance

Every `_SUCCESS.json` stores an input signature covering:

- normalized match objects;
- `config/features/scoring.json`;
- all five resolved model artifact directories;
- the feature implementation version.

An unchanged signature is skipped. A corrected source partition, changed
configuration or changed artifact automatically rescores that match. Models
are validated and loaded once per season/replay process.

Artifacts resolve from `models/features` or `MVP_FEATURE_MODELS_ROOT`.
Inference is Parquet-in/Parquet-out and has no external feature-store
dependency.

## Commands

Compatibility-only check:

```bash
uv run mvp enrichment score-season \
  --competition EPL \
  --season 2025-2026 \
  --max-matches 1 \
  --prepare-only
```

Full incremental feature scoring:

```bash
uv run mvp enrichment score-season \
  --competition EPL \
  --season 2025-2026
```

Eight-match player replay:

```bash
uv run mvp pipeline replay \
  --competition EPL \
  --season 2025-2026 \
  --player-id <whoscored_player_id> \
  --matches 8 \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet
```

The simulation applies already-fitted feature, rating and valuation artifacts
oldest-to-newest. Each step updates ratings/form and, when valuation inputs are
present, writes an estimate, 90% interval, increase probability and the change
from the previous replay step. Those real replay deltas are published to the
serving match-impact table for the selected player.

Valuation failures count as replay failures and produce a partial/failed final
state; they are never silently reported as success.
