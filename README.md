# Market Value Pulse

Market Value Pulse is an incremental football data platform that combines Transfermarkt valuation histories with WhoScored match events, derives player-performance features, produces transparent post-match ratings and projects market-value movement with uncertainty.

The repository is designed for continuous operation. A completed match is the unit of work: raw source data is retained, normalized and feature partitions are idempotent, only new or changed feature partitions are rated, and database writes use stable upsert keys.

## Current implementation

- Responsible WhoScored competition/season discovery, acquisition and normalization
- Transfermarkt roster and historical valuation ingestion and normalization
- Immutable raw runs with manifests, retries and explicit partial/failure states
- Partitioned normalized Parquet tables with schema/range/duplicate validation
- Self-contained Parquet-native inference for xG, xGOT, xA, xT and xPV enrichment
- Native, position-aware `post_match_v2` rating with conservative calibration
- Versioned pass priors, z-score statistics and rating configuration
- Content-hash state for new or changed match detection
- Exponentially weighted, rolling-three and rolling-20 player form state
- Conservative WhoScored-to-Transfermarkt entity resolution and a review queue
- Leakage-safe valuation features, Bayesian model, OLS benchmark and uncertainty ranges
- Idempotent PostgreSQL loading of matches, features, ratings, form and valuations
- FastAPI backed by PostgreSQL in Docker, with a Parquet fallback outside Docker
- Next.js player dashboard connected to the real API
- Chronological replay of the latest eight real matches
- Version-controlled data catalog and lineage metadata

The feature engine, post-match rating, state management and valuation workflow all live in this repository. The five fitted feature-model families are treated as versioned read-only artifacts.

## Architecture and storage lifecycle

```text
source pages
  → data/raw/                 immutable source responses and run manifests
  → data/normalized/          parsed source tables, partitioned by match/run
  → data/features/            enriched actions, player-match features and ratings
  → data/state/               processed hashes and current player form state
  → data/modeling/            valuation training/scoring artifacts
  → data/serving/             API-ready Parquet snapshots
  → PostgreSQL → FastAPI → Next.js
```

The source tree is intentionally flat—there is no second `market_value_pulse/` package below `src/`:

```text
src/
├── api/
├── database/
├── features/
├── entity_resolution/
├── ingestion/
├── pipelines/
├── ratings/
├── serving/
├── valuation/
└── cli.py
```


## Quick demonstration

After the prepared data and model artifacts are available, start the complete
stack with one command:

```bash
docker compose up --build
```

Open:

- Dashboard: http://localhost:3000
- API documentation: http://localhost:8000/docs
- API health: http://localhost:8000/health

To reproduce the prepared June 3, 2026 forecast snapshot with the promoted
valuation model:

```bash
TM_NORM="$(
  find data/normalized/transfermarkt/competition=GB1/season=2025 \
    -mindepth 1 -maxdepth 1 -type d \
  | sort | tail -n 1
)"

uv run mvp model build-current-features \
  --competition EPL \
  --season 2025-2026 \
  --as-of-date 2026-06-03 \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet \
  --ratings data/features/ratings/competition=EPL

uv run mvp model score \
  --model-version active \
  --output data/serving/player_valuation_predictions.parquet

uv run mvp serving build \
  --competition EPL \
  --season 2025-2026 \
  --ratings data/features/ratings/competition=EPL/season=2025-2026/player_match_ratings.parquet \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet \
  --predictions data/serving/player_valuation_predictions.parquet

uv run mvp database load \
  --competition EPL \
  --season 2025-2026 \
  --ratings data/features/ratings/competition=EPL/season=2025-2026/player_match_ratings.parquet \
  --form-state data/state/ratings/competition=EPL/season=2025-2026/player_form_state.parquet \
  --serving-root data/serving
```

The dashboard then displays the latest published Transfermarkt valuation, the
model midpoint, a 90% predictive range, the probability of an increase, recent
form and match-level performance drivers.

## Setup

Requirements are Python 3.12, `uv`, Node 22 for local frontend development, Docker for the full stack and Chromium for live WhoScored acquisition.

```bash
uv sync --extra dev
uv run playwright install chromium
uv run mvp --help
```

Model training has heavier optional dependencies:

```bash
uv sync --extra dev --extra train
```

To reuse an existing download in a fresh copy of the repository without
copying stale generated outputs:

```bash
scripts/migrate-existing-data.sh \
  "/path/to/old-repository" \
  "/path/to/new-repository" \
  "/path/to/feature-model-artifacts"
```

This copies only `data/raw`, `data/normalized` and the supplied fitted feature
artifacts. Features, rating state, serving snapshots and replays are rebuilt.

## 1. Acquire the EPL 2025/26 season

The WhoScored command needs only the configured competition and season. Omit the cap on the first backfill:

```bash
uv run mvp whoscored ingest \
  --competition EPL \
  --season 2025-2026 \
  --workers 2 \
  --delay-ms 1500
```

For later scheduled runs, select only the newest missing completed matches:

```bash
uv run mvp whoscored ingest \
  --competition EPL \
  --season 2025-2026 \
  --max-new-matches 8 \
  --workers 2 \
  --delay-ms 1500
```

Existing normalized `_SUCCESS.json` markers and calendar dates are checked before the cap is applied. Re-running the same command therefore skips completed matches, and future fixtures cannot starve the newest eligible match. Live/incomplete matches are retained in raw storage as `deferred_incomplete` and remain eligible for a later run.

Fetch and normalize Transfermarkt:

```bash
uv run mvp transfermarkt ingest \
  --league-config config/leagues/GB1.json \
  --season 2025 \
  --concurrency 3 \
  --requests-per-minute 30 \
  --timeout-seconds 30 \
  --max-retries 4

TM_MANIFEST="$(
  find data/raw/transfermarkt -type f \
    -path '*/competition=GB1/season=2025/manifest.json' \
  | sort | tail -n 1
)"

uv run mvp transfermarkt normalize \
  --run-directory "$(dirname "$TM_MANIFEST")"
```

Acquisition commands display overall progress, elapsed time, ETA, current match and outcome counts. The same progress events are retained as JSONL in each run directory.

## 2. Install the feature-model artifacts


```text
models/features/
├── goal_probability/xpv_action_v1/{metadata.json,model.json}
├── xa/xa_action_v1/{metadata.json,model.joblib}
├── xg/xg_shot_v1/{metadata.json,metadata.joblib,model.joblib}
├── xgot/xgot_shot_v1/{metadata.json,metadata.joblib,model.joblib}
└── xthreat/xt_action_v1/{metadata.json,model.json}
```


```bash
uv run mvp enrichment score-season \
  --competition EPL \
  --season 2025-2026 \
  --max-matches 1 \
  --prepare-only
```

## 3. Materialize features, ratings, serving data and PostgreSQL

Start PostgreSQL and the idempotent schema initializer. This also upgrades an existing Docker volume:

```bash
docker compose up -d postgres db-init
```

Locate the latest normalized Transfermarkt run:

```bash
TM_NORM="$(
  find data/normalized/transfermarkt/competition=GB1/season=2025 \
    -mindepth 1 -maxdepth 1 -type d \
  | sort | tail -n 1
)"
```

Run the stateful pipeline:

```bash
uv run mvp pipeline materialize \
  --competition EPL \
  --season 2025-2026 \
  --as-of-date 2026-05-24 \
  --transfermarkt-players "$TM_NORM/players.parquet" \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --load-database
```

On the first run this command:

1. enriches every unprocessed normalized match;
2. fits and saves rating priors/z-score statistics from the full season;
3. calculates one immutable rating per player-match;
4. initializes EWM, rolling-three and rolling-20 form state;
5. builds an exact unique-name player mapping and review queue;
6. scores the active valuation model if one has been promoted;
7. builds serving Parquets and idempotently upserts PostgreSQL.

On later runs, completed feature partitions are skipped. Ratings compare each `player_match_features.parquet` SHA-256 with `processed_matches.parquet`. A genuinely new match is scored with frozen artifacts and appended to form state. If an already-rated historical partition changes, the season statistics and ratings are refitted before form state is rebuilt; this prevents a corrected feature population from being scored against stale normalization statistics. PostgreSQL separately records the feature hash and skips unchanged feature partitions.

The important state/artifacts are:

```text
models/ratings/post_match_v2/competition=EPL/season=2025-2026/
├── rating_model_config.json
├── feature_schema.json
├── zscore_stats.parquet
├── pass_completion_priors.parquet
├── season_primary_positions.parquet
└── career_primary_positions.parquet

data/state/ratings/competition=EPL/season=2025-2026/
├── processed_matches.parquet
└── player_form_state.parquet

data/features/ratings/competition=EPL/season=2025-2026/
└── player_match_ratings.parquet
```

Run individual stages when debugging:

```bash
uv run mvp enrichment score-season --competition EPL --season 2025-2026
uv run mvp ratings fit --competition EPL --season 2025-2026
uv run mvp ratings update --competition EPL --season 2025-2026
uv run mvp entities build-player-mapping --competition EPL --season 2025-2026
uv run mvp serving build --competition EPL --season 2025-2026
uv run mvp database load --competition EPL --season 2025-2026
```

## Post-match rating model

`post_match_v2` retains the supplied rating-v3 feature and position logic while using a more conservative final calibration. Counting statistics use a 30–90 minute denominator and log-per-90 transforms. Open-play progressive passes use the notebook's zone-aware 28.6/14.3/9.5 thresholds and count completed passes only. Features are standardized by season and position and clipped to control outliers. Forwards, midfielders and defenders use different transparent weights over threat, creation, progression, retention, attacking xPV, defensive prevention and finishing. Goalkeepers use goals prevented and save percentage from shots assigned to the keeper actually on the pitch; own goals are excluded from the shot-stopping sample.

The final composite uses minutes reliability and a bounded hyperbolic-tangent conversion centered at 6. V2 maps a `z=2.5` composite to about `8.1`, rather than about `9.1`, and applies only a small residual decisive-action bonus because goals and assists already affect finishing, creation and xPV. Explicit penalties remain for cards, missed big chances and own goals. The canonical output column is `post_match_rating`.

See `docs/rating-model.md` for the component weights, fit/apply boundary and known assumptions.

## Entity resolution

Automatic mappings are accepted only when the normalized name is unique in both sources. Missing and ambiguous candidates are written to:

```text
data/normalized/entity_resolution/player_mapping_review.parquet
```

Manual overrides may be supplied as CSV or Parquet with two columns:

```text
whoscored_player_id,transfermarkt_player_id
```

```bash
uv run mvp entities build-player-mapping \
  --competition EPL \
  --season 2025-2026 \
  --manual-overrides config/player_mapping_overrides.csv
```

The resulting crosswalk is validated as one-to-one in both directions. The system deliberately does not fuzzy-match uncertain players silently.


## Modeling rationale

The valuation target is the log change between consecutive Transfermarkt
observations:

```text
log(current market value / previous market value)
```

A hierarchical Bayesian linear regression was selected because football data
has clear nested structure. Players operate in different positions, and the
effect of age or recent form is not expected to be identical for a goalkeeper,
defender, midfielder and forward.

The model contains:

- global feature effects shared across all players;
- position-specific intercepts;
- position-specific age effects;
- position-specific recent-form effects;
- partially pooled player effects;
- a Student-t likelihood to reduce sensitivity to unusually large valuation
  updates.

Partial pooling lets players with substantial history develop an individual
effect while shrinking players with limited data toward the position and
population averages. Players not observed during training receive a zero player
effect rather than an unreliable extrapolated effect.

WhoScored event data is enriched with expected goals (xG), expected goals on
target (xGOT), expected assists (xA), expected threat (xT), expected possession
value (xPV), progression, ball retention, defensive threat prevention and
finishing overperformance. Goals, assists and minutes remain available, but the
model also sees the underlying actions that created or prevented scoring
opportunities.

Recent performance is represented through minutes-weighted valuation-interval
aggregates, an exponentially weighted form rating, rolling-three and rolling-20
form, recent trend and position-aware component averages. This gives recent
matches more influence without discarding longer-term performance history.

The output includes:

- a median projected market value;
- a 90% posterior predictive range;
- the probability that value has increased;
- a direction and confidence indicator.

The estimate is not intended to replace an official Transfermarkt valuation. It
is a model-based estimate of how value may have moved after the latest published
observation.

## Final model evaluation

The final model was evaluated on a chronological holdout rather than a random
split. This prevents later valuation observations from being mixed into the
training sample.

| Metric | Result |
|---|---:|
| Training observations | 2,368 |
| Holdout observations | 933 |
| Holdout split date | 2025-05-30 |
| Bayesian MAE in log change | 0.1247 |
| Approximate MAE percentage | 13.3% |
| Holdout R² | 0.388 |
| Spearman rank correlation | 0.704 |
| Direction accuracy | 56.9% |
| 90% predictive interval coverage | 90.4% |
| OLS MAE | 0.1352 |
| Zero-change baseline MAE | 0.1658 |

The Bayesian model was promoted because it produced lower holdout MAE than both
OLS and the zero-change baseline, retained stronger rank correlation and
provided calibrated predictive uncertainty.

A forecast can decrease even when a player has had a reasonable season. The
model forecasts marginal movement from the player's existing valuation, not
absolute football ability. Age, previous valuation, recent form and the fact
that an already highly valued player has less room for further appreciation can
all produce a flat or declining estimate.

## Valuation model

The target is the log change between consecutive Transfermarkt observations:

```text
log(current_market_value / previous_market_value)
```

Only matches strictly inside a valuation interval enter its performance aggregates, preventing leakage. Features reproduce the successful notebook specification: prior value/change, age and age squared, interval/calendar controls, minutes, appearances, start share, average and recency-weighted ratings, last-90-day rating/trend, rating volatility, position-aware rating components and attacking outcome rates.

The default ratings input is the competition directory, so every available
`season=*/player_match_ratings.parquet` partition is loaded. The final training
run used the available EPL performance history from 2018/19 through 2025/26.
Older seasons increase repeated observations per player and expose the model to
a broader set of valuation regimes.

```bash
for season in \
  2018-2019 2019-2020 2020-2021 2021-2022 \
  2022-2023 2023-2024 2024-2025 2025-2026
do
  uv run mvp enrichment score-season \
    --competition EPL \
    --season "$season"

  uv run mvp ratings fit \
    --competition EPL \
    --season "$season"
done
```

Build the modeling table from all rating partitions and the latest normalized
Transfermarkt history:

```bash
TM_NORM="$(
  find data/normalized/transfermarkt/competition=GB1/season=2025 \
    -mindepth 1 -maxdepth 1 -type d \
  | sort | tail -n 1
)"

uv run mvp model build-features \
  --competition EPL \
  --season 2025-2026 \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet \
  --ratings data/features/ratings/competition=EPL

uv run mvp model train \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 2 \
  --target-accept 0.95
```

The primary estimator is the notebook's hierarchical Bayesian Student-t regression: global shrinkage, position intercepts, position-specific age/form adjustments and player-level partially pooled intercepts. Unseen players receive a zero player effect. It returns a median/mean estimate, a 90% predictive interval and the probability of an increase. OLS with HC3 errors and simple baselines are evaluated on the same chronological holdout. A candidate is promoted only when it beats the zero-change and OLS baselines and passes R², coverage and sampler-diagnostic gates; failed candidates remain inspectable without replacing `active.json`.

When new matches arrive but no new valuation label exists, rebuild current
features and score the saved active model without retraining:

```bash
uv run mvp model build-current-features \
  --competition EPL \
  --season 2025-2026 \
  --as-of-date YYYY-MM-DD \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet \
  --ratings data/features/ratings/competition=EPL

uv run mvp model score \
  --model-version active \
  --output data/serving/player_valuation_predictions.parquet
```

The scoring date should follow the latest eligible domestic match. A player
needs match minutes after their latest published valuation to receive a new live
forecast.

Retraining is appropriate when a new Transfermarkt valuation closes another labeled interval.

## Eight-match continuous-update replay

Use real historical matches for the demo instead of mock values:

```bash
uv run mvp pipeline replay \
  --competition EPL \
  --season 2025-2026 \
  --matches 8
```

Or select one player's latest eight appearances:

```bash
uv run mvp pipeline replay \
  --competition EPL \
  --season 2025-2026 \
  --player-id 12345 \
  --matches 8 \
  --valuations "$TM_NORM/player_valuations.parquet" \
  --mapping data/normalized/entity_resolution/player_mapping_exact.parquet
```

The replay processes matches oldest-to-newest in an isolated directory, enriches each match, scores native ratings and refreshes player form after every step. A `--prepare-only` replay deliberately reports `pending_rating_model` because it does not load feature/rating artifacts. Full replay rating rows report `succeeded`.

When the command receives valuations, an approved mapping and a requested valuation model, it also builds current valuation features after each match and writes a new estimate, 90% range, direction probability and—when a player is selected—the change from the previous replay step. Without all three inputs it reports `skipped_missing_inputs`. Selected-player replay deltas are published to the serving match-impact table; rerun the database load afterward when the API uses PostgreSQL.

## API and dashboard

After data is loaded, start the complete application stack:

```bash
docker compose up --build
```

- Dashboard: http://localhost:3000
- API/OpenAPI: http://localhost:8000/docs
- Health: http://localhost:8000/health

The player screen is wired to `/api/players` and `/api/players/{id}`. It supports search and selection, a local watchlist, valuation history, current estimate/range/confidence, rolling form and match-level component explanations. The catalog, lineage and pipeline-run views use `/api/catalog`, `/api/lineage` and `/api/pipeline-runs`.

In Docker, the API reads PostgreSQL. Without `DATABASE_URL`, it falls back to the Parquet tables under `data/serving/`, which is useful for local UI development.

## Data-source decisions and limitations

WhoScored was selected for rich player-level match events and because the existing scraper could be made incremental at match grain. Transfermarkt was selected for dated player market-value histories and broad player coverage. Both are public web sources rather than stable contracted APIs, so layouts, identifiers and availability can change. The pipeline retains raw evidence, uses low concurrency/rate limits, validates schemas and fails visibly when required payloads disappear. Operators remain responsible for the sources' terms and permitted use.

FBref and Understat were considered but not used in the primary pipeline: they would add another identifier space, do not replace Transfermarkt's historical valuation target and overlap with the event-derived metrics already available here. Proprietary feeds would improve identity stability, injury/context features, goalkeeper event attribution and legally supported continuous delivery.

Known limitations:

- exact-name entity resolution leaves legitimate aliases in a manual review
  queue because the system deliberately avoids unsafe fuzzy matches;
- a small number of appearances may retain an unknown position when neither a
  reliable match position nor sufficient historical starts are available;
- Transfermarkt values are subjective, irregularly dated and not transaction
  prices;
- the current model covers domestic EPL performance and does not include
  international matches, injuries not represented in match data, contract
  duration, salary, release clauses, transfer demand or club negotiating
  position;
- season-position z-scores should be fit only after a representative initial
  sample and then frozen for incremental scoring;
- late source corrections trigger a deterministic match rescore, but old model
  artifacts are not automatically refit;
- monetary replay requires an active trained valuation model and an approved
  source mapping;
- fitted model artifacts must only be distributed when their license or
  ownership permits it.

## What I would improve with more time

- strengthen entity resolution with aliases, date of birth, club and position
  evidence while preserving a manual review path;
- evaluate expanding-window rating normalization to remove any remaining
  full-season calibration leakage;
- add contract, injury, international-performance, opponent-strength and
  transfer-demand features;
- backtest promotion gates across multiple leagues and multiple rolling temporal
  folds;
- calibrate the post-match rating against an external benchmark while retaining
  the transparent component breakdown;
- add automated orchestration and scheduled source-health alerts for production
  operation.

## Tests

```bash
uv run pytest -q
cd frontend && npm run build
```

Tests cover discovery/idempotency, normalization and validation, enrichment adapters, chronological replay selection, Transfermarkt parsing, native rating artifacts and incremental hashing, entity resolution, serving contracts and leakage-safe valuation features.

More detailed operator notes are in `docs/whoscored-ingestion.md`, `docs/transfermarkt-normalization.md`, `docs/enrichment-and-replay.md`, `docs/rating-model.md`, `docs/stateful-pipeline.md` and `docs/valuation-model.md`.
