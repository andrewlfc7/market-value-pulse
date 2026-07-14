# WhoScored ingestion design

## Operator contract

The standard run is scoped by a logical competition and season. WhoScored IDs and URLs are implementation details resolved from the version-controlled competition registry.

```bash
uv run mvp whoscored ingest --competition EPL --season 2025-2026
```

The same command is suitable for a scheduled daily run. It always refreshes discovery, but downloads and normalizes only matches that do not have a successful normalized partition. Fixture dates are read from the calendar and future fixtures are deferred before `--max-new-matches N` is applied, so scheduled matches cannot consume the incremental cap.

## Stages

1. Resolve a competition alias to its WhoScored region and tournament IDs.
2. Fetch the tournament page and resolve the requested season label to a season ID.
3. Fetch the season page and choose its main fixture stage, applying a documented registry override when necessary.
4. Traverse the rendered fixture calendar in both directions and deduplicate stable match IDs.
5. Exclude preview-only fixtures and compare the remainder with normalized `_SUCCESS.json` markers.
6. Defer future fixture dates, sort eligible missing matches newest-first and apply the optional `--max-new-matches` cap.
7. Fetch selected match pages with low concurrency, jitter, and bounded exponential retries.
8. Persist the source page and extracted `matchCentreData` before normalization.
9. Validate and atomically write match, team, player-match, raw-event, event, and shot Parquets.

## Storage contract

```text
data/raw/whoscored/
└── run_date=YYYY-MM-DD/run_id=.../competition=EPL/season=2025-2026/
    ├── manifest.json
    ├── progress.jsonl
    ├── discovered_matches.{csv,parquet}
    ├── discovery/
    │   ├── tournament_page.html
    │   ├── season_page.html
    │   ├── catalog.json
    │   ├── discovery_summary.json
    │   └── fixture_calendar_pages/*.html
    ├── matches/match_id=.../{page.html,matchCentreData.json}
    ├── match_results.parquet
    └── failed_matches.csv

data/normalized/whoscored/
└── competition=EPL/season=2025-2026/matches/match_id=.../
    ├── raw_events.parquet
    ├── matches.parquet
    ├── teams.parquet
    ├── player_matches.parquet
    ├── events.parquet
    ├── shots.parquet
    ├── data_quality.json
    └── _SUCCESS.json
```

Raw runs are immutable and auditable. Normalized match partitions are deterministic. `_SUCCESS.json` is written last and is the idempotency boundary.

WhoScored match payloads do not consistently expose `minutesPlayed`, and bench
players are commonly labeled `position="Sub"`. Normalization therefore
reconciles appearances from the starting XI, regulation-clock substitution
events, dismissals and the nominal 90/120-minute match duration. Used
substitutes inherit the broad position group of their linked outgoing player;
the original `position` is retained, while `minutes_source` and
`position_group_source` record provenance. The feature adapter repeats this
reconciliation for historical normalized partitions created before these
fields existed, so raw data does not need to be fetched again.

## Failure behavior

| Condition | Recorded state | Next run |
|---|---|---|
| Tournament, season, or fixture stage missing | `failed_discovery` | Retry after source/config investigation |
| Calendar count below the configured completed-season expectation | Discovery warning | Run continues; inspect coverage summary |
| Preview fixture | Excluded from the fetch manifest | Discovered after its URL changes |
| Future calendar fixture | `deferred_future_fixture` | Not fetched; reconsidered on a later discovery run |
| Live/incomplete payload | `deferred_incomplete` | Fetched again because no success marker exists |
| Timeout, HTTP block, or missing payload | `failed_fetch` after retries | Included in `failed_matches.csv` |
| Parse, schema, range, or duplicate failure | `failed_normalization` | Raw page retained for debugging |
| Existing successful match | `skipped_existing` | No download or normalization |
| Missing match outside `--max-new-matches` | `not_selected_limit` | Eligible on a later run |

## Targeted recovery

Re-run failures without repeating full calendar discovery:

```bash
uv run mvp whoscored ingest \
  --competition EPL \
  --season 2025-2026 \
  --manifest data/raw/whoscored/.../failed_matches.csv
```

Use `--force` only when a normalized match must deliberately be rebuilt after a parser or schema change.

## Incremental scheduled command

```bash
uv run mvp whoscored ingest \
  --competition EPL \
  --season 2025-2026 \
  --max-new-matches 8 \
  --workers 2 \
  --delay-ms 1500
```

The terminal shows elapsed time, ETA, the current match, and outcome counters. `progress.jsonl` persists the same events for debugging and orchestration. Use `--no-progress` in non-interactive environments.
