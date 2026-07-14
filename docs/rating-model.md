# Post-match rating model

## Ownership and artifacts

`post_match_v2` is implemented in `src/ratings/model.py`. A first historical fit writes:

```text
models/ratings/post_match_v2/competition=EPL/season=YYYY-YYYY/
├── rating_model_config.json
├── feature_schema.json
├── zscore_stats.parquet
├── pass_completion_priors.parquet
├── season_primary_positions.parquet
└── career_primary_positions.parquet
```

Statistics are grouped by season and position. Later new-match updates apply these frozen artifacts. A changed historical feature partition triggers a full season refit because the normalization population changed.

## Transforms

Counting features use `adjusted_minutes = clip(minutes, 30, 90)`, convert to per 90 and then apply `log1p`. Feature z-scores are fitted by season and position and clipped to `[-4.5, 4.5]`.

Pass retention uses an empirical-Bayes completion prior. The season-position completion rate supplies the prior mean; observed between-match variance determines a prior strength clipped to `[5, 60]`. The feature is the smoothed completion rate above that prior.

## Transparent components

| Component | Formula |
|---|---|
| Threat | `0.40 z(log xG90) + 0.35 z(log xGOT90) + 0.25 z(log shots90)` |
| Creation | `0.55 z(log xA90) + 0.25 z(log key passes90) + 0.20 z(log big chances created90)` |
| Progression | `0.45 z(log progressive passes90) + 0.35 z(log progressive carries90) + 0.20 z(log final-third carries90)` |
| Retention | `z(pass completion above expected)` |
| Attacking xPV | `z(xPV added90)` |
| Defense | `0.60 z(opponent threat prevented90) + 0.40 z(defensive net threat reduction90)` |
| Finishing | `0.60 z(goals − xG) + 0.40 z(xGOT − xG)` after clipping |

Position weights:

| Position | Threat | Creation | Progression | Retention | xPV | Defense | Finishing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forward | 0.28 | 0.15 | 0.10 | 0.05 | 0.12 | 0.03 | 0.27 |
| Midfielder | 0.08 | 0.25 | 0.22 | 0.12 | 0.13 | 0.08 | 0.12 |
| Defender | 0.03 | 0.08 | 0.17 | 0.15 | 0.10 | 0.40 | 0.07 |

Each row is shrunk by `0.25 + 0.75 sqrt(clip(minutes / 90, 0, 1))`. The resulting composite is standardized again and converted to the 1–10 scale with `6 + 3.30 tanh(z / 3.30)`. This keeps the neutral point at 6 while reserving 9+ ratings for genuinely exceptional matches.

Final adjustments are deliberately readable:

- goals, assists and positive-only assist overperformance receive a small residual bonus capped at 0.50; the weights are intentionally modest because the same actions already contribute to finishing, creation and xPV;
- yellow and red cards receive direct penalties;
- missed big chances receive a capped count penalty while missed big-chance xG remains an analysis column;
- own goals receive a capped direct penalty.

## Goalkeepers

Shots are assigned to the goalkeeper whose active interval covers the shot minute. Goalkeeper performance combines standardized goals prevented and save percentage, with reliability based on shots on target faced and a small clean-sheet bonus only when at least three shots on target were faced.

## Output and state

The canonical score is `post_match_rating`. One rating is immutable for each `(season, match_id, whoscored_player_id, rating_version)`.

Player form is separate operational state:

- a 90-day half-life EWM using minutes as weight;
- a minutes-weighted rolling three-match rating;
- a minutes-weighted rolling 20-match rating.

The state table retains the EWM numerator/denominator and compact recent-match histories. A changed historical feature partition is replaced by key, then form is recomputed chronologically so late corrections do not corrupt state.

## Known assumptions

- Starter position codes are grouped explicitly. Substitute appearances use the player's primary season position and then their career starter position as a fallback.
- Progressive passes exclude set pieces, use zone-aware 28.6/14.3/9.5 forward-progress thresholds and count completed passes only.
- A big chance is an assisted shot with model xG at least 0.30.
- Very small position-season samples produce unstable z-scores; the initial fit should use a representative backfill.
- The current goalkeeper attribution handles substitutions from event intervals but cannot recover source events that are absent or incorrectly timestamped.
