CREATE TABLE IF NOT EXISTS players (
  player_id BIGINT PRIMARY KEY, display_name TEXT NOT NULL, birth_date DATE, position_group TEXT
);
CREATE TABLE IF NOT EXISTS player_source_ids (
  player_id BIGINT REFERENCES players(player_id), source TEXT NOT NULL, source_player_id TEXT NOT NULL,
  match_method TEXT NOT NULL, confidence NUMERIC(5,4), PRIMARY KEY (source, source_player_id)
);
CREATE TABLE IF NOT EXISTS matches (
  match_id BIGINT PRIMARY KEY, competition TEXT NOT NULL, season TEXT NOT NULL, kickoff_at TIMESTAMPTZ,
  home_team_id BIGINT, away_team_id BIGINT, source_url TEXT
);
CREATE TABLE IF NOT EXISTS player_match_ratings (
  player_id BIGINT REFERENCES players(player_id), match_id BIGINT REFERENCES matches(match_id),
  rating_version TEXT NOT NULL, rating NUMERIC(5,3), minutes NUMERIC(6,2), features JSONB,
  PRIMARY KEY (player_id, match_id, rating_version)
);
CREATE TABLE IF NOT EXISTS player_match_features (
  player_id BIGINT REFERENCES players(player_id), match_id BIGINT REFERENCES matches(match_id),
  feature_version TEXT NOT NULL, match_datetime TIMESTAMPTZ, position_group TEXT, minutes NUMERIC(6,2),
  features JSONB NOT NULL, source_hash TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (player_id, match_id, feature_version)
);
CREATE TABLE IF NOT EXISTS market_valuations (
  player_id BIGINT REFERENCES players(player_id), valuation_date DATE, value_eur BIGINT,
  source TEXT NOT NULL, PRIMARY KEY (player_id, valuation_date, source)
);
CREATE TABLE IF NOT EXISTS valuation_estimates (
  player_id BIGINT REFERENCES players(player_id), scored_at TIMESTAMPTZ, model_version TEXT,
  estimate_eur BIGINT, lower_eur BIGINT, upper_eur BIGINT, direction TEXT, confidence NUMERIC(5,4),
  predicted_pct_change DOUBLE PRECISION, probability_value_increase DOUBLE PRECISION,
  PRIMARY KEY (player_id, scored_at, model_version)
);
ALTER TABLE valuation_estimates
  ADD COLUMN IF NOT EXISTS predicted_pct_change DOUBLE PRECISION;
ALTER TABLE valuation_estimates
  ADD COLUMN IF NOT EXISTS probability_value_increase DOUBLE PRECISION;
CREATE TABLE IF NOT EXISTS valuation_match_impacts (
  player_id BIGINT REFERENCES players(player_id),
  match_id BIGINT REFERENCES matches(match_id),
  replay_run_id TEXT NOT NULL,
  replay_sequence INTEGER,
  estimate_eur BIGINT,
  lower_eur BIGINT,
  upper_eur BIGINT,
  estimated_value_delta_eur BIGINT,
  probability_value_increase DOUBLE PRECISION,
  valuation_status TEXT NOT NULL,
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (player_id, match_id, replay_run_id)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id TEXT PRIMARY KEY, pipeline TEXT NOT NULL, status TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ, counts JSONB, error TEXT
);
CREATE TABLE IF NOT EXISTS player_form_state (
  player_id BIGINT REFERENCES players(player_id), rating_version TEXT NOT NULL,
  last_match_datetime TIMESTAMPTZ, ewm_numerator DOUBLE PRECISION NOT NULL,
  ewm_denominator DOUBLE PRECISION NOT NULL, form_rating_ewm DOUBLE PRECISION,
  rolling_3_match_rating DOUBLE PRECISION, rolling_20_match_rating DOUBLE PRECISION,
  rolling_3_history JSONB NOT NULL, rolling_20_history JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (player_id, rating_version)
);
CREATE TABLE IF NOT EXISTS processed_match_partitions (
  competition TEXT NOT NULL, season TEXT NOT NULL, match_id BIGINT NOT NULL,
  stage TEXT NOT NULL, source_hash TEXT NOT NULL, model_version TEXT,
  processed_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (competition, season, match_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_player_match_ratings_player
  ON player_match_ratings (player_id, match_id);
CREATE INDEX IF NOT EXISTS idx_market_valuations_player_date
  ON market_valuations (player_id, valuation_date);
CREATE INDEX IF NOT EXISTS idx_valuation_estimates_player_scored
  ON valuation_estimates (player_id, scored_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_valuation_estimates_player_model
  ON valuation_estimates (player_id, model_version);
CREATE INDEX IF NOT EXISTS idx_valuation_match_impacts_player_match
  ON valuation_match_impacts (player_id, match_id, scored_at DESC);
