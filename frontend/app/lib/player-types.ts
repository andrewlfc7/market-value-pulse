export type PlayerSummary = {
  player_id: number;
  display_name: string;
  position?: string | null;
  current_form_rating?: number | null;
  rolling_3_match_rating?: number | null;
  rolling_20_match_rating?: number | null;
  latest_valuation_date?: string | null;
  current_market_value_eur?: number | null;
  estimated_value_eur?: number | null;
  estimated_lower_eur?: number | null;
  estimated_upper_eur?: number | null;
  predicted_pct_change?: number | null;
  probability_value_increase?: number | null;
  valuation_model_version?: string | null;
  confidence?: number | null;
  direction?: string | null;
  refreshed_at?: string | null;
};

export type ValuationPoint = {
  valuation_date: string;
  value_eur: number;
  source?: string;
};

export type MatchImpact = {
  match_id: number;
  match_datetime: string;
  rating?: number | null;
  minutes?: number | null;
  explanation?: string;
  performance_impact_score?: number | null;
  impact_direction?: string;
  estimated_value_delta_eur?: number | null;
};

export type PlayerDetail = PlayerSummary & {
  valuation_history: ValuationPoint[];
  match_impacts: MatchImpact[];
};

export type View = "player" | "watchlist" | "catalog" | "health";
