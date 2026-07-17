select
    p.player_id,
    p.display_name,
    p.birth_date,
    p.position_group,
    f.rating_version,
    f.last_match_datetime,
    f.form_rating_ewm as current_form_rating,
    f.rolling_3_match_rating,
    f.rolling_20_match_rating,
    r.rated_appearances,
    r.rated_minutes,
    r.average_rating,
    r.rating_volatility,
    v.valuation_date as latest_valuation_date,
    v.value_eur as current_market_value_eur,
    e.scored_at as forecast_refreshed_at,
    e.model_version as valuation_model_version,
    e.estimate_eur as estimated_value_eur,
    e.lower_eur as estimated_lower_eur,
    e.upper_eur as estimated_upper_eur,
    e.predicted_pct_change,
    e.probability_value_increase,
    e.direction,
    e.confidence
from {{ ref('stg_players') }} p
left join {{ ref('int_current_player_form') }} f
    on f.player_id = p.player_id
left join {{ ref('int_player_rating_summary') }} r
    on r.player_id = p.player_id
   and r.rating_version = f.rating_version
left join {{ ref('int_latest_player_valuation') }} v
    on v.player_id = p.player_id
left join {{ ref('int_latest_valuation_estimate') }} e
    on e.player_id = p.player_id
