select
    player_id::bigint as player_id,
    scored_at::timestamptz as scored_at,
    model_version::text as model_version,
    estimate_eur::bigint as estimate_eur,
    lower_eur::bigint as lower_eur,
    upper_eur::bigint as upper_eur,
    direction::text as direction,
    confidence::double precision as confidence,
    predicted_pct_change::double precision as predicted_pct_change,
    probability_value_increase::double precision
        as probability_value_increase
from {{ source('app', 'valuation_estimates') }}
