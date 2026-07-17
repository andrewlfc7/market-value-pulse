with ranked as (
    select
        *,
        row_number() over (
            partition by player_id
            order by scored_at desc, model_version desc
        ) as row_number
    from {{ ref('stg_valuation_estimates') }}
)

select
    player_id,
    scored_at,
    model_version,
    estimate_eur,
    lower_eur,
    upper_eur,
    direction,
    confidence,
    predicted_pct_change,
    probability_value_increase
from ranked
where row_number = 1
