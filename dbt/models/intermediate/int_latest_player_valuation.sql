with ranked as (
    select
        *,
        row_number() over (
            partition by player_id
            order by valuation_date desc, source
        ) as row_number
    from {{ ref('stg_market_valuations') }}
)

select
    player_id,
    valuation_date,
    value_eur,
    source
from ranked
where row_number = 1
