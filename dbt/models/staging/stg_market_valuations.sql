select
    player_id::bigint as player_id,
    valuation_date::date as valuation_date,
    value_eur::bigint as value_eur,
    source::text as source
from {{ source('app', 'market_valuations') }}
