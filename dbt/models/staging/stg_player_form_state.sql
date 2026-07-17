select
    player_id::bigint as player_id,
    rating_version::text as rating_version,
    last_match_datetime::timestamptz as last_match_datetime,
    form_rating_ewm::double precision as form_rating_ewm,
    rolling_3_match_rating::double precision
        as rolling_3_match_rating,
    rolling_20_match_rating::double precision
        as rolling_20_match_rating,
    updated_at::timestamptz as updated_at
from {{ source('app', 'player_form_state') }}
