with ranked as (
    select
        *,
        row_number() over (
            partition by player_id
            order by updated_at desc, rating_version desc
        ) as row_number
    from {{ ref('stg_player_form_state') }}
)

select
    player_id,
    rating_version,
    last_match_datetime,
    form_rating_ewm,
    rolling_3_match_rating,
    rolling_20_match_rating,
    updated_at
from ranked
where row_number = 1
