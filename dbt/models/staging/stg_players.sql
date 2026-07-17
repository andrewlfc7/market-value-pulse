select
    player_id::bigint as player_id,
    display_name::text as display_name,
    birth_date::date as birth_date,
    position_group::text as position_group
from {{ source('app', 'players') }}
