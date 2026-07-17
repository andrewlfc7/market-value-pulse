select *
from {{ ref('stg_player_match_ratings') }}
where rating is not null
  and (rating < 1 or rating > 10)
