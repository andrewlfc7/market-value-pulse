select *
from {{ ref('player_outlook') }}
where estimated_value_eur is not null
  and (
      estimated_lower_eur > estimated_value_eur
      or estimated_value_eur > estimated_upper_eur
  )
