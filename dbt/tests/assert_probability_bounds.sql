select *
from {{ ref('player_outlook') }}
where probability_value_increase is not null
  and (
      probability_value_increase < 0
      or probability_value_increase > 1
  )
