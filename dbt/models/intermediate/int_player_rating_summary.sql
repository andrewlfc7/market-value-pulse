select
    player_id,
    rating_version,
    count(*) as rated_appearances,
    sum(minutes) as rated_minutes,
    avg(rating) as average_rating,
    stddev_pop(rating) as rating_volatility,
    avg(threat_component) as threat_component_average,
    avg(creation_component) as creation_component_average,
    avg(progression_component) as progression_component_average,
    avg(retention_component) as retention_component_average,
    avg(attacking_xpv_component) as attacking_xpv_component_average,
    avg(defensive_component) as defensive_component_average,
    avg(finishing_component) as finishing_component_average,
    avg(goalkeeper_component) as goalkeeper_component_average
from {{ ref('stg_player_match_ratings') }}
group by player_id, rating_version
