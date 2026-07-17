select
    player_id::bigint as player_id,
    match_id::bigint as match_id,
    rating_version::text as rating_version,
    rating::double precision as rating,
    minutes::double precision as minutes,
    features,
    nullif(features ->> 'threat_component', '')::double precision
        as threat_component,
    nullif(features ->> 'creation_component', '')::double precision
        as creation_component,
    nullif(features ->> 'progression_component', '')::double precision
        as progression_component,
    nullif(features ->> 'retention_component', '')::double precision
        as retention_component,
    nullif(features ->> 'attacking_xpv_component', '')::double precision
        as attacking_xpv_component,
    nullif(features ->> 'defensive_component', '')::double precision
        as defensive_component,
    nullif(features ->> 'finishing_component', '')::double precision
        as finishing_component,
    nullif(features ->> 'goalkeeper_component', '')::double precision
        as goalkeeper_component
from {{ source('app', 'player_match_ratings') }}
