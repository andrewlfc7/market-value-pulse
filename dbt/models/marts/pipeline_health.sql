select
    run_id,
    pipeline,
    status,
    started_at,
    completed_at,
    extract(
        epoch from (completed_at - started_at)
    )::double precision as duration_seconds,
    counts,
    error,
    case
        when status = 'failed' then true
        when completed_at is null and status <> 'running' then true
        else false
    end as needs_attention
from {{ ref('stg_pipeline_runs') }}
