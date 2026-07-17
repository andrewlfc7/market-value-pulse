select
    run_id::text as run_id,
    pipeline::text as pipeline,
    status::text as status,
    started_at::timestamptz as started_at,
    completed_at::timestamptz as completed_at,
    counts,
    error::text as error
from {{ source('app', 'pipeline_runs') }}
