select
    model_version,
    count(*) as active_predictions,
    min(scored_at) as earliest_score_at,
    max(scored_at) as latest_score_at,
    avg(predicted_pct_change) as average_predicted_pct_change,
    avg(probability_value_increase) as average_probability_increase,
    avg(confidence) as average_confidence,
    avg(upper_eur - lower_eur) as average_interval_width_eur,
    percentile_cont(0.5) within group (
        order by upper_eur - lower_eur
    ) as median_interval_width_eur
from {{ ref('stg_valuation_estimates') }}
group by model_version
