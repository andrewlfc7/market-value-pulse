# Rating model artifacts

`uv run mvp ratings fit --competition <competition> --season <season>` creates a versioned artifact directory such as:

```text
post_match_v2/
├── rating_model_config.json
├── feature_schema.json
├── zscore_stats.parquet
└── pass_completion_priors.parquet
```

These files are fitted from historical player-match feature partitions and versioned independently from the event-feature artifacts.
