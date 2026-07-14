# Feature inference artifacts

Place the five versioned inference families here:

```text
models/features/
├── goal_probability/xpv_action_v1/{metadata.json,model.json}
├── xa/xa_action_v1/{metadata.json,model.joblib}
├── xg/xg_shot_v1/{metadata.json,metadata.joblib,model.joblib}
├── xgot/xgot_shot_v1/{metadata.json,metadata.joblib,model.joblib}
└── xthreat/xt_action_v1/{metadata.json,model.json}
```

Each `metadata.json` names its primary artifact through `model_file`. The xG
and xGOT directories also contain their fitted feature metadata. Model files
are read only by the Parquet-native feature engine and are loaded once per
process. No external feature database or model-training repository is needed.
