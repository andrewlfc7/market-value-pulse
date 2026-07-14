#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 OLD_REPOSITORY NEW_REPOSITORY [FEATURE_MODEL_SOURCE]" >&2
  exit 2
fi

old_repo="$(cd "$1" && pwd)"
mkdir -p "$2"
new_repo="$(cd "$2" && pwd)"
model_source="${3:-$old_repo/models/features}"

for layer in raw normalized; do
  source_path="$old_repo/data/$layer"
  if [[ -d "$source_path" ]]; then
    mkdir -p "$new_repo/data/$layer"
    rsync -a "$source_path/" "$new_repo/data/$layer/"
  fi
done

if [[ -d "$model_source" ]]; then
  mkdir -p "$new_repo/models/features"
  missing_models=0
  for family in goal_probability xa xg xgot xthreat; do
    if [[ -d "$model_source/$family" ]]; then
      mkdir -p "$new_repo/models/features/$family"
      rsync -a "$model_source/$family/" "$new_repo/models/features/$family/"
    else
      echo "Required feature model family not found: $model_source/$family" >&2
      missing_models=1
    fi
  done
  if [[ "$missing_models" -ne 0 ]]; then
    exit 1
  fi
else
  echo "Feature model source not found: $model_source" >&2
  exit 1
fi

echo "Reused raw + normalized data and the five required feature-model families."
echo "Generated features/state/serving/replays were intentionally not copied."
