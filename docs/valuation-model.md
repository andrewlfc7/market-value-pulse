# Valuation model design

## Prediction target

For consecutive Transfermarkt valuation records:

```text
target = log(current value / previous value)
```

A row is retained when the interval is between 14 and 365 days and contains at least 180 WhoScored minutes.

## Recency

The notebook-aligned rating variables are:

1. `average_rating`: minute-weighted rating inside the valuation interval.
2. `recency_weighted_rating`: minutes multiplied by a 90-day exponential decay.
3. `rating_last_90_days`: minute-weighted rating in the final 90 days of the interval.
4. `recent_rating_trend`: last-90-day rating minus the interval average.
5. `rating_volatility`: within-interval rating standard deviation.

Every selected performance aggregate is strictly after the previous valuation and strictly before the target valuation.

## Hierarchy

The production model partially pools by position:

```text
global shrinkage regression
  + position intercept
  + position-specific age adjustment
  + position-specific form adjustment
  + partially pooled player intercept
```

Age remains continuous and includes a quadratic term. Player effects are learned only for training players; an unseen player receives a neutral zero effect at scoring time.

## Likelihood

The residual uses a Student-t distribution. The degrees of freedom are inferred rather than fixed. This makes the model robust to rare, very large valuation updates while retaining a conditional predictive distribution.

## Evaluation

The latest 20% of unique valuation dates form a chronological holdout. Outputs include:

- MAE and approximate percentage MAE
- RMSE and R²
- direction accuracy
- Spearman rank correlation
- mean error
- 90% posterior predictive coverage
- calibration by predicted decile
- OLS HC3 comparison
- zero-change and previous-change baselines

Promotion is gated: the hierarchical candidate must beat both zero-change MAE/direction and OLS MAE, have positive holdout R², achieve at least 80% coverage, and pass divergence/R-hat checks. A failed candidate never replaces the active model.

## Updating

- New matches: rebuild current features and score with the existing posterior.
- New valuations: rebuild labeled intervals, retrain, compare, and promote a new immutable model version.
