from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesianFitConfig:
    num_warmup: int = 1_000
    num_samples: int = 1_000
    num_chains: int = 2
    target_accept_probability: float = 0.93
    max_tree_depth: int = 10
    random_seed: int = 42


@dataclass(frozen=True)
class BayesianFitResult:
    mcmc: Any
    posterior_samples: dict[str, np.ndarray]
    mean_predictions: np.ndarray
    median_predictions: np.ndarray
    expected_samples: np.ndarray
    predictive_lower_90: np.ndarray
    predictive_upper_90: np.ndarray
    global_coefficients: pd.DataFrame
    position_effects: pd.DataFrame
    player_effects: pd.DataFrame
    diagnostics: dict[str, Any]


def position_hierarchical_model(
    X,
    position_code,
    player_code,
    number_of_positions: int,
    number_of_players: int,
    age_index: int,
    form_index: int,
    y=None,
) -> None:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    number_of_features = X.shape[1]
    global_intercept = numpyro.sample("global_intercept", dist.Normal(0.0, 0.5))
    global_beta_scale = numpyro.sample("global_beta_scale", dist.HalfNormal(0.30))
    global_beta = numpyro.sample(
        "global_beta",
        dist.Normal(0.0, global_beta_scale).expand([number_of_features]),
    )

    position_intercept_scale = numpyro.sample(
        "position_intercept_scale", dist.HalfNormal(0.25)
    )
    position_intercept_raw = numpyro.sample(
        "position_intercept_raw",
        dist.Normal(0.0, 1.0).expand([number_of_positions]),
    )
    position_intercept = numpyro.deterministic(
        "position_intercept", position_intercept_raw * position_intercept_scale
    )

    position_age_scale = numpyro.sample("position_age_scale", dist.HalfNormal(0.15))
    position_age_raw = numpyro.sample(
        "position_age_raw", dist.Normal(0.0, 1.0).expand([number_of_positions])
    )
    position_age_effect = numpyro.deterministic(
        "position_age_effect", position_age_raw * position_age_scale
    )

    position_form_scale = numpyro.sample("position_form_scale", dist.HalfNormal(0.15))
    position_form_raw = numpyro.sample(
        "position_form_raw", dist.Normal(0.0, 1.0).expand([number_of_positions])
    )
    position_form_effect = numpyro.deterministic(
        "position_form_effect", position_form_raw * position_form_scale
    )

    player_intercept_scale = numpyro.sample(
        "player_intercept_scale", dist.HalfNormal(0.20)
    )
    player_intercept_raw = numpyro.sample(
        "player_intercept_raw",
        dist.Normal(0.0, 1.0).expand([number_of_players]),
    )
    player_intercept = numpyro.deterministic(
        "player_intercept", player_intercept_raw * player_intercept_scale
    )

    expected_change = (
        global_intercept
        + jnp.matmul(X, global_beta)
        + position_intercept[position_code]
        + player_intercept[player_code]
        + position_age_effect[position_code] * X[:, age_index]
        + position_form_effect[position_code] * X[:, form_index]
    )

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.40))
    nu_minus_two = numpyro.sample("nu_minus_two", dist.Exponential(0.10))
    nu = numpyro.deterministic("nu", nu_minus_two + 2.0)

    with numpyro.plate("observations", X.shape[0]):
        numpyro.sample(
            "observed_change",
            dist.StudentT(df=nu, loc=expected_change, scale=sigma),
            obs=y,
        )


def _reconstruct_effects(
    samples: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_intercept = (
        samples["position_intercept_raw"]
        * samples["position_intercept_scale"][:, None]
    )
    position_age = (
        samples["position_age_raw"] * samples["position_age_scale"][:, None]
    )
    position_form = (
        samples["position_form_raw"] * samples["position_form_scale"][:, None]
    )
    player_intercept = (
        samples["player_intercept_raw"]
        * samples["player_intercept_scale"][:, None]
    )
    return position_intercept, position_age, position_form, player_intercept


def _posterior_diagnostics(mcmc) -> dict[str, Any]:
    from numpyro.diagnostics import summary

    grouped = mcmc.get_samples(group_by_chain=True)
    selected = {
        key: value
        for key, value in grouped.items()
        if key
        in {
            "global_intercept",
            "global_beta",
            "global_beta_scale",
            "position_intercept_scale",
            "position_intercept_raw",
            "position_age_scale",
            "position_age_raw",
            "position_form_scale",
            "position_form_raw",
            "player_intercept_scale",
            "player_intercept_raw",
            "sigma",
            "nu_minus_two",
            "nu",
        }
    }
    diagnostic_summary = summary(selected, group_by_chain=True)
    r_hats = []
    effective_samples = []
    serializable: dict[str, Any] = {}
    for site, values in diagnostic_summary.items():
        row = {}
        for key, value in values.items():
            array = np.asarray(value)
            row[key] = array.tolist() if array.ndim else float(array)
            if key == "r_hat":
                r_hats.extend(array.reshape(-1).tolist())
            if key == "n_eff":
                effective_samples.extend(array.reshape(-1).tolist())
        serializable[site] = row

    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return {
        "divergences": divergences,
        "maximum_r_hat": float(np.nanmax(r_hats)) if r_hats else None,
        "minimum_effective_sample_size": (
            float(np.nanmin(effective_samples)) if effective_samples else None
        ),
        "posterior_summary": serializable,
    }


def fit_position_hierarchical_bayesian(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_position_code: np.ndarray,
    train_player_code: np.ndarray,
    X_test: np.ndarray,
    test_position_code: np.ndarray,
    test_player_code: np.ndarray,
    feature_names: list[str],
    position_levels: list[str],
    player_levels: list[int],
    age_feature: str = "age_at_valuation",
    form_feature: str = "recency_weighted_rating",
    config: BayesianFitConfig | None = None,
) -> BayesianFitResult:
    config = config or BayesianFitConfig()
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    age_index = feature_names.index(age_feature)
    form_index = feature_names.index(form_feature)

    kernel = NUTS(
        position_hierarchical_model,
        target_accept_prob=config.target_accept_probability,
        max_tree_depth=config.max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(config.random_seed),
        X=jnp.asarray(X_train, dtype=jnp.float32),
        position_code=jnp.asarray(train_position_code, dtype=jnp.int32),
        player_code=jnp.asarray(train_player_code, dtype=jnp.int32),
        number_of_positions=len(position_levels),
        number_of_players=len(player_levels),
        age_index=age_index,
        form_index=form_index,
        y=jnp.asarray(y_train, dtype=jnp.float32),
    )

    samples = {
        key: np.asarray(value)
        for key, value in mcmc.get_samples(group_by_chain=False).items()
    }
    position_intercept, position_age, position_form, player_intercept = (
        _reconstruct_effects(samples)
    )
    seen_player = np.asarray(test_player_code) < len(player_levels)
    safe_player_code = np.clip(
        np.asarray(test_player_code), 0, max(len(player_levels) - 1, 0)
    )
    test_player_effect = (
        player_intercept[:, safe_player_code] * seen_player[None, :]
    )
    expected_samples = (
        samples["global_intercept"][:, None]
        + samples["global_beta"] @ np.asarray(X_test).T
        + position_intercept[:, test_position_code]
        + test_player_effect
        + position_age[:, test_position_code]
        * np.asarray(X_test)[:, age_index][None, :]
        + position_form[:, test_position_code]
        * np.asarray(X_test)[:, form_index][None, :]
    )
    mean_predictions = expected_samples.mean(axis=0)
    median_predictions = np.median(expected_samples, axis=0)

    generator = np.random.default_rng(config.random_seed + 1)
    noise = generator.standard_t(
        df=samples["nu"][:, None],
        size=(len(samples["nu"]), X_test.shape[0]),
    )
    predictive_samples = expected_samples + samples["sigma"][:, None] * noise
    lower = np.quantile(predictive_samples, 0.05, axis=0)
    upper = np.quantile(predictive_samples, 0.95, axis=0)

    beta = samples["global_beta"]
    global_coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "posterior_mean": beta.mean(axis=0),
            "posterior_sd": beta.std(axis=0),
            "credible_lower_90": np.quantile(beta, 0.05, axis=0),
            "credible_upper_90": np.quantile(beta, 0.95, axis=0),
            "probability_positive": (beta > 0).mean(axis=0),
        }
    )
    global_coefficients["absolute_posterior_mean"] = global_coefficients[
        "posterior_mean"
    ].abs()
    global_coefficients = global_coefficients.sort_values(
        "absolute_posterior_mean", ascending=False
    )

    position_rows = []
    for index, position in enumerate(position_levels):
        for effect_name, values in [
            ("intercept", position_intercept[:, index]),
            ("age_slope_adjustment", position_age[:, index]),
            ("form_slope_adjustment", position_form[:, index]),
        ]:
            position_rows.append(
                {
                    "position_group": position,
                    "effect": effect_name,
                    "posterior_mean": float(values.mean()),
                    "posterior_sd": float(values.std()),
                    "credible_lower_90": float(np.quantile(values, 0.05)),
                    "credible_upper_90": float(np.quantile(values, 0.95)),
                    "probability_positive": float((values > 0).mean()),
                }
            )
    position_effects = pd.DataFrame(position_rows)
    player_effects = pd.DataFrame(
        {
            "transfermarkt_player_id": player_levels,
            "posterior_mean": player_intercept.mean(axis=0),
            "posterior_sd": player_intercept.std(axis=0),
            "credible_lower_90": np.quantile(player_intercept, 0.05, axis=0),
            "credible_upper_90": np.quantile(player_intercept, 0.95, axis=0),
            "probability_positive": (player_intercept > 0).mean(axis=0),
        }
    )

    return BayesianFitResult(
        mcmc=mcmc,
        posterior_samples=samples,
        mean_predictions=mean_predictions,
        median_predictions=median_predictions,
        expected_samples=expected_samples,
        predictive_lower_90=lower,
        predictive_upper_90=upper,
        global_coefficients=global_coefficients,
        position_effects=position_effects,
        player_effects=player_effects,
        diagnostics=_posterior_diagnostics(mcmc),
    )
