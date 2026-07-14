from __future__ import annotations

import polars as pl


class RatingValidationError(RuntimeError):
    """Raised when a rating dataset violates an invariant."""


def validate_rating_frame(frame: pl.DataFrame) -> None:
    required = {
        "season",
        "match_id",
        "whoscored_player_id",
        "post_match_rating",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RatingValidationError(f"Missing rating columns: {missing}")

    duplicates = (
        frame.group_by(["season", "match_id", "whoscored_player_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicates:
        raise RatingValidationError(f"Found {duplicates} duplicate player-match keys")

    invalid = frame.filter(
        pl.col("post_match_rating").is_not_null()
        & ~pl.col("post_match_rating").is_between(1.0, 10.0, closed="both")
    ).height
    if invalid:
        raise RatingValidationError(f"Found {invalid} ratings outside [1, 10]")
