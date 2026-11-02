from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, last_days: int, date_col: str = "event_date"):
    """Split off the last `last_days` calendar days.

    A random split would let the model learn from impressions that happen *after* the ones
    it is evaluated on, which overstates offline accuracy for a model that will always
    predict the future in production.
    """
    if last_days < 1:
        raise ValueError("last_days must be >= 1")
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max().normalize() - pd.Timedelta(days=last_days - 1)
    before, after = df[dates < cutoff], df[dates >= cutoff]
    if before.empty or after.empty:
        raise ValueError(
            f"time split with last_days={last_days} leaves an empty side "
            f"({dates.min().date()} .. {dates.max().date()})"
        )
    return before.reset_index(drop=True), after.reset_index(drop=True)
