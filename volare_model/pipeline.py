from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEPARTURE_ORDER = {
    "Before 6 AM": 0,
    "6 AM - 12 PM": 1,
    "12 PM - 6 PM": 2,
    "After 6 PM": 3,
}
ARRIVAL_ORDER = DEPARTURE_ORDER.copy()
JOURNEY_DAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
SEASON_MAP = {
    1: "winter",
    2: "winter",
    3: "summer",
    4: "summer",
    5: "summer",
    6: "monsoon",
    7: "monsoon",
    8: "monsoon",
    9: "shoulder",
    10: "festive",
    11: "festive",
    12: "winter",
}
PEAK_SEASON_LABELS = {"summer", "festive"}
INDIAN_HOLIDAYS_MD = {
    (1, 26),   # Republic Day
    (3, 8),    # Holi 2023
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (11, 12),  # Diwali 2023
    (12, 25),  # Christmas
}


def _parse_total_stops(value: Any) -> float | np.nan:
    if pd.isna(value):
        return np.nan
    value = str(value).lower().strip()
    if "non" in value:
        return 0.0
    digits = "".join(ch for ch in value if ch.isdigit())
    return float(digits) if digits else np.nan


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate domain-specific features for the fare estimation problem."""

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        df_feat = X.copy()
        dates = pd.to_datetime(df_feat.get("Date_of_journey"), errors="coerce")
        df_feat["month"] = dates.dt.month
        df_feat["journey_weekday"] = dates.dt.weekday
        df_feat["is_weekend"] = (dates.dt.weekday >= 5).astype(float)

        journey_day_series = (
            df_feat["Journey_day"] if "Journey_day" in df_feat else pd.Series(index=df_feat.index, dtype=object)
        )
        weekday_from_text = journey_day_series.map(JOURNEY_DAY_MAP)
        df_feat["journey_weekday"] = df_feat["journey_weekday"].fillna(weekday_from_text)
        df_feat["is_weekend"] = df_feat["is_weekend"].fillna((weekday_from_text >= 5).astype(float))

        df_feat["departure_code"] = df_feat.get("Departure").map(DEPARTURE_ORDER)
        df_feat["arrival_code"] = df_feat.get("Arrival").map(ARRIVAL_ORDER)

        df_feat["total_stops_num"] = df_feat.get("Total_stops").apply(_parse_total_stops)
        df_feat["is_nonstop"] = (df_feat["total_stops_num"] == 0).astype(int)

        df_feat["Duration_in_hours"] = pd.to_numeric(df_feat.get("Duration_in_hours"), errors="coerce")
        df_feat["Days_left"] = pd.to_numeric(df_feat.get("Days_left"), errors="coerce")
        df_feat["duration_per_stop"] = df_feat["Duration_in_hours"] / (df_feat["total_stops_num"] + 1)

        df_feat["is_last_minute"] = (df_feat["Days_left"] <= 3).astype(int)
        df_feat["is_early_booking"] = (df_feat["Days_left"] >= 60).astype(int)

        source_series = df_feat["Source"] if "Source" in df_feat else pd.Series(index=df_feat.index, dtype=object)
        dest_series = df_feat["Destination"] if "Destination" in df_feat else pd.Series(index=df_feat.index, dtype=object)
        class_series = df_feat["Class"] if "Class" in df_feat else pd.Series(index=df_feat.index, dtype=object)

        df_feat["route"] = (
            source_series.fillna("Desconocido")
            + "-"
            + dest_series.fillna("Desconocido")
        )
        df_feat["route_class"] = df_feat["route"] + "::" + class_series.fillna("Desconocido")

        df_feat["season"] = df_feat["month"].map(SEASON_MAP).fillna("unknown")
        df_feat["season_code"] = df_feat["season"].astype("category").cat.codes
        df_feat["is_peak_season"] = df_feat["season"].isin(PEAK_SEASON_LABELS).astype(int)

        df_feat["is_holiday"] = dates.apply(
            lambda dt: int((dt.month, dt.day) in INDIAN_HOLIDAYS_MD) if pd.notna(dt) else 0
        )

        return df_feat


NUMERIC_FEATURES = [
    "Duration_in_hours",
    "Days_left",
    "month",
    "journey_weekday",
    "departure_code",
    "arrival_code",
    "total_stops_num",
    "duration_per_stop",
    "is_weekend",
    "is_nonstop",
    "is_last_minute",
    "is_early_booking",
    "season_code",
    "is_peak_season",
    "is_holiday",
]

CAT_FEATURES = ["Airline", "Class", "Source", "Destination", "route", "route_class", "season"]


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CAT_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("feature_gen", FeatureGenerator()),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


__all__ = [
    "FeatureGenerator",
    "build_preprocessor",
    "build_pipeline",
    "NUMERIC_FEATURES",
    "CAT_FEATURES",
]

