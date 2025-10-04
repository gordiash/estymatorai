from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List


def _safe_median(s: pd.Series, default: float = 0.0) -> float:
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return default
        return float(np.median(arr))
    except Exception:
        return default


def _safe_quantile(s: pd.Series, q: float, default: float | None = None) -> float:
    try:
        val = float(np.nanpercentile(s.values.astype(float), q * 100))
        if np.isnan(val):
            return default if default is not None else 0.0
        return val
    except Exception:
        return default if default is not None else 0.0


def CleanAndEngineerFeatures(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    """Czyści dane, tworzy cechy i zwraca (Xy, CatCols, TargetCol).
    Target: price.
    """
    data = df.copy()

    # Czyszczenie prostych anomalii
    data = data.drop_duplicates(subset=[c for c in ["listing_id", "url"] if c in data.columns])
    data = data[(data["price"] > 10_000) & (data["area"] > 5) & (data["area"] < 1000)]

    # Outliery cenowe per miasto (winsoryzacja bezpieczna)
    lower_g = data.groupby("city")["price"].transform(lambda s: _safe_quantile(s, 0.01)) if "city" in data.columns else None
    upper_g = data.groupby("city")["price"].transform(lambda s: _safe_quantile(s, 0.99)) if "city" in data.columns else None
    lower = lower_g if lower_g is not None else _safe_quantile(data["price"], 0.01)
    upper = upper_g if upper_g is not None else _safe_quantile(data["price"], 0.99)
    data["price"] = np.clip(data["price"], lower, upper)

    # Cechy podstawowe (bez wycieku z targetu)
    data["log_area"] = np.log1p(data["area"]) if "area" in data.columns else np.nan

    # Czas
    if "listing_date" in data.columns:
        data["listing_date"] = pd.to_datetime(data["listing_date"], errors="coerce")
        data["listing_year"] = data["listing_date"].dt.year
        data["listing_month"] = data["listing_date"].dt.month
        data["listing_quarter"] = data["listing_date"].dt.quarter
    else:
        data["listing_year"] = np.nan
        data["listing_month"] = np.nan
        data["listing_quarter"] = np.nan

    # Wiek budynku
    if "year_of_construction" in data.columns:
        current_year = pd.Timestamp.utcnow().year
        median_yoc = _safe_median(data["year_of_construction"], default=current_year)
        data["building_age"] = (data["listing_year"].fillna(current_year) - data["year_of_construction"].fillna(median_yoc)).astype(float)
    else:
        data["building_age"] = np.nan

    # Interakcje
    if "area" in data.columns:
        data["area_x_floor"] = data.get("area", np.nan) * data.get("floor", 0).fillna(0)
        if "rooms" in data.columns:
            data["sqm_per_room"] = data["area"] / data["rooms"].replace(0, np.nan)
            data["sqm_per_room"] = data["sqm_per_room"].fillna(data["sqm_per_room"].median())
    if "total_floors" in data.columns and "floor" in data.columns:
        data["floor_ratio"] = data["floor"].fillna(0) / data["total_floors"].replace(0, np.nan)
        data["floor_ratio"] = data["floor_ratio"].fillna(data["floor_ratio"].median())
    if "building_age" in data.columns:
        data["age_squared"] = (data["building_age"] ** 2).astype(float)

    # Prawdziwe kategorie do CatBoosta (string), bez flag 0/1
    cat_cols = [
        c for c in [
            "market", "district", "city", "province", "building_type", "heating_type", "standard_of_finish",
        ] if c in data.columns
    ]
    for c in cat_cols:
        data[c] = data[c].astype("string").fillna("Unknown")

    # Flagi 0/1 traktujemy jako numeryczne
    bool_like = [
        "has_balcony", "has_garage", "has_garden", "has_elevator", "has_basement",
        "has_separate_kitchen", "has_dishwasher", "has_fridge", "has_oven",
    ]
    for c in [b for b in bool_like if b in data.columns]:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0).astype(int)

    target_col = "price"

    # Kolumny ID/tekstowe luźne
    drop_cols = [c for c in [
        "ad_id", "listing_id", "url", "title_raw", "address_raw", "street",
        "security_features", "media_features",
        "source", "source_page", "source_position",
    ] if c in data.columns]

    # Imputacja liczb: mediana z bezpiecznym fallbackiem
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    for c in numeric_cols:
        med = _safe_median(data[c], default=0.0)
        data[c] = data[c].fillna(med)

    Xy = data.drop(columns=drop_cols, errors="ignore")
    Xy = Xy.dropna(subset=[target_col]).reset_index(drop=True)

    return Xy, cat_cols, target_col
