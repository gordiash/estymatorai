from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


EARTH_RADIUS_METERS = 6371000.0


def HaversineDistanceMeters(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_METERS * c


def NearestGeoJoin(left: pd.DataFrame, right: pd.DataFrame, left_lat: str, left_lon: str, right_lat: str, right_lon: str, right_cols: list[str]) -> pd.DataFrame:
    """Łączy ramki danych po najbliższym sąsiedzie (1-NN) w metryce haversine.
    Zwraca kopię `left` poszerzoną o `right_cols` z prawej ramki w kolejności wierszy left.
    """
    # Przygotuj wynik z domyślnymi brakami
    result = left.copy()
    # Utwórz brakujące kolumny prawej tabeli z odpowiednimi typami
    for col in right_cols:
        if col not in result.columns:
            rdt = right[col].dtype if col in right.columns else None
            if rdt is not None and is_numeric_dtype(rdt):
                result[col] = np.nan
            elif rdt is not None and is_datetime64_any_dtype(rdt):
                result[col] = pd.NaT
            else:
                result[col] = pd.Series([pd.NA] * len(result), dtype="object")
    if "geo_nn_distance_m" not in result.columns:
        result["geo_nn_distance_m"] = np.nan

    if left.empty or right.empty:
        return result

    # Współrzędne i maski ważnych wierszy (bez NaN)
    left_xy = left[[left_lat, left_lon]].to_numpy(copy=False)
    right_xy = right[[right_lat, right_lon]].to_numpy(copy=False)
    left_valid = ~np.isnan(left_xy).any(axis=1)
    right_valid = ~np.isnan(right_xy).any(axis=1)

    if not left_valid.any() or not right_valid.any():
        return result

    left_rad = np.radians(left_xy[left_valid])
    right_rad = np.radians(right_xy[right_valid])

    tree = BallTree(right_rad, metric="haversine")
    dists, idxs = tree.query(left_rad, k=1)
    idxs = idxs.flatten()
    dists_m = dists.flatten() * EARTH_RADIUS_METERS

    right_selected = right[right_valid].iloc[idxs].reset_index(drop=True)

    valid_idx = left.index[left_valid]
    for col in right_cols:
        result.loc[valid_idx, col] = right_selected[col].values
    result.loc[valid_idx, "geo_nn_distance_m"] = dists_m

    return result
