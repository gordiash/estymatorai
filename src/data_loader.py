from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Tuple
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from .utils.geo import NearestGeoJoin


def LoadTables(engine: Engine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Pobierz wszystkie kolumny, aby uniknąć błędów nazw
    nier_query = text(
        """
        SELECT *
        FROM nieruchomosci
        WHERE price IS NOT NULL AND area IS NOT NULL AND area > 5
        """
    )

    geo_query = text(
        """
        SELECT *
        FROM geographic_features
        """
    )



    nier = pd.read_sql(nier_query, con=engine)
    geo = pd.read_sql(geo_query, con=engine)

    # Typy
    if "listing_date" in nier.columns:
        nier["listing_date"] = pd.to_datetime(nier["listing_date"], errors="coerce")

    # Zapewnij klucz miasta i współrzędne po lewej
    if "city" not in nier.columns:
        nier["city"] = nier.get("district")
    # Normalizacja współrzędnych w danych lewych
    if "latitude" not in nier.columns and "lat" in nier.columns:
        nier["latitude"] = nier["lat"]
    if "longitude" not in nier.columns and "lon" in nier.columns:
        nier["longitude"] = nier["lon"]

    # Ujednolicenie kolumn geo: prefiksuj wszystkie kolumny gf_, aby uniknąć kolizji nazw
    geo = geo.rename(columns={c: (c if c.startswith("gf_") else f"gf_{c}") for c in geo.columns})

    # Join po mieście (lewy), jeśli mamy kolumny miast
    if "city" in nier.columns and "gf_city" in geo.columns:
        merged = nier.merge(
            geo,
            left_on="city",
            right_on="gf_city",
            how="left",
            suffixes=("", ""),
        )
    else:
        merged = nier.copy()

    # Fallback: NN po współrzędnych tam, gdzie brak dopasowania, tylko jeśli mamy kolumny współrzędnych
    has_left_xy = {"latitude", "longitude"}.issubset(merged.columns)
    has_right_xy = {"gf_lat", "gf_lon"}.issubset(geo.columns)
    if has_left_xy and has_right_xy:
        if "gf_lat" in merged.columns:
            need_nn_mask = merged["gf_lat"].isna()
        else:
            need_nn_mask = pd.Series(False, index=merged.index)

        if need_nn_mask.any():
            left_nn = merged.loc[need_nn_mask, [c for c in merged.columns if not c.startswith("gf_")]].copy()
            right_cols = [c for c in geo.columns if c.startswith("gf_")]

            nn_joined = NearestGeoJoin(
                left_nn,
                geo,
                left_lat="latitude",
                left_lon="longitude",
                right_lat="gf_lat",
                right_lon="gf_lon",
                right_cols=right_cols,
            )
            # Zapewnij zgodne typy kolumn wynikowych przed przypisaniem
            for col in right_cols:
                if col not in merged.columns:
                    rdt = geo[col].dtype if col in geo.columns else None
                    if rdt is not None and is_numeric_dtype(rdt):
                        merged[col] = np.nan
                    elif rdt is not None and is_datetime64_any_dtype(rdt):
                        merged[col] = pd.NaT
                    else:
                        merged[col] = pd.Series([pd.NA] * len(merged), dtype="object")
                else:
                    if col in geo.columns:
                        rdt = geo[col].dtype
                        if is_datetime64_any_dtype(rdt):
                            merged[col] = pd.to_datetime(merged[col], errors="coerce")
                        elif not is_numeric_dtype(rdt) and str(merged[col].dtype) != "object":
                            merged[col] = merged[col].astype("object")

            for col in right_cols + ["geo_nn_distance_m"]:
                merged.loc[need_nn_mask, col] = nn_joined[col].values

    return merged, geo
