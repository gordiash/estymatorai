from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import shap
import json

from .config import CreateSqlAlchemyEngine
from .data_loader import LoadTables
from .feature_engineering import CleanAndEngineerFeatures
from .utils.reporting import ComputeMape, SaveJson
from .utils.paths import ToSafeName


def FormatPln(x: float) -> str:
    try:
        return f"{x:,.0f} zł".replace(",", " ")
    except Exception:
        return str(x)


def AlignToSchema(df: pd.DataFrame, schema_path: Path) -> pd.DataFrame:
    if not schema_path.exists():
        return df
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    cols = schema.get("feature_names", [])
    cat_cols = set(schema.get("categorical_names", []))
    # Usuń nadmiarowe
    aligned = df[[c for c in df.columns if c in cols]].copy()
    # Dodaj brakujące w odpowiedniej kolejności
    for c in cols:
        if c not in aligned.columns:
            aligned[c] = np.nan
    # Uporządkuj kolejność i typy
    aligned = aligned[cols]
    for c in aligned.columns:
        if c in cat_cols:
            aligned[c] = aligned[c].astype("string").fillna("Unknown")
    return aligned


def _load_global_model(artifacts: Path) -> Pipeline | None:
    model_path = artifacts / "model_rf.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path.as_posix())


def _load_city_model(artifacts: Path, city: str) -> tuple[Pipeline | None, Path | None]:
    safe = ToSafeName(city)
    city_dir = artifacts / "per_city" / safe
    model_path = city_dir / "model_rf.joblib"
    if not model_path.exists():
        return None, None
    return joblib.load(model_path.as_posix()), city_dir


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Liczba rekordów do testu")
    args = parser.parse_args()

    artifacts = Path("artifacts")

    # Wczytaj dane i przygotuj cechy
    engine = CreateSqlAlchemyEngine()
    raw_df, _ = LoadTables(engine)
    Xy, cat_cols, target_col = CleanAndEngineerFeatures(raw_df)
    sample = Xy.sample(n=min(args.n, len(Xy)), random_state=123).reset_index(drop=True)

    # Dla inferencji per-wiersz: jeśli mamy kolumnę miasta, spróbuj użyć dedykowanego modelu
    # W przypadku braku miasta lub modelu – fallback do modelu globalnego
    global_model = _load_global_model(artifacts)
    overall_rows = []
    preds = np.zeros(len(sample), dtype=float)
    if "city" in sample.columns:
        for city, sdf in sample.groupby(sample["city"].astype(str)):
            city_model, city_dir = _load_city_model(artifacts, city)
            if city_model is None:
                city_model = global_model
                schema_path = artifacts / "feature_schema.json"
            else:
                schema_path = (city_dir or artifacts) / "feature_schema.json"
            Xc = sdf.drop(columns=[target_col])
            # Random Forest nie potrzebuje AlignToSchema - pipeline sam obsługuje preprocessing
            pred_c = city_model.predict(Xc)
            preds[sdf.index.values] = pred_c
            overall_rows.append({"city": str(city), "rows": int(len(sdf))})
    else:
        if global_model is None:
            raise SystemExit("Brak artifacts/model_rf.joblib ani modeli per-miasto – najpierw uruchom trening.")
        X = sample.drop(columns=[target_col])
        # Random Forest nie potrzebuje AlignToSchema - pipeline sam obsługuje preprocessing
        preds = global_model.predict(X)
        overall_rows.append({"city": "GLOBAL", "rows": int(len(sample))})
    y = sample[target_col].values.astype(float)

    # Model zwraca log-cenę → przelicz na zł
    pred = preds

    abs_err = np.abs(y - pred)
    pct_err = np.abs((y - pred) / np.clip(y, 1e-6, None))

    mae = float(mean_absolute_error(y, pred))
    mape = float(ComputeMape(y, pred))
    medae = float(np.median(abs_err))
    p90 = float(np.quantile(abs_err, 0.90))
    p95 = float(np.quantile(abs_err, 0.95))

    # Lokalny SHAP dla największego błędu (na log-skali, informacyjnie)
    try:
        explainer = shap.TreeExplainer(model)
        worst_idx = int(abs_err.argmax())
        shap_vals = explainer.shap_values(X.iloc[[worst_idx]])
        contrib = shap_vals[0]
        names = X.columns
        order = np.argsort(-np.abs(contrib))[:8]
        top_local = [(names[i], float(contrib[i])) for i in order]
    except Exception:
        top_local = []

    # Tabela wyników (pierwsze 10)
    out_df = pd.DataFrame({
        "y_true": y,
        "y_pred": pred,
        "abs_err": abs_err,
        "pct_err": pct_err,
    })
    show = out_df.head(10).copy()
    show["y_true"] = show["y_true"].round(0)
    show["y_pred"] = show["y_pred"].round(0)
    show["abs_err"] = show["abs_err"].round(0)
    show["pct_err"] = (show["pct_err"] * 100).round(2)

    # Raport tekstowy
    print("\n=== Raport testu inferencji ===")
    print(f"Próbka: {len(sample)} rekordów")
    print(f"MAE: {FormatPln(mae)}  |  MedAE: {FormatPln(medae)}  |  P90 abs err: {FormatPln(p90)}  |  P95 abs err: {FormatPln(p95)}")
    print(f"MAPE: {mape*100:.2f}%")

    print("\nPrzykładowe predykcje (pierwsze 10):")
    print(show.rename(columns={"y_true": "CenaRzeczywista", "y_pred": "Predykcja", "abs_err": "BladBezwzgledny", "pct_err": "BladProc[%]"}).to_string(index=False))

    if top_local:
        print("\nWyjaśnienia (SHAP) dla rekordu o największym błędzie (na log-skali):")
        for name, val in top_local:
            sign = "+" if val >= 0 else "-"
            print(f"- {name}: {sign}{abs(val):,.3f}")

    # Zapis JSON
    report = {
        "SampleRows": int(len(sample)),
        "MAE": mae,
        "MAPE": mape,
        "MedAE": medae,
        "P90AbsError": p90,
        "P95AbsError": p95,
        "TopLocalShap": top_local,
        "Groups": overall_rows,
    }
    SaveJson(report, artifacts / "test_report.json")

    print("\nSzczegóły zapisano do artifacts/test_report.json")


if __name__ == "__main__":
    Main()
