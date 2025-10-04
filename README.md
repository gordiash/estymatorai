### Trening wyceny nieruchomości

Wymagania: Python 3.10+.

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```
2. Skonfiguruj `.env` na podstawie `.env.example`.
3. Uruchom trening:
```bash
python -m src.train --test-days 60
```
Artefakty (model i raport JSON) pojawią się w `artifacts/`.

### Tryb per-miasto z GPU

Trenowanie oddzielnych modeli dla każdego miasta, z opcjonalnym użyciem GPU (CatBoost `task_type=GPU`).

Uruchomienie:
```bash
python -m src.train_per_city --test-days 60 --gpu
```

Parametry:
- `--gpu`: włącza `task_type=GPU` (wymaga zainstalowanego środowiska CUDA kompatybilnego z CatBoost)
- `--limit-cities N`: opcjonalny limit liczby miast do przetworzenia

Wyjście:
- katalogi `artifacts/per_city/<MiastoSafe>/` z `model.cbm`, `feature_schema.json`, `report.json`
- indeks `artifacts/per_city_index.json`

Inferencja testowa (`src/test_inference.py`) automatycznie używa modelu dedykowanego dla miasta (jeśli dostępny), a w przeciwnym razie spada do modelu globalnego z `artifacts/model.cbm`.