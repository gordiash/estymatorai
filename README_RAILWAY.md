# API Wyceny Nieruchomości - Random Forest

API do wyceny nieruchomości wykorzystujące model Random Forest.

## 🚀 Deployment na Railway

### Kroki deployment:

1. **Przygotuj repozytorium Git:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Random Forest API"
   ```

2. **Połącz z Railway:**
   - Zaloguj się na [railway.app](https://railway.app)
   - Utwórz nowy projekt
   - Połącz z repozytorium GitHub

3. **Railway automatycznie:**
   - Wykryje Python aplikację
   - Zainstaluje zależności z `requirements.txt`
   - Uruchomi aplikację na porcie z zmiennej `PORT`

### 📁 Struktura plików dla Railway:

```
├── app.py                 # Główna aplikacja Flask
├── Procfile              # Komenda startowa
├── railway.json          # Konfiguracja Railway
├── requirements.txt      # Zależności Python
├── .railwayignore       # Pliki do ignorowania
├── artifacts/           # Modele ML
│   ├── model_rf.joblib  # Model Random Forest
│   └── feature_schema.json
└── src/                 # Kod źródłowy (opcjonalny)
```

## 🔧 API Endpoints

### `GET /`
Informacje o API

### `GET /health`
Sprawdzenie stanu API i modelu

### `POST /predict`
Predykcja ceny mieszkania

**Request body:**
```json
{
  "city": "Toruń",
  "area": 62.0,
  "rooms": 3,
  "floor": 4,
  "total_floors": 10,
  "has_elevator": true,
  "has_balcony": true,
  "building_type": "blok",
  "heating_type": "miejskie",
  "standard_of_finish": "do wykończenia"
}
```

**Response:**
```json
{
  "predicted_price": 537033.0,
  "price_per_sqm": 8662.0,
  "area": 62.0,
  "city": "Toruń",
  "rooms": 3,
  "floor": 4,
  "total_floors": 10,
  "has_elevator": true,
  "success": true
}
```

## 🧪 Testowanie lokalnie

```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom aplikację
python app.py

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Toruń", "area": 62, "rooms": 3, "floor": 4, "total_floors": 10}'
```

## 📊 Model

Model Random Forest został wytrenowany na danych nieruchomości i zawiera:
- Cechy podstawowe (powierzchnia, pokoje, piętro)
- Cechy budynku (typ, ogrzewanie, wyposażenie)
- Cechy geograficzne (lokalizacja, odległości)
- Cechy czasowe (data publikacji)

Model jest automatycznie ładowany przy starcie aplikacji z pliku `artifacts/model_rf.joblib`.
