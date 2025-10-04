# 🚀 Instrukcje Deployment na Railway

## Przygotowanie do deployment

### 1. Sprawdź pliki
Upewnij się, że masz następujące pliki:
- ✅ `app.py` - aplikacja Flask
- ✅ `Procfile` - komenda startowa
- ✅ `railway.json` - konfiguracja Railway
- ✅ `requirements.txt` - zależności Python
- ✅ `.railwayignore` - pliki do ignorowania
- ✅ `artifacts/model_rf.joblib` - model Random Forest

### 2. Test lokalny
```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom aplikację
python app.py

# W drugim terminalu przetestuj API
python test_api.py
```

## Deployment na Railway

### Opcja 1: Railway CLI
```bash
# Zainstaluj Railway CLI
npm install -g @railway/cli

# Zaloguj się
railway login

# Utwórz projekt
railway init

# Deploy
railway up
```

### Opcja 2: Railway Dashboard
1. Przejdź na [railway.app](https://railway.app)
2. Zaloguj się
3. Kliknij "New Project"
4. Wybierz "Deploy from GitHub repo"
5. Połącz swoje repozytorium
6. Railway automatycznie wykryje Python i wdroży aplikację

### Opcja 3: GitHub Integration
1. Wypchnij kod do GitHub:
   ```bash
   git init
   git add .
   git commit -m "Random Forest API for Railway"
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

2. W Railway Dashboard:
   - Kliknij "New Project"
   - Wybierz "Deploy from GitHub repo"
   - Wybierz swoje repozytorium
   - Railway automatycznie wdroży aplikację

## 🔧 Konfiguracja Railway

### Zmienne środowiskowe (opcjonalne)
W Railway Dashboard możesz ustawić:
- `PORT` - port aplikacji (Railway automatycznie ustawia)
- `FLASK_ENV` - środowisko Flask (production/development)

### Health Check
Railway automatycznie sprawdzi endpoint `/health` aby upewnić się, że aplikacja działa.

## 📊 Testowanie po deployment

Po wdrożeniu otrzymasz URL aplikacji (np. `https://your-app.railway.app`).

### Test API:
```bash
# Test health
curl https://your-app.railway.app/health

# Test predict
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Toruń", "area": 62, "rooms": 3, "floor": 4, "total_floors": 10}'
```

### Przykład odpowiedzi:
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

## 🐛 Troubleshooting

### Problem: Model nie został załadowany
- Sprawdź czy plik `artifacts/model_rf.joblib` istnieje
- Sprawdź logi Railway w Dashboard

### Problem: Błąd 500 przy predykcji
- Sprawdź czy wszystkie wymagane pola są w request
- Sprawdź logi aplikacji w Railway Dashboard

### Problem: Aplikacja nie startuje
- Sprawdź `requirements.txt` - wszystkie zależności muszą być zainstalowane
- Sprawdź `Procfile` - komenda musi być poprawna
- Sprawdź logi build w Railway Dashboard

## 📈 Monitoring

Railway Dashboard zapewnia:
- Logi aplikacji w czasie rzeczywistym
- Metryki użycia (CPU, RAM, Network)
- Historia deploymentów
- Automatyczne restartowanie przy błędach

## 🔄 Aktualizacja modelu

Aby zaktualizować model:
1. Wytrenuj nowy model lokalnie
2. Zastąp `artifacts/model_rf.joblib`
3. Wypchnij zmiany do GitHub
4. Railway automatycznie wdroży nową wersję
