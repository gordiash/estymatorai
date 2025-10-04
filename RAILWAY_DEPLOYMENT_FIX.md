# 🚀 Railway Deployment - Instrukcje bez dużych plików

## ⚠️ Problem z dużymi plikami

Model `model_rf.joblib` ma 358 MB, co przekracza limit GitHub (100 MB). 

## 🔧 Rozwiązania

### Opcja 1: Railway z zewnętrznym storage

1. **Przygotuj model na Railway:**
   - Wytrenuj model lokalnie
   - Prześlij model do Railway przez SSH lub webhook

2. **Użyj zmiennych środowiskowych:**
   ```bash
   # W Railway Dashboard ustaw:
   MODEL_URL=https://your-storage.com/model_rf.joblib
   ```

3. **Zaktualizuj app.py:**
   ```python
   def load_model():
       global model
       try:
           # Spróbuj załadować z URL
           model_url = os.environ.get('MODEL_URL')
           if model_url:
               import urllib.request
               urllib.request.urlretrieve(model_url, 'temp_model.joblib')
               model = joblib.load('temp_model.joblib')
               os.remove('temp_model.joblib')
           else:
               # Fallback do lokalnego pliku
               model_path = Path("artifacts/model_rf.joblib")
               if model_path.exists():
                   model = joblib.load(model_path.as_posix())
           return True
       except Exception as e:
           print(f"Błąd ładowania modelu: {e}")
           return False
   ```

### Opcja 2: Railway z Docker

1. **Stwórz Dockerfile:**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   # Skopiuj model (jeśli jest dostępny)
   COPY artifacts/model_rf.joblib artifacts/
   
   EXPOSE 5000
   
   CMD ["python", "app.py"]
   ```

2. **Deploy przez Docker:**
   - Railway automatycznie wykryje Dockerfile
   - Skopiuj model do kontenera podczas build

### Opcja 3: Railway z Git LFS

1. **Zainstaluj Git LFS:**
   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   ```

2. **Dodaj model:**
   ```bash
   git add artifacts/model_rf.joblib
   git commit -m "Add model with LFS"
   git push origin master
   ```

### Opcja 4: Railway z webhook (zalecane)

1. **Stwórz endpoint do upload modelu:**
   ```python
   @app.route('/upload-model', methods=['POST'])
   def upload_model():
       if 'model' not in request.files:
           return jsonify({"error": "Brak pliku modelu"}), 400
       
       file = request.files['model']
       if file.filename.endswith('.joblib'):
           file.save('artifacts/model_rf.joblib')
           return jsonify({"message": "Model załadowany"})
       return jsonify({"error": "Nieprawidłowy format"}), 400
   ```

2. **Po deploy wyślij model:**
   ```bash
   curl -X POST https://your-app.railway.app/upload-model \
     -F "model=@artifacts/model_rf.joblib"
   ```

## 🎯 Zalecane rozwiązanie

**Użyj Opcji 4 (webhook)** - najprostsze i najbardziej elastyczne:

1. Deploy aplikację bez modelu
2. Po deploy wyślij model przez API
3. Model zostanie zapisany na Railway

## 📋 Kroki deployment

1. **Deploy aplikację:**
   ```bash
   git push origin master
   ```

2. **W Railway Dashboard:**
   - Utwórz nowy projekt
   - Połącz z GitHub
   - Railway wdroży aplikację

3. **Wyślij model:**
   ```bash
   curl -X POST https://your-app.railway.app/upload-model \
     -F "model=@artifacts/model_rf.joblib"
   ```

4. **Test API:**
   ```bash
   curl https://your-app.railway.app/health
   ```

## 🔍 Sprawdzenie

Po deployment sprawdź:
- ✅ Aplikacja startuje (`/health`)
- ✅ Model został załadowany
- ✅ API odpowiada (`/predict`)

Model będzie dostępny na Railway bez problemów z limitami GitHub!
