from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from typing import Dict, Any
import tempfile

app = Flask(__name__)

# Globalna zmienna dla modelu
model = None

def load_model():
    """Ładuje model Random Forest"""
    global model
    try:
        # Sprawdź czy model został przesłany przez API
        model_path = Path("artifacts/model_rf.joblib")
        if model_path.exists():
            model = joblib.load(model_path.as_posix())
            print("Model Random Forest załadowany pomyślnie")
            return True
        else:
            print("Brak pliku modelu! Wyślij model przez /upload-model")
            return False
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        return False

def prepare_apartment_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Przygotowuje dane mieszkania do predykcji"""
    
    # Podstawowe dane mieszkania
    apartment_data = {
        # Podstawowe informacje
        'city': data.get('city', 'Toruń'),
        'area': float(data.get('area', 62.0)),
        'rooms': int(data.get('rooms', 3)),
        'floor': int(data.get('floor', 4)),
        'total_floors': int(data.get('total_floors', 10)),
        
        # Cechy budynku
        'building_type': data.get('building_type', 'blok'),
        'year_of_construction': int(data.get('year_of_construction', 2010)),
        'heating_type': data.get('heating_type', 'miejskie'),
        'standard_of_finish': data.get('standard_of_finish', 'do wykończenia'),
        
        # Flagi wyposażenia
        'has_balcony': int(data.get('has_balcony', 1)),
        'has_garage': int(data.get('has_garage', 0)),
        'has_garden': int(data.get('has_garden', 0)),
        'has_elevator': int(data.get('has_elevator', 1)),
        'has_basement': int(data.get('has_basement', 0)),
        'has_separate_kitchen': int(data.get('has_separate_kitchen', 1)),
        'has_dishwasher': int(data.get('has_dishwasher', 0)),
        'has_fridge': int(data.get('has_fridge', 0)),
        'has_oven': int(data.get('has_oven', 0)),
        
        # Dodatkowe cechy
        'market': data.get('market', 'wtórny'),
        'district': data.get('district', 'Centrum'),
        'province': data.get('province', 'kujawsko-pomorskie'),
        
        # Cechy czasowe
        'listing_year': int(data.get('listing_year', 2024)),
        'listing_month': int(data.get('listing_month', 10)),
        'listing_quarter': int(data.get('listing_quarter', 4)),
        
        # Cechy geograficzne (domyślne wartości dla Torunia)
        'latitude': float(data.get('latitude', 53.0138)),
        'longitude': float(data.get('longitude', 18.5984)),
        'distance_to_city_center': float(data.get('distance_to_city_center', 2.5)),
        'distance_to_nearest_school': float(data.get('distance_to_nearest_school', 0.5)),
        'distance_to_nearest_kindergarten': float(data.get('distance_to_nearest_kindergarten', 0.3)),
        'distance_to_nearest_supermarket': float(data.get('distance_to_nearest_supermarket', 0.2)),
        'distance_to_nearest_public_transport': float(data.get('distance_to_nearest_public_transport', 0.1)),
        'distance_to_university': float(data.get('distance_to_university', 1.5)),
        'distance_to_nearest_lake': float(data.get('distance_to_nearest_lake', 5.0)),
        
        # Cechy geograficzne gf_ (domyślne wartości)
        'gf_lat': float(data.get('gf_lat', 53.0138)),
        'gf_lon': float(data.get('gf_lon', 18.5984)),
        'gf_city': data.get('gf_city', 'Toruń'),
        'gf_population': float(data.get('gf_population', 200000)),
        'gf_urbanization_score': float(data.get('gf_urbanization_score', 0.7)),
        'gf_transport_accessibility': float(data.get('gf_transport_accessibility', 0.8)),
        'gf_healthcare_accessibility': float(data.get('gf_healthcare_accessibility', 0.6)),
        'gf_shopping_accessibility': float(data.get('gf_shopping_accessibility', 0.7)),
        'gf_recreation_score': float(data.get('gf_recreation_score', 0.6)),
        
        # Liczniki obiektów w promieniu
        'gf_count_szkola_in_5km': int(data.get('gf_count_szkola_in_5km', 5)),
        'gf_count_szkola_in_10km': int(data.get('gf_count_szkola_in_10km', 15)),
        'gf_count_szpital_in_5km': int(data.get('gf_count_szpital_in_5km', 2)),
        'gf_count_szpital_in_10km': int(data.get('gf_count_szpital_in_10km', 5)),
        'gf_count_park_in_5km': int(data.get('gf_count_park_in_5km', 3)),
        'gf_count_park_in_10km': int(data.get('gf_count_park_in_10km', 8)),
        'gf_count_centrum_handlowe_in_5km': int(data.get('gf_count_centrum_handlowe_in_5km', 2)),
        'gf_count_centrum_handlowe_in_10km': int(data.get('gf_count_centrum_handlowe_in_10km', 5)),
        'gf_count_dworzec_in_5km': int(data.get('gf_count_dworzec_in_5km', 1)),
        'gf_count_dworzec_in_10km': int(data.get('gf_count_dworzec_in_10km', 2)),
        'gf_count_muzeum_in_5km': int(data.get('gf_count_muzeum_in_5km', 2)),
        'gf_count_muzeum_in_10km': int(data.get('gf_count_muzeum_in_10km', 4)),
        'gf_count_teatr_in_5km': int(data.get('gf_count_teatr_in_5km', 1)),
        'gf_count_teatr_in_10km': int(data.get('gf_count_teatr_in_10km', 2)),
        'gf_count_hotel_in_5km': int(data.get('gf_count_hotel_in_5km', 3)),
        'gf_count_hotel_in_10km': int(data.get('gf_count_hotel_in_10km', 8)),
        'gf_count_rzeka_in_5km': int(data.get('gf_count_rzeka_in_5km', 1)),
        'gf_count_rzeka_in_10km': int(data.get('gf_count_rzeka_in_10km', 2)),
        'gf_count_jezioro_in_5km': int(data.get('gf_count_jezioro_in_5km', 0)),
        'gf_count_jezioro_in_10km': int(data.get('gf_count_jezioro_in_10km', 1)),
        
        # Odległości do obiektów
        'gf_distance_to_nearest_szkola': float(data.get('gf_distance_to_nearest_szkola', 0.5)),
        'gf_distance_to_nearest_szpital': float(data.get('gf_distance_to_nearest_szpital', 1.2)),
        'gf_distance_to_nearest_park': float(data.get('gf_distance_to_nearest_park', 0.8)),
        'gf_distance_to_nearest_centrum_handlowe': float(data.get('gf_distance_to_nearest_centrum_handlowe', 0.6)),
        'gf_distance_to_nearest_dworzec': float(data.get('gf_distance_to_nearest_dworzec', 2.0)),
        'gf_distance_to_nearest_muzeum': float(data.get('gf_distance_to_nearest_muzeum', 1.5)),
        'gf_distance_to_nearest_teatr': float(data.get('gf_distance_to_nearest_teatr', 1.8)),
        'gf_distance_to_nearest_hotel': float(data.get('gf_distance_to_nearest_hotel', 0.9)),
        'gf_distance_to_nearest_rzeka': float(data.get('gf_distance_to_nearest_rzeka', 0.3)),
        'gf_distance_to_nearest_jezioro': float(data.get('gf_distance_to_nearest_jezioro', 8.0)),
        
        # Dodatkowe cechy
        'gf_id': int(data.get('gf_id', 1)),
        'geo_nn_distance_m': float(data.get('geo_nn_distance_m', 100.0)),
        'rent_amount': float(data.get('rent_amount', 0.0)),
    }
    
    # Stwórz DataFrame
    df = pd.DataFrame([apartment_data])
    
    return df

@app.route('/')
def home():
    """Strona główna API"""
    return jsonify({
        "message": "API wyceny nieruchomości Random Forest",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - predykcja ceny mieszkania",
            "/health": "GET - sprawdzenie stanu API",
            "/upload-model": "POST - przesłanie modelu (.joblib)"
        },
        "model_status": "loaded" if model is not None else "not_loaded"
    })

@app.route('/health')
def health():
    """Sprawdzenie stanu API"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/upload-model', methods=['POST'])
def upload_model():
    """Endpoint do przesłania modelu"""
    try:
        if 'model' not in request.files:
            return jsonify({"error": "Brak pliku modelu w request"}), 400
        
        file = request.files['model']
        if file.filename == '':
            return jsonify({"error": "Nie wybrano pliku"}), 400
        
        if not file.filename.endswith('.joblib'):
            return jsonify({"error": "Plik musi mieć rozszerzenie .joblib"}), 400
        
        # Utwórz katalog artifacts jeśli nie istnieje
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Zapisz plik
        model_path = artifacts_dir / "model_rf.joblib"
        file.save(model_path.as_posix())
        
        # Przeładuj model
        global model
        if load_model():
            return jsonify({
                "message": "Model został przesłany i załadowany pomyślnie",
                "file_size": f"{model_path.stat().st_size / (1024*1024):.2f} MB",
                "success": True
            })
        else:
            return jsonify({"error": "Model został przesłany, ale nie udało się go załadować"}), 500
            
    except Exception as e:
        return jsonify({
            "error": f"Błąd podczas przesyłania modelu: {str(e)}",
            "success": False
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predykcja ceny mieszkania"""
    if model is None:
        return jsonify({"error": "Model nie został załadowany"}), 500
    
    try:
        # Pobierz dane z request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Brak danych w request"}), 400
        
        # Przygotuj dane mieszkania
        df = prepare_apartment_data(data)
        
        # Wykonaj predykcję
        prediction = model.predict(df)[0]
        
        # Oblicz cenę za m²
        area = float(data.get('area', 62.0))
        price_per_sqm = prediction / area
        
        # Zwróć wynik
        return jsonify({
            "predicted_price": round(prediction, 2),
            "price_per_sqm": round(price_per_sqm, 2),
            "area": area,
            "city": data.get('city', 'Toruń'),
            "rooms": int(data.get('rooms', 3)),
            "floor": int(data.get('floor', 4)),
            "total_floors": int(data.get('total_floors', 10)),
            "has_elevator": bool(data.get('has_elevator', True)),
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Błąd predykcji: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    # Załaduj model przy starcie
    if load_model():
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Nie można załadować modelu. Sprawdź czy plik artifacts/model_rf.joblib istnieje.")
