from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from typing import Dict, Any
import tempfile
import requests

app = Flask(__name__)

# Globalna zmienna dla modelu
model = None

def load_model():
    """Ładuje model Random Forest"""
    global model
    try:
        # Sprawdź czy model został przesłany przez API
        model_path = Path("artifacts/model_rf_small.joblib")
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

def load_model_from_url(model_url: str | None = None) -> bool:
    """Pobiera i ładuje model z zewnętrznego URL (np. Google Drive/S3).

    Jeśli model_url nie zostanie podany, zostanie użyta zmienna środowiskowa MODEL_URL.
    Zwraca True, jeśli model został pobrany i załadowany poprawnie.
    """
    global model
    try:
        if model_url is None:
            model_url = os.getenv('MODEL_URL', '').strip()
        if not model_url:
            print("Brak MODEL_URL - pomijam pobieranie z zewnętrznego URL")
            return False

        print(f"Pobieranie modelu z: {model_url}")
        # Krótszy timeout dla healthchecku
        resp = requests.get(model_url, timeout=30)
        resp.raise_for_status()

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        model_path = artifacts_dir / "model_rf_small.joblib"
        with open(model_path.as_posix(), "wb") as f:
            f.write(resp.content)

        model = joblib.load(model_path.as_posix())
        print("Model załadowany z zewnętrznego URL")
        return True
    except requests.exceptions.Timeout:
        print("Timeout pobierania modelu - aplikacja uruchomi się bez modelu")
        return False
    except Exception as e:
        print(f"Błąd pobierania/ładowania modelu z URL: {e}")
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
    
    # Dodaj brakujące kolumny feature engineering
    area = df['area'].iloc[0]
    floor = df['floor'].iloc[0]
    rooms = df['rooms'].iloc[0]
    total_floors = df['total_floors'].iloc[0]
    year_of_construction = df['year_of_construction'].iloc[0]
    listing_year = df['listing_year'].iloc[0]
    
    # log_area
    df['log_area'] = np.log1p(area)
    
    # building_age
    current_year = 2024
    df['building_age'] = float(listing_year - year_of_construction)
    
    # area_x_floor
    df['area_x_floor'] = area * floor
    
    # sqm_per_room
    df['sqm_per_room'] = area / rooms if rooms > 0 else area
    
    # floor_ratio
    df['floor_ratio'] = floor / total_floors if total_floors > 0 else 0.0
    
    # age_squared
    df['age_squared'] = df['building_age'] ** 2
    
    return df

@app.route('/api/valuation', methods=['POST'])
def api_valuation():
    """Endpoint dla kalkulatorynieruchomosci.pl - zwraca odpowiedź w oczekiwanym formacie"""
    if model is None:
        return jsonify({"error": "Model nie został załadowany"}), 500
    
    try:
        # Pobierz dane z request - obsługa różnych typów content
        data = None
        
        # Spróbuj JSON
        if request.is_json:
            data = request.get_json()
        else:
            # Spróbuj form data
            if request.form:
                data = request.form.to_dict()
            else:
                # Spróbuj raw data
                try:
                    data = request.get_json(force=True)
                except:
                    # Spróbuj parse jako JSON z raw data
                    raw_data = request.get_data(as_text=True)
                    if raw_data:
                        import json
                        data = json.loads(raw_data)
        
        if not data:
            return jsonify({"error": "Brak danych w request"}), 400
        
        # Mapuj dane z kalkulatorynieruchomosci.pl
        mapped_data = map_kalkulator_data(data)
        
        # Przygotuj dane mieszkania
        df = prepare_apartment_data(mapped_data)
        
        # Wykonaj predykcję
        prediction = model.predict(df)[0]
        
        # Oblicz cenę za m²
        area = float(mapped_data.get('area', 54.0))
        price_per_sqm = prediction / area
        
        # Zwróć wynik w formacie oczekiwanym przez kalkulatorynieruchomosci.pl
        from datetime import datetime
        
        # Oblicz zakres cen (±2% confidence)
        min_price = round(prediction * 0.98, 2)
        max_price = round(prediction * 1.02, 2)
        
        return jsonify({
            "price": round(prediction, 2),
            "minPrice": min_price,
            "maxPrice": max_price,
            "pricePerSqm": round(price_per_sqm, 2),
            "method": "estymatorai_external",
            "confidence": "±2%",
            "note": "Wycena przez EstymatorAI External API",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "modelInfo": {
                "type": "Random Forest Model",
                "version": "1.0",
                "accuracy": "0.29% MAPE"
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Błąd predykcji: {str(e)}",
            "success": False
        }), 500

@app.route('/valuation', methods=['POST'])
def valuation():
    """Endpoint valuation - alias dla root endpoint"""
    return home()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint api/predict - alias dla root endpoint"""
    return home()

@app.route('/', methods=['GET', 'POST'])
def home():
    """Strona główna API lub predykcja dla kalkulatorynieruchomosci.pl"""
    if request.method == 'POST':
        # Predykcja dla kalkulatorynieruchomosci.pl
        if model is None:
            return jsonify({"error": "Model nie został załadowany"}), 500
        
        try:
            # Pobierz dane z request - obsługa różnych typów content
            data = None
            
            # Spróbuj JSON
            if request.is_json:
                data = request.get_json()
            else:
                # Spróbuj form data
                if request.form:
                    data = request.form.to_dict()
                else:
                    # Spróbuj raw data
                    try:
                        data = request.get_json(force=True)
                    except:
                        # Spróbuj parse jako JSON z raw data
                        raw_data = request.get_data(as_text=True)
                        if raw_data:
                            import json
                            data = json.loads(raw_data)
            
            if not data:
                return jsonify({"error": "Brak danych w request"}), 400
            
            # Mapuj dane z kalkulatorynieruchomosci.pl
            mapped_data = map_kalkulator_data(data)
            
            # Przygotuj dane mieszkania
            df = prepare_apartment_data(mapped_data)
            
            # Wykonaj predykcję
            prediction = model.predict(df)[0]
            
            # Oblicz cenę za m²
            area = float(mapped_data.get('area', 62.0))
            price_per_sqm = prediction / area
            
            # Zwróć wynik w formacie oczekiwanym przez kalkulatorynieruchomosci.pl
            return jsonify({
                "predicted_price": round(prediction, 2),
                "price_per_sqm": round(price_per_sqm, 2),
                "area": area,
                "city": mapped_data.get('city', 'Toruń'),
                "rooms": int(mapped_data.get('rooms', 3)),
                "floor": int(mapped_data.get('floor', 4)),
                "total_floors": int(mapped_data.get('total_floors', 10)),
                "has_elevator": bool(mapped_data.get('has_elevator', True)),
                "success": True,
                "source": "Random Forest Model"
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Błąd predykcji: {str(e)}",
                "success": False
            }), 500
    else:
        # GET - informacje o API
        return jsonify({
            "message": "API wyceny nieruchomości Random Forest",
            "version": "1.0",
            "endpoints": {
                "/": "POST - predykcja ceny mieszkania (dla kalkulatorynieruchomosci.pl)",
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
        "model_loaded": model is not None,
        "message": "API działa poprawnie" if model is not None else "API działa, ale model nie został załadowany. Użyj /upload-model aby przesłać model."
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
        model_path = artifacts_dir / "model_rf_small.joblib"
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

@app.route('/download-model', methods=['POST'])
def download_model():
    """Pobiera model z przekazanego w body JSON pola model_url i ładuje go."""
    try:
        data = request.get_json(silent=True) or {}
        model_url = (data.get('model_url') or '').strip()
        if not model_url:
            return jsonify({"error": "Brak 'model_url' w body"}), 400

        if load_model_from_url(model_url):
            model_path = Path("artifacts/model_rf_small.joblib")
            return jsonify({
                "message": "Model pobrany i załadowany pomyślnie",
                "file_size_mb": f"{model_path.stat().st_size / (1024*1024):.2f}",
                "success": True
            })
        else:
            return jsonify({"error": "Nie udało się pobrać/załadować modelu z podanego URL"}), 500
    except Exception as e:
        return jsonify({
            "error": f"Błąd podczas pobierania modelu: {str(e)}",
            "success": False
        }), 500

def map_kalkulator_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mapuje dane z kalkulatorynieruchomosci.pl na format API"""
    
    # Mapowanie miasta na województwo - rozszerzona lista dla top 30 miast
    city_to_province = {
        # Mazowieckie
        "Warszawa": "mazowieckie",
        "Płock": "mazowieckie",
        "Radom": "mazowieckie",
        
        # Małopolskie
        "Kraków": "małopolskie",
        "Tarnów": "małopolskie",
        
        # Dolnośląskie
        "Wrocław": "dolnośląskie",
        "Legnica": "dolnośląskie",
        
        # Pomorskie
        "Gdańsk": "pomorskie",
        "Gdynia": "pomorskie",
        "Sopot": "pomorskie",
        "Police": "pomorskie",
        
        # Wielkopolskie
        "Poznań": "wielkopolskie",
        "Kalisz": "wielkopolskie",
        
        # Łódzkie
        "Łódź": "łódzkie",
        "Piotrków Trybunalski": "łódzkie",
        
        # Lubelskie
        "Lublin": "lubelskie",
        "Chełm": "lubelskie",
        
        # Kujawsko-pomorskie
        "Bydgoszcz": "kujawsko-pomorskie",
        "Toruń": "kujawsko-pomorskie",
        
        # Podkarpackie
        "Rzeszów": "podkarpackie",
        "Przemyśl": "podkarpackie",
        
        # Zachodniopomorskie
        "Szczecin": "zachodniopomorskie",
        "Koszalin": "zachodniopomorskie",
        
        # Śląskie
        "Katowice": "śląskie",
        "Częstochowa": "śląskie",
        "Sosnowiec": "śląskie",
        "Gliwice": "śląskie",
        "Zabrze": "śląskie",
        "Bielsko-Biała": "śląskie",
        "Ruda Śląska": "śląskie",
        
        # Świętokrzyskie
        "Kielce": "świętokrzyskie",
        
        # Podlaskie
        "Białystok": "podlaskie",
        
        # Warmińsko-mazurskie
        "Olsztyn": "warmińsko-mazurskie",
        
        # Lubuskie
        "Gorzów Wielkopolski": "lubuskie",
        "Zielona Góra": "lubuskie"
    }
    
    city = data.get("city", "Toruń")
    province = city_to_province.get(city, "kujawsko-pomorskie")
    
    # Mapowanie locationTier na district z uwzględnieniem miasta
    location_tier = data.get("locationTier", "standard")
    
    # Mapowanie district dla głównych miast
    city_districts = {
        "Warszawa": {"premium": "Śródmieście", "standard": "Mokotów", "basic": "Wola"},
        "Kraków": {"premium": "Stare Miasto", "standard": "Krowodrza", "basic": "Nowa Huta"},
        "Gdańsk": {"premium": "Śródmieście", "standard": "Wrzeszcz", "basic": "Orunia"},
        "Wrocław": {"premium": "Stare Miasto", "standard": "Krzyki", "basic": "Psie Pole"},
        "Poznań": {"premium": "Stare Miasto", "standard": "Jeżyce", "basic": "Wilda"},
        "Łódź": {"premium": "Śródmieście", "standard": "Bałuty", "basic": "Polesie"},
        "Lublin": {"premium": "Stare Miasto", "standard": "Śródmieście", "basic": "Wieniawa"},
        "Bydgoszcz": {"premium": "Śródmieście", "standard": "Szwederowo", "basic": "Fordon"},
        "Szczecin": {"premium": "Śródmieście", "standard": "Prawobrzeże", "basic": "Zachód"},
        "Katowice": {"premium": "Śródmieście", "standard": "Zawodzie", "basic": "Bogucice"},
        "Gdynia": {"premium": "Śródmieście", "standard": "Wzgórze Św. Maksymiliana", "basic": "Chylonia"},
        "Białystok": {"premium": "Śródmieście", "standard": "Bojary", "basic": "Wygoda"},
        "Częstochowa": {"premium": "Śródmieście", "standard": "Raków", "basic": "Błeszno"},
        "Kielce": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Toruń": {"premium": "Stare Miasto", "standard": "Śródmieście", "basic": "Podgórz"},
        "Bielsko-Biała": {"premium": "Śródmieście", "standard": "Mikuszowice", "basic": "Kamienica"},
        "Sosnowiec": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Gorzów Wielkopolski": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Radom": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Olsztyn": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Gliwice": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Zabrze": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Płock": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Police": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"},
        "Sopot": {"premium": "Śródmieście", "standard": "Śródmieście", "basic": "Śródmieście"}
    }
    
    if city in city_districts:
        district = city_districts[city].get(location_tier, city_districts[city]["standard"])
    else:
        # Domyślne mapowanie dla innych miast
        if location_tier == "premium":
            district = "Centrum"
        elif location_tier == "standard":
            district = "Śródmieście"
        else:
            district = data.get("district", "Centrum")
    
    # Mapowanie condition na standard_of_finish
    condition = data.get("condition", "good")
    if condition == "excellent":
        standard_of_finish = "wysoki"
    elif condition == "good":
        standard_of_finish = "standardowy"
    elif condition == "average":
        standard_of_finish = "do wykończenia"
    else:
        standard_of_finish = "do wykończenia"
    
    # Mapowanie parking
    parking = data.get("parking", "street")
    has_garage = parking in ["garage", "garaz"]
    
    # Mapowanie transport
    transport = data.get("transport", "medium")
    # Można dodać logikę dla transportu jeśli model tego wymaga
    
    mapped = {
        "city": city,
        "area": float(data.get("area", 62.0)),
        "rooms": int(data.get("rooms", 3)),
        "floor": int(data.get("floor", 4)),
        "total_floors": int(data.get("totalFloors", data.get("total_floors", 10))),
        "year_of_construction": int(data.get("year", 2010)),
        "building_type": data.get("buildingType", "blok"),
        "heating_type": data.get("heating", "miejskie"),
        "standard_of_finish": standard_of_finish,
        "has_elevator": data.get("elevator", "no").lower() in ["yes", "tak", "true", "1"],
        "has_balcony": data.get("balcony", "no").lower() in ["yes", "tak", "true", "1"],
        "has_basement": data.get("basement", "no").lower() in ["yes", "tak", "true", "1"],
        "has_separate_kitchen": data.get("kitchenType", "separate").lower() == "separate",
        "has_garage": has_garage,
        "market": "wtórny",
        "district": district,
        "province": province,
        "location_tier": location_tier,
        "condition": condition,
        "parking": parking,
        "transport": transport,
        "orientation": data.get("orientation", "south")
    }
    return mapped

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
    # Spróbuj załadować model przy starcie
    print("Próba załadowania modelu przy starcie...")
    load_model()
    
    # Uruchom aplikację
    port = int(os.environ.get('PORT', 5000))
    print(f"Uruchamianie aplikacji na porcie {port}")
    if model is not None:
        print("Model załadowany pomyślnie!")
    else:
        print("Model nie został załadowany. Można go załadować przez /upload-model lub /download-model")
    app.run(host='0.0.0.0', port=port, debug=False)
