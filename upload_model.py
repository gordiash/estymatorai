#!/usr/bin/env python3
"""
Skrypt do przesłania modelu na Railway
"""
import requests
import os
from pathlib import Path

def upload_model_to_railway(railway_url, model_path="artifacts/model_rf.joblib"):
    """Przesyła model na Railway"""
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Plik modelu nie istnieje: {model_path}")
        return False
    
    print(f"📤 Przesyłanie modelu: {model_file.name}")
    print(f"📊 Rozmiar: {model_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        with open(model_file, 'rb') as f:
            files = {'model': (model_file.name, f, 'application/octet-stream')}
            response = requests.post(f"{railway_url}/upload-model", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model przesłany pomyślnie!")
            print(f"📝 Odpowiedź: {result.get('message', 'OK')}")
            if 'file_size' in result:
                print(f"📊 Rozmiar na serwerze: {result['file_size']} MB")
            return True
        else:
            print(f"❌ Błąd przesyłania: {response.status_code}")
            print(f"📝 Odpowiedź: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Nie można połączyć się z {railway_url}")
        print("Sprawdź czy aplikacja jest uruchomiona na Railway")
        return False
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False

def test_api(railway_url):
    """Test API po przesłaniu modelu"""
    try:
        # Test health
        print("\n🔍 Testowanie /health...")
        health_response = requests.get(f"{railway_url}/health")
        print(f"Status: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
        
        # Test predict
        print("\n🏠 Testowanie /predict...")
        apartment_data = {
            "city": "Toruń",
            "area": 62.0,
            "rooms": 3,
            "floor": 4,
            "total_floors": 10,
            "has_elevator": True,
            "has_balcony": True
        }
        
        predict_response = requests.post(
            f"{railway_url}/predict",
            headers={"Content-Type": "application/json"},
            json=apartment_data
        )
        
        print(f"Status: {predict_response.status_code}")
        result = predict_response.json()
        print(f"Response: {result}")
        
        if result.get("success"):
            print(f"\n✅ Test zakończony pomyślnie!")
            print(f"💰 Przewidywana cena: {result['predicted_price']:,.0f} zł")
            print(f"📊 Cena za m²: {result['price_per_sqm']:,.0f} zł/m²")
            return True
        else:
            print(f"\n❌ Test nieudany: {result.get('error', 'Nieznany błąd')}")
            return False
            
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")
        return False

def main():
    """Główna funkcja"""
    print("🚀 Railway Model Upload Tool")
    print("=" * 50)
    
    # Pobierz URL Railway
    railway_url = input("Podaj URL aplikacji Railway (np. https://your-app.railway.app): ").strip()
    
    if not railway_url:
        print("❌ Musisz podać URL aplikacji Railway")
        return
    
    if not railway_url.startswith('http'):
        railway_url = f"https://{railway_url}"
    
    print(f"\n🎯 Cel: {railway_url}")
    
    # Prześlij model
    if upload_model_to_railway(railway_url):
        print("\n" + "=" * 50)
        # Test API
        test_api(railway_url)
    else:
        print("\n❌ Nie udało się przesłać modelu")

if __name__ == "__main__":
    main()
