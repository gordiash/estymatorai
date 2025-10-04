#!/usr/bin/env python3
"""
Test API wyceny nieruchomości
"""
import requests
import json

def test_api():
    """Test API lokalnie"""
    base_url = "http://localhost:5000"
    
    # Test danych mieszkania z Torunia
    apartment_data = {
        "city": "Toruń",
        "area": 62.0,
        "rooms": 3,
        "floor": 4,
        "total_floors": 10,
        "has_elevator": True,
        "has_balcony": True,
        "building_type": "blok",
        "heating_type": "miejskie",
        "standard_of_finish": "do wykończenia"
    }
    
    try:
        # Test health endpoint
        print("🔍 Testowanie /health...")
        health_response = requests.get(f"{base_url}/health")
        print(f"Status: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
        print()
        
        # Test predict endpoint
        print("🏠 Testowanie /predict...")
        predict_response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=apartment_data
        )
        
        print(f"Status: {predict_response.status_code}")
        result = predict_response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result.get("success"):
            print(f"\n✅ Test zakończony pomyślnie!")
            print(f"💰 Przewidywana cena: {result['predicted_price']:,.0f} zł")
            print(f"📊 Cena za m²: {result['price_per_sqm']:,.0f} zł/m²")
        else:
            print(f"\n❌ Test nieudany: {result.get('error', 'Nieznany błąd')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Nie można połączyć się z API. Upewnij się, że aplikacja jest uruchomiona.")
        print("Uruchom: python app.py")
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")

if __name__ == "__main__":
    test_api()
