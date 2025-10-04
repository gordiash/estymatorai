#!/usr/bin/env python3
"""
Test konkretnego mieszkania z Torunia
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def test_torun_apartment():
    """Test mieszkania z Torunia: 62m2, 3 pokoje, 4 piętro w 10-piętrowym budynku z windą"""
    
    # Wczytaj model
    artifacts = Path("artifacts")
    model_path = artifacts / "model_rf.joblib"
    
    if not model_path.exists():
        print("❌ Brak modelu Random Forest! Uruchom najpierw trening.")
        return
    
    model = joblib.load(model_path.as_posix())
    
    # Dane mieszkania z Torunia
    apartment_data = {
        # Podstawowe informacje
        'city': 'Toruń',
        'area': 62.0,
        'rooms': 3,
        'floor': 4,
        'total_floors': 10,
        
        # Cechy budynku
        'building_type': 'blok',
        'year_of_construction': 2010,  # Przykładowy rok
        'heating_type': 'miejskie',
        'standard_of_finish': 'do wykończenia',
        
        # Flagi wyposażenia
        'has_balcony': 1,
        'has_garage': 0,
        'has_garden': 0,
        'has_elevator': 1,  # Budynek ma windę
        'has_basement': 0,
        'has_separate_kitchen': 1,
        'has_dishwasher': 0,
        'has_fridge': 0,
        'has_oven': 0,
        
        # Dodatkowe cechy
        'market': 'wtórny',
        'district': 'Centrum',
        'province': 'kujawsko-pomorskie',
        
        # Cechy czasowe (przykładowe)
        'listing_year': 2024,
        'listing_month': 10,
        'listing_quarter': 4,
    }
    
    # Stwórz DataFrame
    df = pd.DataFrame([apartment_data])
    
    # Usuń kolumnę price (jeśli istnieje) - nie potrzebujemy jej do predykcji
    if 'price' in df.columns:
        df = df.drop(columns=['price'])
    
    # Wykonaj predykcję
    try:
        prediction = model.predict(df)[0]
        
        print("🏠 TEST MIESZKANIA Z TORUNIA")
        print("=" * 50)
        print(f"📍 Lokalizacja: Toruń, Centrum")
        print(f"📐 Powierzchnia: {apartment_data['area']} m²")
        print(f"🚪 Liczba pokoi: {apartment_data['rooms']}")
        print(f"🏢 Piętro: {apartment_data['floor']}/{apartment_data['total_floors']}")
        print(f"🏗️ Typ budynku: {apartment_data['building_type']}")
        print(f"🛗 Winda: {'TAK' if apartment_data['has_elevator'] else 'NIE'}")
        print(f"🌡️ Ogrzewanie: {apartment_data['heating_type']}")
        print(f"🔧 Stan: {apartment_data['standard_of_finish']}")
        print(f"🌿 Balkon: {'TAK' if apartment_data['has_balcony'] else 'NIE'}")
        print()
        print(f"💰 PREDYKCJA CENY: {prediction:,.0f} zł")
        print(f"💰 PREDYKCJA CENY: {prediction:,.0f} zł".replace(",", " "))
        
        # Dodatkowe informacje
        price_per_sqm = prediction / apartment_data['area']
        print(f"📊 Cena za m²: {price_per_sqm:,.0f} zł/m²")
        print(f"📊 Cena za m²: {price_per_sqm:,.0f} zł/m²".replace(",", " "))
        
        print("\n✅ Test zakończony pomyślnie!")
        
    except Exception as e:
        print(f"❌ Błąd podczas predykcji: {e}")
        print("Sprawdź czy model został wytrenowany poprawnie.")

if __name__ == "__main__":
    test_torun_apartment()
