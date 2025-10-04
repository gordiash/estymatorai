#!/usr/bin/env python3
"""
Test konkretnego mieszkania z Torunia - używa istniejących danych jako wzorca
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def test_torun_apartment_simple():
    """Test mieszkania z Torunia używając istniejących danych jako wzorca"""
    
    # Wczytaj model
    artifacts = Path("artifacts")
    model_path = artifacts / "model_rf.joblib"
    
    if not model_path.exists():
        print("Brak modelu Random Forest! Uruchom najpierw trening.")
        return
    
    model = joblib.load(model_path.as_posix())
    
    # Wczytaj przykładowe dane z Torunia
    try:
        from src.config import CreateSqlAlchemyEngine
        from src.data_loader import LoadTables
        from src.feature_engineering import CleanAndEngineerFeatures
        
        engine = CreateSqlAlchemyEngine()
        raw_df, _ = LoadTables(engine)
        Xy, cat_cols, target_col = CleanAndEngineerFeatures(raw_df)
        
        # Znajdź mieszkania z Torunia
        torun_data = Xy[Xy['city'] == 'Toruń'].copy()
        
        if len(torun_data) == 0:
            print("Brak danych z Torunia w zbiorze!")
            return
            
        # Wybierz mieszkanie podobne do opisanego (62m2, 3 pokoje, 4 piętro)
        similar = torun_data[
            (torun_data['area'] >= 60) & (torun_data['area'] <= 65) &
            (torun_data['rooms'] == 3) &
            (torun_data['floor'] >= 3) & (torun_data['floor'] <= 5)
        ]
        
        if len(similar) == 0:
            # Jeśli nie ma dokładnie podobnych, weź najbliższe
            similar = torun_data[
                (torun_data['area'] >= 55) & (torun_data['area'] <= 70) &
                (torun_data['rooms'] == 3)
            ]
        
        if len(similar) == 0:
            # Jeśli nadal brak, weź pierwsze mieszkanie z Torunia
            similar = torun_data.head(1)
        
        # Wybierz pierwsze mieszkanie z podobnych
        apartment = similar.iloc[0].copy()
        
        # Modyfikuj dane zgodnie z opisem
        apartment['area'] = 62.0
        apartment['rooms'] = 3
        apartment['floor'] = 4
        apartment['total_floors'] = 10
        apartment['has_elevator'] = 1  # Budynek ma windę
        apartment['has_balcony'] = 1
        
        # Usuń cenę (target)
        if 'price' in apartment.index:
            apartment = apartment.drop('price')
        
        # Stwórz DataFrame z jednym wierszem
        df = pd.DataFrame([apartment])
        
        # Wykonaj predykcję
        prediction = model.predict(df)[0]
        
        print("TEST MIESZKANIA Z TORUNIA")
        print("=" * 50)
        print(f"Lokalizacja: Torun, {apartment.get('district', 'Centrum')}")
        print(f"Powierzchnia: {apartment['area']} m2")
        print(f"Liczba pokoi: {apartment['rooms']}")
        print(f"Pietro: {apartment['floor']}/{apartment['total_floors']}")
        print(f"Typ budynku: {apartment.get('building_type', 'blok')}")
        print(f"Winda: {'TAK' if apartment.get('has_elevator', 0) else 'NIE'}")
        print(f"Ogrzewanie: {apartment.get('heating_type', 'miejskie')}")
        print(f"Stan: {apartment.get('standard_of_finish', 'do wykończenia')}")
        print(f"Balkon: {'TAK' if apartment.get('has_balcony', 0) else 'NIE'}")
        print()
        print(f"PREDYKCJA CENY: {prediction:,.0f} zl")
        
        # Dodatkowe informacje
        price_per_sqm = prediction / apartment['area']
        print(f"Cena za m2: {price_per_sqm:,.0f} zl/m2")
        
        print("\nTest zakonczony pomyslnie!")
        
    except Exception as e:
        print(f"Blad podczas testu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_torun_apartment_simple()
