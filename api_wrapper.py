#!/usr/bin/env python3
"""
Wrapper API dla kalkulatorynieruchomosci.pl
"""
import requests
import json
from typing import Dict, Any, Optional

class NieruchomosciAPI:
    """Klasa do komunikacji z API wyceny nieruchomości"""
    
    def __init__(self, base_url: str = "https://your-app.railway.app"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'kalkulatorynieruchomosci.pl/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Sprawdza stan API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False
            }
    
    def wycen_mieszkanie(self, dane: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wycenia mieszkanie na podstawie podanych danych
        
        Args:
            dane: Słownik z danymi mieszkania
            
        Returns:
            Słownik z wynikiem wyceny
        """
        try:
            # Walidacja podstawowych pól
            wymagane_pola = ['city', 'area', 'rooms', 'floor', 'total_floors']
            for pole in wymagane_pola:
                if pole not in dane:
                    return {
                        "success": False,
                        "error": f"Brakuje wymaganego pola: {pole}"
                    }
            
            # Wyślij request
            response = self.session.post(
                f"{self.base_url}/predict",
                json=dane,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("success"):
                return {
                    "success": True,
                    "cena": result["predicted_price"],
                    "cena_za_m2": result["price_per_sqm"],
                    "powierzchnia": result["area"],
                    "miasto": result["city"],
                    "pokoje": result["rooms"],
                    "pietro": result["floor"],
                    "calkowite_pietra": result["total_floors"],
                    "ma_winde": result["has_elevator"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Nieznany błąd")
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Timeout - API nie odpowiada"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Błąd połączenia z API"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Błąd HTTP: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Nieoczekiwany błąd: {e}"
            }
    
    def format_cena(self, cena: float) -> str:
        """Formatuje cenę do wyświetlenia"""
        return f"{cena:,.0f} zł".replace(",", " ")
    
    def format_cena_za_m2(self, cena: float) -> str:
        """Formatuje cenę za m² do wyświetlenia"""
        return f"{cena:,.0f} zł/m²".replace(",", " ")

# Przykład użycia
def main():
    """Przykład użycia API"""
    api = NieruchomosciAPI()
    
    # Sprawdź stan API
    print("🔍 Sprawdzanie stanu API...")
    health = api.health_check()
    print(f"Status: {health.get('status')}")
    print(f"Model załadowany: {health.get('model_loaded')}")
    
    if not health.get('model_loaded'):
        print("❌ Model nie został załadowany!")
        return
    
    # Dane mieszkania
    dane_mieszkania = {
        "city": "Toruń",
        "area": 62.0,
        "rooms": 3,
        "floor": 4,
        "total_floors": 10,
        "has_elevator": True,
        "has_balcony": True,
        "building_type": "blok",
        "heating_type": "miejskie",
        "standard_of_finish": "do wykończenia",
        "year_of_construction": 2010,
        "market": "wtórny",
        "district": "Centrum"
    }
    
    # Wycen mieszkanie
    print("\n🏠 Wycenianie mieszkania...")
    wynik = api.wycen_mieszkanie(dane_mieszkania)
    
    if wynik["success"]:
        print("✅ Wycena zakończona pomyślnie!")
        print(f"💰 Przewidywana cena: {api.format_cena(wynik['cena'])}")
        print(f"📊 Cena za m²: {api.format_cena_za_m2(wynik['cena_za_m2'])}")
        print(f"📐 Powierzchnia: {wynik['powierzchnia']} m²")
        print(f"🏢 Miasto: {wynik['miasto']}")
        print(f"🚪 Pokoje: {wynik['pokoje']}")
        print(f"🏢 Piętro: {wynik['pietro']}/{wynik['calkowite_pietra']}")
        print(f"🛗 Winda: {'TAK' if wynik['ma_winde'] else 'NIE'}")
    else:
        print(f"❌ Błąd wyceny: {wynik['error']}")

if __name__ == "__main__":
    main()
