#!/usr/bin/env python3
"""
Test aplikacji bez modelu - sprawdza czy API startuje
"""
import requests
import time

def test_app_startup(url):
    """Test czy aplikacja startuje"""
    print(f"🔍 Testowanie aplikacji: {url}")
    
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Próba {attempt}/{max_attempts}...")
            response = requests.get(f"{url}/health", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Aplikacja działa!")
                print(f"Status: {result.get('status')}")
                print(f"Model załadowany: {result.get('model_loaded')}")
                print(f"Wiadomość: {result.get('message')}")
                return True
            else:
                print(f"❌ Błąd HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Brak połączenia (próba {attempt})")
        except requests.exceptions.Timeout:
            print(f"❌ Timeout (próba {attempt})")
        except Exception as e:
            print(f"❌ Błąd: {e}")
        
        if attempt < max_attempts:
            print("⏳ Czekam 10 sekund...")
            time.sleep(10)
    
    print(f"❌ Aplikacja nie odpowiada po {max_attempts} próbach")
    return False

def test_basic_endpoints(url):
    """Test podstawowych endpointów"""
    print(f"\n🧪 Testowanie endpointów...")
    
    # Test głównej strony
    try:
        response = requests.get(f"{url}/", timeout=5)
        if response.status_code == 200:
            print("✅ GET / - OK")
        else:
            print(f"❌ GET / - HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ GET / - Błąd: {e}")
    
    # Test health
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ GET /health - OK")
        else:
            print(f"❌ GET /health - HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ GET /health - Błąd: {e}")

def main():
    """Główna funkcja"""
    print("🚀 Test aplikacji Railway")
    print("=" * 50)
    
    url = input("Podaj URL aplikacji Railway: ").strip()
    if not url:
        print("❌ Musisz podać URL")
        return
    
    if not url.startswith('http'):
        url = f"https://{url}"
    
    print(f"🎯 Testowanie: {url}")
    
    # Test startu aplikacji
    if test_app_startup(url):
        # Test endpointów
        test_basic_endpoints(url)
        print(f"\n✅ Aplikacja działa poprawnie!")
        print(f"📤 Teraz możesz przesłać model przez:")
        print(f"   python upload_model.py")
    else:
        print(f"\n❌ Aplikacja nie działa. Sprawdź logi w Railway Dashboard.")

if __name__ == "__main__":
    main()
