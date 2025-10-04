# 🔗 Integracja API z kalkulatorynieruchomosci.pl

## 📋 Przegląd

API Random Forest jest gotowe do integracji z aplikacją kalkulatorynieruchomosci.pl. Oferuje prosty endpoint do wyceny nieruchomości z wysoką dokładnością.

## 🚀 Szybki start

### **1. URL API:**
```
https://your-app.railway.app
```

### **2. Test połączenia:**
```bash
curl https://your-app.railway.app/health
```

### **3. Przykład wyceny:**
```bash
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Toruń",
    "area": 62,
    "rooms": 3,
    "floor": 4,
    "total_floors": 10,
    "has_elevator": true
  }'
```

## 🔧 Integracja JavaScript

### **Podstawowa funkcja:**
```javascript
async function wycenMieszkanie(dane) {
  const response = await fetch('https://your-app.railway.app/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dane)
  });
  
  return await response.json();
}
```

### **Przykład użycia:**
```javascript
const dane = {
  city: "Toruń",
  area: 62,
  rooms: 3,
  floor: 4,
  total_floors: 10,
  has_elevator: true,
  has_balcony: true
};

wycenMieszkanie(dane).then(wynik => {
  if (wynik.success) {
    console.log(`Cena: ${wynik.predicted_price.toLocaleString()} zł`);
    console.log(`Za m²: ${wynik.price_per_sqm.toLocaleString()} zł/m²`);
  }
});
```

## 🔧 Integracja PHP

### **Funkcja wyceny:**
```php
function wycenMieszkanie($dane) {
    $url = 'https://your-app.railway.app/predict';
    
    $options = [
        'http' => [
            'header' => "Content-Type: application/json\r\n",
            'method' => 'POST',
            'content' => json_encode($dane)
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    
    return json_decode($result, true);
}
```

### **Przykład użycia:**
```php
$dane = [
    'city' => 'Toruń',
    'area' => 62,
    'rooms' => 3,
    'floor' => 4,
    'total_floors' => 10,
    'has_elevator' => true
];

$wynik = wycenMieszkanie($dane);

if ($wynik['success']) {
    echo "Cena: " . number_format($wynik['predicted_price']) . " zł";
    echo "Za m²: " . number_format($wynik['price_per_sqm']) . " zł/m²";
}
```

## 📊 Wymagane pola

### **Obowiązkowe:**
- `city` - miasto (string)
- `area` - powierzchnia w m² (number)
- `rooms` - liczba pokoi (number)
- `floor` - piętro (number)
- `total_floors` - całkowita liczba pięter (number)

### **Opcjonalne:**
- `has_elevator` - winda (boolean, default: false)
- `has_balcony` - balkon (boolean, default: false)
- `building_type` - typ budynku (string, default: "blok")
- `heating_type` - ogrzewanie (string, default: "miejskie")
- `standard_of_finish` - stan (string, default: "do wykończenia")
- `year_of_construction` - rok budowy (number, default: 2010)
- `market` - rynek (string, default: "wtórny")
- `district` - dzielnica (string, default: "Centrum")
- `province` - województwo (string, default: "kujawsko-pomorskie")

## 📈 Format odpowiedzi

### **Sukces:**
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

### **Błąd:**
```json
{
  "error": "Opis błędu",
  "success": false
}
```

## 🔒 Bezpieczeństwo

### **CORS:**
- API obsługuje CORS dla wszystkich domen
- Można ograniczyć do konkretnych domen

### **Rate Limiting:**
- Brak limitów (można dodać w przyszłości)
- Zalecane cache'owanie wyników

### **Walidacja:**
- Wszystkie pola są walidowane
- Błędy zwracane w formacie JSON

## 📱 Przykład integracji z formularzem

### **HTML:**
```html
<form id="wycenaForm">
  <input type="text" id="city" placeholder="Miasto" required>
  <input type="number" id="area" placeholder="Powierzchnia (m²)" required>
  <input type="number" id="rooms" placeholder="Liczba pokoi" required>
  <input type="number" id="floor" placeholder="Piętro" required>
  <input type="number" id="total_floors" placeholder="Całkowite piętra" required>
  <button type="submit">Wyceń</button>
</form>
```

### **JavaScript:**
```javascript
document.getElementById('wycenaForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const dane = {
    city: document.getElementById('city').value,
    area: parseFloat(document.getElementById('area').value),
    rooms: parseInt(document.getElementById('rooms').value),
    floor: parseInt(document.getElementById('floor').value),
    total_floors: parseInt(document.getElementById('total_floors').value)
  };
  
  try {
    const wynik = await wycenMieszkanie(dane);
    
    if (wynik.success) {
      alert(`Przewidywana cena: ${wynik.predicted_price.toLocaleString()} zł`);
    } else {
      alert(`Błąd: ${wynik.error}`);
    }
  } catch (error) {
    alert(`Błąd połączenia: ${error.message}`);
  }
});
```

## 🎯 Najlepsze praktyki

### **1. Cache'owanie:**
```javascript
const cache = new Map();

async function wycenMieszkanieZCache(dane) {
  const key = JSON.stringify(dane);
  
  if (cache.has(key)) {
    return cache.get(key);
  }
  
  const wynik = await wycenMieszkanie(dane);
  cache.set(key, wynik);
  
  return wynik;
}
```

### **2. Error handling:**
```javascript
async function bezpiecznaWycena(dane) {
  try {
    const wynik = await wycenMieszkanie(dane);
    
    if (!wynik.success) {
      throw new Error(wynik.error);
    }
    
    return wynik;
  } catch (error) {
    console.error('Błąd wyceny:', error);
    
    // Fallback lub domyślna wartość
    return {
      success: false,
      error: error.message,
      predicted_price: null
    };
  }
}
```

### **3. Loading states:**
```javascript
async function wycenMieszkanieZLoading(dane) {
  // Pokaż loading
  document.getElementById('loading').style.display = 'block';
  
  try {
    const wynik = await wycenMieszkanie(dane);
    return wynik;
  } finally {
    // Ukryj loading
    document.getElementById('loading').style.display = 'none';
  }
}
```

## 📞 Wsparcie

### **Test API:**
```bash
# Sprawdź stan
curl https://your-app.railway.app/health

# Test wyceny
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Toruń", "area": 62, "rooms": 3, "floor": 4, "total_floors": 10}'
```

### **Debugowanie:**
- Sprawdź logi w Railway Dashboard
- Użyj narzędzi deweloperskich przeglądarki
- Test lokalny: `python api_wrapper.py`

## 🚀 Gotowe pliki

1. **`API_SPECIFICATION.md`** - Pełna specyfikacja API
2. **`api_wrapper.py`** - Python wrapper
3. **`kalkulator_wyceny.html`** - Przykład integracji HTML/JS
4. **`test_api.py`** - Skrypt testowy

API jest gotowe do integracji z kalkulatorynieruchomosci.pl!
