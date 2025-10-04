# 🏠 API dla kalkulatorynieruchomosci.pl

## 📋 Specyfikacja API

### **Base URL:** `https://your-app.railway.app`

## 🔗 Endpointy API

### 1. **GET /** - Informacje o API
```http
GET /
```

**Odpowiedź:**
```json
{
  "message": "API wyceny nieruchomości Random Forest",
  "version": "1.0",
  "endpoints": {
    "/predict": "POST - predykcja ceny mieszkania",
    "/health": "GET - sprawdzenie stanu API",
    "/upload-model": "POST - przesłanie modelu (.joblib)"
  },
  "model_status": "loaded"
}
```

### 2. **GET /health** - Sprawdzenie stanu
```http
GET /health
```

**Odpowiedź:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API działa poprawnie"
}
```

### 3. **POST /predict** - Predykcja ceny mieszkania
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "city": "Toruń",
  "area": 62.0,
  "rooms": 3,
  "floor": 4,
  "total_floors": 10,
  "has_elevator": true,
  "has_balcony": true,
  "building_type": "blok",
  "heating_type": "miejskie",
  "standard_of_finish": "do wykończenia",
  "year_of_construction": 2010,
  "market": "wtórny",
  "district": "Centrum",
  "province": "kujawsko-pomorskie"
}
```

**Odpowiedź:**
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

## 🔧 Integracja z kalkulatorynieruchomosci.pl

### **Przykład integracji JavaScript:**

```javascript
// Funkcja wyceny mieszkania
async function wycenMieszkanie(daneMieszkania) {
  try {
    const response = await fetch('https://your-app.railway.app/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(daneMieszkania)
    });
    
    const result = await response.json();
    
    if (result.success) {
      return {
        cena: result.predicted_price,
        cenaZaM2: result.price_per_sqm,
        sukces: true
      };
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('Błąd wyceny:', error);
    return {
      sukces: false,
      blad: error.message
    };
  }
}

// Przykład użycia
const daneMieszkania = {
  city: "Toruń",
  area: 62,
  rooms: 3,
  floor: 4,
  total_floors: 10,
  has_elevator: true,
  has_balcony: true,
  building_type: "blok",
  heating_type: "miejskie"
};

wycenMieszkanie(daneMieszkania).then(wynik => {
  if (wynik.sukces) {
    console.log(`Przewidywana cena: ${wynik.cena.toLocaleString()} zł`);
    console.log(`Cena za m²: ${wynik.cenaZaM2.toLocaleString()} zł/m²`);
  } else {
    console.error('Błąd:', wynik.blad);
  }
});
```

### **Przykład integracji PHP:**

```php
<?php
function wycenMieszkanie($daneMieszkania) {
    $url = 'https://your-app.railway.app/predict';
    
    $options = [
        'http' => [
            'header' => "Content-Type: application/json\r\n",
            'method' => 'POST',
            'content' => json_encode($daneMieszkania)
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    
    if ($result === FALSE) {
        return ['sukces' => false, 'blad' => 'Błąd połączenia z API'];
    }
    
    $odpowiedz = json_decode($result, true);
    
    if ($odpowiedz['success']) {
        return [
            'sukces' => true,
            'cena' => $odpowiedz['predicted_price'],
            'cenaZaM2' => $odpowiedz['price_per_sqm']
        ];
    } else {
        return ['sukces' => false, 'blad' => $odpowiedz['error']];
    }
}

// Przykład użycia
$daneMieszkania = [
    'city' => 'Toruń',
    'area' => 62,
    'rooms' => 3,
    'floor' => 4,
    'total_floors' => 10,
    'has_elevator' => true,
    'has_balcony' => true,
    'building_type' => 'blok',
    'heating_type' => 'miejskie'
];

$wynik = wycenMieszkanie($daneMieszkania);

if ($wynik['sukces']) {
    echo "Przewidywana cena: " . number_format($wynik['cena']) . " zł\n";
    echo "Cena za m²: " . number_format($wynik['cenaZaM2']) . " zł/m²\n";
} else {
    echo "Błąd: " . $wynik['blad'] . "\n";
}
?>
```

## 📊 Wymagane pola

### **Pola obowiązkowe:**
- `city` - miasto (string)
- `area` - powierzchnia w m² (number)
- `rooms` - liczba pokoi (number)
- `floor` - piętro (number)
- `total_floors` - całkowita liczba pięter (number)

### **Pola opcjonalne:**
- `has_elevator` - czy ma windę (boolean, default: false)
- `has_balcony` - czy ma balkon (boolean, default: false)
- `building_type` - typ budynku (string, default: "blok")
- `heating_type` - typ ogrzewania (string, default: "miejskie")
- `standard_of_finish` - stan wykończenia (string, default: "do wykończenia")
- `year_of_construction` - rok budowy (number, default: 2010)
- `market` - rynek (string, default: "wtórny")
- `district` - dzielnica (string, default: "Centrum")
- `province` - województwo (string, default: "kujawsko-pomorskie")

## 🔒 Bezpieczeństwo

### **Rate Limiting:**
- Brak limitów (można dodać w przyszłości)
- Zalecane cache'owanie wyników

### **CORS:**
- API obsługuje CORS dla wszystkich domen
- Można ograniczyć do konkretnych domen

### **Walidacja:**
- Wszystkie pola są walidowane
- Błędy zwracane w formacie JSON

## 📈 Monitoring

### **Logi:**
- Wszystkie requesty są logowane
- Railway Dashboard → Logs

### **Metryki:**
- Railway Dashboard → Metrics
- CPU, RAM, Network usage

## 🚀 Deployment

### **Railway:**
- Automatyczny deployment z GitHub
- HTTPS z certyfikatem SSL
- Skalowanie automatyczne

### **Backup:**
- Model można przesłać ponownie przez `/upload-model`
- Dane nie są przechowywane (stateless API)

## 🔄 Aktualizacja modelu

```bash
# Przesłanie nowego modelu
curl -X POST https://your-app.railway.app/upload-model \
  -F "model=@artifacts/model_rf.joblib"
```

## 📞 Wsparcie

### **Test API:**
```bash
# Test health
curl https://your-app.railway.app/health

# Test predykcji
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Toruń", "area": 62, "rooms": 3, "floor": 4, "total_floors": 10}'
```

### **Debugowanie:**
- Sprawdź logi w Railway Dashboard
- Test lokalny: `python test_api.py`
