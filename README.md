# Sales Analysis & Prediction 📈

Model Machine Learning przewidujący sprzedaż na podstawie czynników zewnętrznych.

## Wyniki
- Błąd modelu: **±88 zł** (tylko 7% odchylenia)
- Algorytm: Random Forest Regressor (100 drzew)
- Zbiór danych: 1000 rekordów sprzedażowych

## Co odkrył model?
| Czynnik | Wpływ na sprzedaż |
|---|---|
| Temperatura | 45% |
| Promocja | 38% |
| Konkurencja w pobliżu | 11% |
| Miesiąc | 4% |
| Dzień tygodnia | 3% |

## Technologie
- Python 3.11
- scikit-learn (Random Forest Regressor)
- pandas (przetwarzanie danych)
- matplotlib (wizualizacja)
- numpy (obliczenia numeryczne)

## Jak uruchomić

1. Zainstaluj biblioteki:
```bash
pip install pandas numpy scikit-learn matplotlib
```

2. Uruchom analizę:
```bash
python sales.py
```

## Przykład przewidywania
Grudniowa promocja przy temperaturze 5°C i 2 konkurentach w pobliżu:
→ Przewidywana sprzedaż: **1217 zł**

## Wizualizacja



```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Generujemy przykładowe dane sprzedażowe
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'miesiac': np.random.randint(1, 13, n),
    'dzien_tygodnia': np.random.randint(0, 7, n),
    'promocja': np.random.randint(0, 2, n),
    'temperatura': np.random.uniform(-10, 35, n),
    'konkurencja_w_pobliżu': np.random.randint(0, 5, n),
    'sprzedaz': None
})

# Sprzedaż zależy od kilku czynników
df['sprzedaz'] = (
    1000
    + df['promocja'] * 300
    + df['temperatura'] * 10
    - df['konkurencja_w_pobliżu'] * 50
    + np.random.normal(0, 100, n)
).round(2)

print("Dane sprzedażowe:")
print(df.head())
print(f"\nŚrednia sprzedaż: {df['sprzedaz'].mean():.2f} zł")

# Budujemy model przewidywania sprzedaży
cechy = ['miesiac', 'dzien_tygodnia', 'promocja', 
         'temperatura', 'konkurencja_w_pobliżu']
X = df[cechy]
y = df['sprzedaz']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nBłąd modelu: ±{mae:.2f} zł")

# Wykres - ważność cech
plt.figure(figsize=(10, 6))
waznosc = pd.Series(
    model.feature_importances_, 
    index=cechy
).sort_values(ascending=True)

waznosc.plot(kind='barh', color='steelblue')
plt.title('Co najbardziej wpływa na sprzedaż?')
plt.xlabel('Ważność cechy')
plt.tight_layout()
plt.savefig('waznosc_cech.png')
print("\nWykres zapisany jako waznosc_cech.png!")

# Przewidywanie dla nowego dnia
nowy_dzien = pd.DataFrame({
    'miesiac': [12],
    'dzien_tygodnia': [6],
    'promocja': [1],
    'temperatura': [5],
    'konkurencja_w_pobliżu': [2]
})

przewidywana = model.predict(nowy_dzien)
print(f"\nPrzewidywana sprzedaż w Święta z promocją: {przewidywana[0]:.2f} zł")
```
