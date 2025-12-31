# ======================================================
# REGRESI LINIER BERGANDA
# STUDI KASUS: PRODUKSI PADI
# ======================================================

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------

# 2. LOAD DATASET
file_path = "produksiPadi.xlsx"   # pastikan file ada di folder yang sama
data = pd.read_excel(file_path)

print("Data awal:")
print(data.head())
print("\nInfo dataset:")
print(data.info())

# ------------------------------------------------------

# 3. CEK DATA KOSONG
print("\nJumlah data kosong tiap kolom:")
print(data.isnull().sum())

# Jika ada data kosong (opsional)
data = data.dropna()

# ------------------------------------------------------

# 4. MENENTUKAN VARIABEL INDEPENDEN (X) DAN DEPENDEN (Y)
# Ganti nama kolom jika berbeda dengan dataset kamu

X = data[['Luas Panen', 'Curah Hujan']]
y = data['Produksi Padi']

print("\nVariabel X:")
print(X.head())

print("\nVariabel Y:")
print(y.head())

# ------------------------------------------------------

# 5. SPLIT DATA TRAINING DAN TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])

# ------------------------------------------------------

# 6. MEMBUAT MODEL REGRESI LINIER BERGANDA
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------

# 7. MENAMPILKAN HASIL MODEL
print("\n=== HASIL REGRESI LINIER BERGANDA ===")
print("Intercept:", model.intercept_)

for i, col in enumerate(X.columns):
    print(f"Koefisien {col}: {model.coef_[i]}")

# ------------------------------------------------------

# 8. PREDIKSI DATA TESTING
y_pred = model.predict(X_test)

# ------------------------------------------------------

# 9. EVALUASI MODEL
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== EVALUASI MODEL ===")
print("Mean Squared Error (MSE):", mse)
print("R-Squared (R²):", r2)

# ------------------------------------------------------

# 10. VISUALISASI AKTUAL VS PREDIKSI
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Nilai Aktual Produksi Padi")
plt.ylabel("Nilai Prediksi Produksi Padi")
plt.title("Aktual vs Prediksi Produksi Padi")
plt.grid(True)
plt.show()

# ------------------------------------------------------

# 11. CONTOH PREDIKSI DATA BARU
# Misalnya:
# Luas Panen = 1500 hektar
# Curah Hujan = 220 mm

data_baru = np.array([[1500, 220]])
prediksi = model.predict(data_baru)

print("\nPrediksi Produksi Padi (data baru):", prediksi[0])

# ======================================================
# SELESAI
# ======================================================
