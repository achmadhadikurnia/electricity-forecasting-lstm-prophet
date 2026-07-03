import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Matikan peringatan (warning) dari mesin TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sembunyikan Info & Warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Matikan notifikasi oneDNN

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from prophet import Prophet

warnings.filterwarnings('ignore')

def muat_data(filepath='data/household_power_consumption.txt'):
    print("\n[TAHAP 1] Pengumpulan dan Pemuatan Data (Data Loading)")
    print(f"Membaca dataset dari {filepath}...")
    df = pd.read_csv(filepath, sep=';', 
                     low_memory=False, 
                     na_values=['nan', '?'])
    
    # Gabungkan Date dan Time secara manual (Pandas 3.0+ compatibility)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')
    
    # Mengisi nilai kosong (imputasi maju)
    df.ffill(inplace=True)
    return df

def praproses_data(df):
    print("\n[TAHAP 2] Pra-Pemrosesan Data (Data Preprocessing)")
    print("Memproses data (mengubah resolusi menjadi per jam / hourly)...")
    df_jam = df.resample('h').mean()
    df_jam.ffill(inplace=True)
    return df_jam

def eksplorasi_data():
    print("\n[TAHAP 3] Eksplorasi Data (Exploratory Data Analysis / EDA)")
    print("Menganalisis korelasi variabel dan tren musiman...")


def hitung_metrik(y_asli, y_prediksi):
    mae = mean_absolute_error(y_asli, y_prediksi)
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mape = np.mean(np.abs((y_asli - y_prediksi) / y_asli)) * 100
    r2 = r2_score(y_asli, y_prediksi)
    return mae, rmse, mape, r2

def latih_lstm(df_jam, window_kebelakang=24, iterasi_epoch=10, ukuran_batch=32):
    print("\n[TAHAP 4A] Pemodelan dan Pelatihan Algoritma LSTM (Model Training)")
    data = df_jam[['Global_active_power']].values

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_ternormalisasi = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_kebelakang, len(data_ternormalisasi)):
        X.append(data_ternormalisasi[i-window_kebelakang:i, 0])
        y.append(data_ternormalisasi[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Pembagian data latih/uji (80/20)
    batas = int(0.8 * len(X))
    X_latih, X_uji = X[:batas], X[batas:]
    y_latih, y_uji = y[:batas], y[batas:]

    # Arsitektur Jaringan Saraf Tiruan (Deep Learning)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_latih.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Melatih model LSTM...")
    model.fit(X_latih, y_latih, batch_size=ukuran_batch, epochs=iterasi_epoch,
              validation_split=0.1, verbose=1)

    print("Melakukan prediksi data uji...")
    prediksi_lstm = model.predict(X_uji)

    # Mengembalikan skala nilai ke bentuk kW asli
    prediksi_lstm = scaler.inverse_transform(prediksi_lstm)
    y_uji_asli = scaler.inverse_transform(y_uji.reshape(-1, 1))

    mae, rmse, mape, r2 = hitung_metrik(y_uji_asli, prediksi_lstm)
    print(f"\n[Hasil Evaluasi LSTM]:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

    # Pembuatan Grafik visual
    plt.figure(figsize=(12, 6))
    plt.plot(y_uji_asli[:200], label='Data Aktual', color='blue', alpha=0.6)
    plt.plot(prediksi_lstm[:200], label='Prediksi', color='red', alpha=0.8)
    plt.title('Prediksi LSTM vs Aktual (200 Jam Pertama)')
    plt.xlabel('Jam (Hours)')
    plt.ylabel('Daya Aktif (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('grafik_hasil_lstm.png', bbox_inches='tight')
    print("--> Grafik disimpan sebagai: grafik_hasil_lstm.png")
    
    plt.show()
    plt.close()

    return {
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2
    }

def latih_prophet(df_jam):
    print("\n[TAHAP 4B] Pemodelan dan Pelatihan Algoritma Prophet (Model Training)")
    # Prophet difokuskan pada analisis tren harian
    df_harian = df_jam.resample('D').mean().reset_index()

    df_prophet = pd.DataFrame({
        'ds': df_harian['datetime'],
        'y': df_harian['Global_active_power']
    })

    batas = int(0.8 * len(df_prophet))
    data_latih = df_prophet.iloc[:batas]
    data_uji = df_prophet.iloc[batas:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    print("Melatih model Prophet...")
    model.fit(data_latih)

    print("Melakukan peramalan data uji...")
    future = model.make_future_dataframe(periods=len(data_uji))
    forecast = model.predict(future)

    prediksi_prophet = forecast['yhat'].iloc[batas:].values
    y_uji_asli = data_uji['y'].values

    mae, rmse, mape, r2 = hitung_metrik(y_uji_asli, prediksi_prophet)
    print(f"\n[Hasil Evaluasi Prophet]:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_uji_asli, label='Data Aktual', color='blue', alpha=0.6)
    plt.plot(prediksi_prophet, label='Prediksi', color='green', alpha=0.8)
    plt.title('Prediksi Prophet vs Aktual (Siklus Harian)')
    plt.xlabel('Hari (Days)')
    plt.ylabel('Daya Aktif (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('grafik_hasil_prophet.png', bbox_inches='tight')
    print("--> Grafik disimpan sebagai: grafik_hasil_prophet.png")
    
    plt.show()
    plt.close()

    return {
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2
    }

def utama():
    print("=====================================================")
    print("   PREDIKSI KONSUMSI LISTRIK: LSTM vs PROPHET")
    print("=====================================================\n")

    if not os.path.exists('data/household_power_consumption.txt'):
        print("Error: Dataset tidak ditemukan di 'data/household_power_consumption.txt'.")
        print("Pastikan dataset sudah di-download.")
        return

    df_raw = muat_data()
    df_jam = praproses_data(df_raw)
    
    eksplorasi_data()

    # Epoch diatur ke 10 sebagai nilai wajar untuk CLI
    hasil_lstm = latih_lstm(df_jam, iterasi_epoch=10)
    hasil_prophet = latih_prophet(df_jam)

    print("\n[TAHAP 5] Evaluasi dan Kesimpulan (Evaluation & Conclusion)")
    print("=======================================================")
    print("              KESIMPULAN PERBANDINGAN MODEL")
    print("=======================================================")
    print(f"{'Metrik':<10} | {'LSTM (Per Jam)':<18} | {'Prophet (Per Hari)':<18}")
    print("-" * 55)
    print(f"{'MAE':<10} | {hasil_lstm['mae']:<18.4f} | {hasil_prophet['mae']:<18.4f}")
    print(f"{'RMSE':<10} | {hasil_lstm['rmse']:<18.4f} | {hasil_prophet['rmse']:<18.4f}")
    print(f"{'MAPE':<10} | {hasil_lstm['mape']:<17.2f}% | {hasil_prophet['mape']:<17.2f}%")
    print(f"{'Skor R2':<10} | {hasil_lstm['r2']:<18.4f} | {hasil_prophet['r2']:<18.4f}")
    print("=======================================================")

if __name__ == '__main__':
    utama()
