# ⚡ Prediksi Konsumsi Energi Listrik Rumah Tangga

*Baca dalam bahasa [Inggris](README.md)*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Meta-0467DF?style=for-the-badge&logo=meta&logoColor=white)

Repositori ini memuat hasil penelitian dan eksperimen komparatif antara algoritma **Jaringan Saraf Tiruan (LSTM)** dan algoritma **Regresi Waktu (Facebook Prophet)** dalam memprediksi konsumsi daya listrik skala rumah tangga. 

Penelitian ini ditujukan sebagai pemenuhan **Ujian Akhir Semester (UAS)** mata kuliah Big Data and Visualization pada Program Studi Magister Teknik Informatika, Universitas Pamulang.

---

## 📋 Metodologi Penelitian

Penelitian ini menggunakan dataset **[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)** dari UCI Machine Learning Repository yang merekam konsumsi listrik per-menit selama hampir 4 tahun. 

Proses penelitian analisis data (Data Science Lifecycle) dilakukan melalui tahapan yang sistematis dan eksplisit dalam kode:

1. **[TAHAP 1] Pengumpulan dan Pemuatan Data (Data Loading):**
   - Mengimpor dataset mentah `household_power_consumption.txt` berukuran jutaan baris dan mem-*parsing* atribut tanggal & waktu menjadi indeks basis waktu (Datetime Index).
   
2. **[TAHAP 2] Pra-Pemrosesan Data (Data Preprocessing):**
   - **Data Cleaning:** Menangani dan mengimputasi *missing values* (NaN) menggunakan metode *forward-fill* agar kontinuitas deret waktu tidak terputus.
   - **Data Transformation (Resampling):** Mengagregasikan resolusi data dari per-menit menjadi skala per-jam (Hourly) untuk menangkap pola jangka pendek.

3. **[TAHAP 3] Eksplorasi Data (Exploratory Data Analysis / EDA):**
   - Menganalisis korelasi variabel (Kuat Arus vs Daya Aktif) dan mengidentifikasi pola temporal (jam sibuk harian dan lonjakan akhir pekan).
   
4. **[TAHAP 4] Pemodelan dan Pelatihan Algoritma (Model Training):** 
   - **[4A] LSTM (Deep Learning):** Membentuk struktur jaringan saraf rekuren dengan memanfaatkan urutan jendela (*lookback window*) 24 jam ke belakang. Data dinormalisasi secara ketat ke skala 0-1.
   - **[4B] Facebook Prophet:** Melakukan *resampling* lebih jauh menjadi skala harian (Daily) untuk memfokuskan Prophet pada penangkapan tren siklus musiman panjang (*weekly & yearly seasonality*).
   
5. **[TAHAP 5] Evaluasi dan Kesimpulan (Evaluation & Conclusion):**
   - Mengukur margin kesalahan kedua model secara objektif menggunakan metrik regresi standar (MAE, RMSE, MAPE) serta koefisien penjelasan varians (R²). Tabel komparasi dicetak untuk penarikan kesimpulan strategis.

---

## 🔬 Hasil Eksplorasi Data (EDA)

Sebelum permodelan, eksplorasi dilakukan untuk memahami karakteristik kelistrikan rumah tangga tersebut.
Berdasarkan data historis 4 tahun, tren konsumsi memiliki pola musiman yang dipengaruhi oleh aktivitas penghuni dan kemungkinan perubahan suhu/musim.

<img src="img/Beranda%20-%20Tren%20Konsumsi%20Energi%20Listrik%20Harian%20(kW).png" width="800">

### Pola Konsumsi Temporal
Secara spesifik, ditemukan pola perilaku penghuni yang sangat konsisten:
- **Pola Harian:** Puncak konsumsi listrik selalu terjadi di malam hari antara pukul **18:00 hingga 21:00**, bertepatan dengan waktu aktif penghuni rumah yang kembali dari rutinitas harian.
- **Pola Mingguan:** Terjadi peningkatan penggunaan daya pada akhir pekan (**Sabtu dan Minggu**) dibandingkan hari kerja, mengindikasikan tingginya aktivitas di dalam rumah pada hari libur.

<p float="left">
  <img src="img/EDA%20-%20Pola%20Jam.png" width="400" />
  <img src="img/EDA%20-%20Pola%20Mingguan.png" width="400" />
</p>

### Korelasi Variabel
Uji korelasi Pearson menunjukkan bahwa `Global_active_power` (Daya Aktif) memiliki korelasi linier sempurna (0.99) terhadap `Global_intensity` (Kuat Arus), namun berbanding terbalik dengan nilai tegangan (*Voltage*).

<p float="left">
  <img src="img/EDA%20-%20Heatmap.png" width="400" />
  <img src="img/EDA%20-%20Korelasi.png" width="400" /> 
</p>

---

## 📈 Hasil Penelitian & Evaluasi Model

Eksperimen komparatif menghasilkan temuan kuantitatif yang sangat menarik terkait karakteristik kedua algoritma saat dihadapkan pada volatilitas data kelistrikan.

### 1. Kinerja Facebook Prophet (*Additive Regression*)
Model Prophet, yang dilatih pada agregasi harian (*daily*), menunjukkan performa tingkat galat (*error*) yang sangat memuaskan:
- **MAE:** 0.2007 kW
- **RMSE:** 0.2741 kW
- **MAPE:** 27.39%
- **R² Score:** 0.2422

Prophet terbukti sangat tangguh dalam memetakan tren dasar (baseline) dan mendekomposisi efek siklus mingguan serta tahunan. Dengan meratakan (*smoothing*) fluktuasi menit-ke-menit menjadi data harian, Prophet tidak terdistraksi oleh lonjakan mendadak (*noise*), sehingga menghasilkan nilai *error* absolut (MAE) dan persentase (MAPE) yang paling rendah.

<img src="img/Hasil%20-%20Prophet%20-%20Aktual%20vs%20Prediksi.png" width="800">
<img src="img/Hasil%20-%20Prophet%20-%20Scatter%20Plot.png" width="400">


### 2. Kinerja Model LSTM (*Deep Learning*)
Di sisi lain, model LSTM yang dilatih secara mikroskopis menggunakan data per-jam (*hourly*) dengan *window* 24 jam menghasilkan evaluasi sebagai berikut:
- **MAE:** 0.3606 kW
- **RMSE:** 0.5113 kW
- **MAPE:** 51.50%
- **R² Score:** 0.5103

Secara nilai galat (MAE, RMSE, MAPE), LSTM menghasilkan *error* yang lebih besar dibandingkan Prophet. Hal ini wajar secara keilmuan data karena memprediksi pergerakan tiap jam di rumah tangga (seperti menyalanya *microwave* atau pemanas air secara tiba-tiba) jauh lebih sulit dan fluktuatif dibandingkan memprediksi rata-rata satu hari penuh. 
Namun, menariknya LSTM meraih **skor R² yang jauh lebih tinggi (0.5103 vs 0.2422)**. Ini membuktikan bahwa LSTM jauh lebih baik dalam menjelaskan porsi varians dari data dan mengikuti bentuk fluktuasi data secara presisi (seperti terlihat pada kurva yang menempel pada data aktual), meskipun terkadang jarak mutlak (*absolute error*) tebakannya meleset pada saat terjadi lonjakan ekstrem.

<img src="img/Hasil%20-%20LTSM%20-%20Aktual%20vs%20Prediksi.png" width="800">
<p float="left">
  <img src="img/Hasil%20-%20LTSM%20-%20Aktual%20vs%20Prediksi%20Scatter%20Plot.png" width="400" />
  <img src="img/Hasil%20-%20LTSM%20-%20Training.png" width="400" />
</p>

---

## 🏆 Kesimpulan Penelitian

Berdasarkan hasil eksperimen, **Facebook Prophet ditetapkan sebagai model pemenang** karena berhasil memenangkan 3 dari 4 metrik evaluasi utama (menghasilkan MAE, RMSE, dan MAPE yang lebih rendah dibandingkan LSTM).

**Sintesis Ilmiah:**
Penelitian ini membuktikan prinsip penting dalam *Time-Series Forecasting*:
1. **Untuk prediksi peramalan beban dasar (*Baseline Load/Macro Trend*):** Prophet lebih superior. Agregasi harian meredam sifat *chaotic* (acak) dari konsumsi listrik individu, memungkinkan Prophet memetakan musiman secara akurat dengan tingkat *error* yang minim (MAPE 27%).
2. **Untuk pemodelan dinamika mikroskopis (*Micro Fluctuation*):** Meskipun LSTM kalah dalam kompetisi metrik absolut, nilai R² yang jauh melebihi Prophet (0.51 vs 0.24) menunjukkan bahwa *Deep Learning* sangat andal dalam menangkap bentuk kelokan data (*pattern shape*) dalam jangka pendek (per-jam), yang gagal ditangkap oleh perataan kurva regresi linier biasa.

Kesimpulannya, pemilihan algoritma untuk *smart grid* kelistrikan sangat bergantung pada tujuan bisnis: gunakan Prophet untuk estimasi kapasitas pasokan harian/bulanan pembangkit, dan gunakan LSTM untuk sistem peringatan anomali lonjakan daya sesaat (*real-time*).

---

## ⚙️ Reproduksi Eksperimen (Menjalankan Aplikasi)

Penelitian ini menyediakan dua cara untuk menjalankan eksperimen: melalui terminal biasa atau melalui antarmuka web interaktif (Streamlit).

1. Kloning repositori:
   ```bash
   git clone https://github.com/achmadhadikurnia/uas-big-data-and-visualization.git
   cd uas-big-data-and-visualization
   ```
2. Instal pustaka (*libraries*) yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. **Pilihan 1: Menjalankan versi Terminal**
   ```bash
   python app.py
   ```
   *Catatan: Hasil analisis akan dicetak ke layar terminal, dan grafik akan disimpan sebagai file gambar `.png`.*

4. **Pilihan 2: Menjalankan versi Web Dashboard**
   ```bash
   streamlit run app_streamlit.py
   ```
