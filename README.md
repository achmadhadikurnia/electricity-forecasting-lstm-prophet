# ⚡ Household Electricity Consumption Prediction

*Read this in [Indonesian](README_id.md)*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Meta-0467DF?style=for-the-badge&logo=meta&logoColor=white)

This repository contains the research findings and comparative experiments between **Artificial Neural Network (LSTM)** and **Time-Series Regression (Facebook Prophet)** algorithms in predicting household-scale active power consumption.

This research was conducted as a final project (UAS) for the Big Data and Visualization course in the Master of Informatics Engineering program at Universitas Pamulang.

---

## 📋 Research Methodology

This research utilizes the **[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)** dataset from the UCI Machine Learning Repository, which records per-minute electricity consumption over a period of almost 4 years.

The Data Science Lifecycle research process is conducted through systematic and explicitly coded stages:

1. **[STAGE 1] Data Collection & Loading:**
   - Importing the raw `household_power_consumption.txt` dataset containing millions of rows and parsing the date & time attributes into a proper Datetime Index.
   
2. **[STAGE 2] Data Preprocessing:**
   - **Data Cleaning:** Handling and imputing missing values (NaN) using the forward-fill method to preserve time-series continuity.
   - **Data Transformation (Resampling):** Aggregating the data resolution from per-minute to an Hourly scale to capture short-term patterns.

3. **[STAGE 3] Exploratory Data Analysis (EDA):**
   - Analyzing variable correlations (e.g. Current vs Active Power) and identifying temporal patterns (daily peak hours and weekend surges).
   
4. **[STAGE 4] Algorithm Modeling & Training:** 
   - **[4A] LSTM (Deep Learning):** Building a recurrent neural network structure utilizing a 24-hour lookback window. Data is strictly normalized to a 0-1 scale.
   - **[4B] Facebook Prophet:** Further resampling the data into a Daily scale to focus Prophet on capturing long-term seasonal trends (weekly & yearly seasonality).
   
5. **[STAGE 5] Evaluation & Conclusion:**
   - Objectively measuring the error margins of both models using standard regression metrics (MAE, RMSE, MAPE) and the variance explanation coefficient (R²). A comparison table is printed for strategic conclusions.

---

## 🔬 Exploratory Data Analysis (EDA)

Before modeling, an exploration was conducted to understand the household's electricity characteristics. Based on 4 years of historical data, the consumption trend exhibits a seasonal pattern influenced by occupant activity and potentially seasonal temperature changes.

<img src="img/Beranda%20-%20Tren%20Konsumsi%20Energi%20Listrik%20Harian%20(kW).png" width="800">

### Temporal Consumption Patterns
Specifically, highly consistent occupant behavioral patterns were discovered:
- **Daily Pattern:** Electricity consumption peaks always occur at night between **18:00 and 21:00**, coinciding with the active hours of the occupants returning from their daily routines.
- **Weekly Pattern:** There is a significant increase in power usage on weekends (**Saturday and Sunday**) compared to weekdays, indicating higher activity inside the house on days off.

<p float="left">
  <img src="img/EDA%20-%20Pola%20Jam.png" width="400" />
  <img src="img/EDA%20-%20Pola%20Mingguan.png" width="400" />
</p>

### Variable Correlation
Pearson correlation testing shows that `Global_active_power` (Active Power) has a nearly perfect linear correlation (0.99) with `Global_intensity` (Current), but is inversely proportional to the `Voltage` value.

<p float="left">
  <img src="img/EDA%20-%20Heatmap.png" width="400" />
  <img src="img/EDA%20-%20Korelasi.png" width="400" /> 
</p>

---

## 📈 Research Results & Model Evaluation

The comparative experiment yielded highly interesting quantitative findings regarding the characteristics of both algorithms when faced with electricity data volatility.

### 1. Facebook Prophet Performance (Additive Regression)
The Prophet model, trained on daily aggregations, demonstrated highly satisfactory error performance:
- **MAE:** 0.2007 kW
- **RMSE:** 0.2741 kW
- **MAPE:** 27.39%
- **R² Score:** 0.2422

Prophet proved to be highly robust in mapping the baseline trend and decomposing weekly and yearly seasonal effects. By smoothing the minute-by-minute fluctuations into daily data, Prophet was not distracted by sudden spikes (noise), resulting in the lowest absolute error (MAE) and percentage error (MAPE).

<img src="img/Hasil%20-%20Prophet%20-%20Aktual%20vs%20Prediksi.png" width="800">
<img src="img/Hasil%20-%20Prophet%20-%20Scatter%20Plot.png" width="400">

### 2. LSTM Performance (Deep Learning)
On the other hand, the LSTM model, trained microscopically using hourly data with a 24-hour window, produced the following evaluation:
- **MAE:** 0.3606 kW
- **RMSE:** 0.5113 kW
- **MAPE:** 51.50%
- **R² Score:** 0.5103

In terms of error values (MAE, RMSE, MAPE), LSTM produced a larger error compared to Prophet. This is scientifically expected because predicting hourly movements in a household (such as a microwave or water heater suddenly turning on) is much more difficult and volatile compared to predicting a full-day average. 
However, interestingly, LSTM achieved a **much higher R² score (0.5103 vs 0.2422)**. This proves that LSTM is far better at explaining the proportion of data variance and precisely following the shape of data fluctuations (as seen in the curve hugging the actual data), even though its absolute guesses sometimes miss during extreme spikes.

<img src="img/Hasil%20-%20LTSM%20-%20Aktual%20vs%20Prediksi.png" width="800">
<p float="left">
  <img src="img/Hasil%20-%20LTSM%20-%20Aktual%20vs%20Prediksi%20Scatter%20Plot.png" width="400" />
  <img src="img/Hasil%20-%20LTSM%20-%20Training.png" width="400" />
</p>

---

## 🏆 Conclusion

Based on the experimental results, **Facebook Prophet is designated as the winning model** because it successfully won 3 out of the 4 main evaluation metrics (producing lower MAE, RMSE, and MAPE compared to LSTM).

**Scientific Synthesis:**
This research proves an important principle in Time-Series Forecasting:
1. **For baseline load prediction (Macro Trend):** Prophet is superior. Daily aggregation dampens the chaotic nature of individual electricity consumption, allowing Prophet to map seasonality accurately with minimal error (MAPE 27%).
2. **For modeling microscopic dynamics (Micro Fluctuation):** Although LSTM lost in absolute metric competitions, its R² value, which far exceeds Prophet's (0.51 vs 0.24), demonstrates that Deep Learning is highly reliable in capturing pattern shapes in the short term (hourly), which ordinary linear regression curves fail to catch.

In conclusion, the choice of algorithm for smart grids depends heavily on business objectives: use Prophet for estimating daily/monthly power plant supply capacity, and use LSTM for real-time anomalous power spike warning systems.

---

## ⚙️ Reproducing the Experiment (Running the Application)

This research provides two ways to run the experiment: via a standard terminal or through an interactive web interface (Streamlit).

1. Clone the repository:
   ```bash
   git clone https://github.com/achmadhadikurnia/uas-big-data-and-visualization.git
   cd uas-big-data-and-visualization
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. **Option 1: Running the Terminal version**
   ```bash
   python app.py
   ```
   *Note: The analysis results will be printed to the terminal screen, and the charts will be saved as `.png` image files.*

4. **Option 2: Running the Web Dashboard version**
   ```bash
   streamlit run app_streamlit.py
   ```
