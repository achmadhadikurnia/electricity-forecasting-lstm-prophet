"""
=============================================================================
Prediksi Konsumsi Energi Listrik Rumah Tangga
Perbandingan Algoritma LSTM dan Prophet
=============================================================================
Streamlit Dashboard untuk UAS Big Data and Visualization
Dataset: UCI - Individual Household Electric Power Consumption
Author: Achmad Hadi Kurnia
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
import pickle
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Prediksi Konsumsi Energi Listrik",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        background: var(--secondary-background-color);
        padding: 2rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: var(--text-color);
        border: 1px solid var(--primary-color);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--primary-color);
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Metric cards */
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-card h3 {
        color: var(--primary-color);
        font-size: 1.8rem;
        margin: 0;
    }
    .metric-card p {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
    }

    /* Section headers */
    .section-header {
        background: var(--secondary-background-color);
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 1.5rem 0 1rem 0;
        color: var(--text-color);
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Model comparison */
    .model-winner {
        background: var(--primary-color);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.1rem;
        margin: 1rem 0;
    }

    /* Info boxes */
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the household power consumption dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "household_power_consumption.txt")

    if not os.path.exists(data_path):
        st.error(f"❌ Dataset tidak ditemukan di: {data_path}")
        st.info("Jalankan `python download_dataset.py` terlebih dahulu")
        st.stop()

    # Load dataset
    df = pd.read_csv(
        data_path,
        sep=';',
        low_memory=False,
        na_values=['?']
    )

    # Combine Date and Time into a single datetime column manually
    # Pandas 3.0+ no longer supports combining columns via parse_dates in read_csv
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

    return df


@st.cache_data
def preprocess_data(df):
    """Preprocess: handle missing values, feature engineering, resampling."""
    df = df.copy()

    # Set datetime as index
    df.set_index('datetime', inplace=True)

    # Convert to numeric
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values - forward fill then backward fill
    df = df.ffill().bfill()

    # Drop string columns (Date and Time) before resampling since they are in the index now
    if 'Date' in df.columns:
        df = df.drop(columns=['Date', 'Time'])

    # Resample to hourly (from per-minute) for more manageable size
    # In Pandas 3.0+, we must explicitly ignore or drop non-numeric columns when calculating mean()
    df_hourly = df.resample('h').mean(numeric_only=True)

    # Feature engineering
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['day_of_week'] = df_hourly.index.dayofweek
    df_hourly['month'] = df_hourly.index.month
    df_hourly['day_of_year'] = df_hourly.index.dayofyear
    df_hourly['is_weekend'] = (df_hourly.index.dayofweek >= 5).astype(int)
    df_hourly['quarter'] = df_hourly.index.quarter

    # Total sub metering
    df_hourly['total_sub_metering'] = (df_hourly['Sub_metering_1'] +
                                        df_hourly['Sub_metering_2'] +
                                        df_hourly['Sub_metering_3'])

    # Remove any remaining NaN
    df_hourly = df_hourly.dropna()

    return df_hourly


@st.cache_data
def resample_daily(df_hourly):
    """Resample to daily for Prophet model."""
    df_daily = df_hourly[['Global_active_power']].resample('D').mean()
    df_daily = df_daily.dropna()
    return df_daily


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================
@st.cache_resource
def train_lstm_model(df_hourly, lookback=24, epochs=20, batch_size=32):
    """Train LSTM model for energy consumption prediction."""
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        st.error("TensorFlow tidak terinstall. Jalankan: pip install tensorflow")
        return None

    # Prepare data
    target = df_hourly['Global_active_power'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split: 80% train, 20% test (chronological)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)

    # MAPE (avoid division by zero)
    mask = y_test_inv != 0
    mape = np.mean(np.abs((y_test_inv[mask] - y_pred_inv[mask]) / y_test_inv[mask])) * 100

    # Get datetime index for test data
    test_index = df_hourly.index[lookback + split:lookback + split + len(y_test)]

    results = {
        'model_name': 'LSTM',
        'y_test': y_test_inv,
        'y_pred': y_pred_inv,
        'test_index': test_index,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'history': history.history,
        'scaler': scaler,
        'model': model,
        'lookback': lookback,
        'architecture': f"LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)",
        'params': {
            'lookback': lookback,
            'epochs': len(history.history['loss']),
            'batch_size': batch_size,
            'optimizer': 'Adam',
            'loss_function': 'MSE'
        }
    }

    return results


@st.cache_resource
def train_prophet_model(df_hourly):
    """Train Prophet model for energy consumption prediction."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        from prophet import Prophet
    except ImportError:
        st.error("Prophet tidak terinstall. Jalankan: pip install prophet")
        return None

    # Prepare daily data for Prophet
    df_daily = df_hourly[['Global_active_power']].resample('D').mean().dropna()

    # Prophet requires 'ds' and 'y' columns
    df_prophet = df_daily.reset_index()
    df_prophet.columns = ['ds', 'y']

    # Split: 80% train, 20% test (chronological)
    split = int(len(df_prophet) * 0.8)
    train_df = df_prophet[:split]
    test_df = df_prophet[split:]

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    model.fit(train_df)

    # Predict on test period
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)

    # Get predictions for test period only
    forecast_test = forecast.iloc[split:]
    y_test = test_df['y'].values
    y_pred = forecast_test['yhat'].values

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    results = {
        'model_name': 'Prophet',
        'y_test': y_test,
        'y_pred': y_pred,
        'test_index': test_df['ds'].values,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'model': model,
        'forecast': forecast,
        'train_df': train_df,
        'test_df': test_df,
        'components': forecast_test,
        'architecture': "Prophet (Additive/Multiplicative Seasonal Decomposition)",
        'params': {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_mode': 'multiplicative'
        }
    }

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_time_series(df_hourly):
    """Plot overall time series of energy consumption."""
    # Resample to daily for cleaner visualization
    daily = df_hourly['Global_active_power'].resample('D').mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values,
        mode='lines',
        name='Konsumsi Harian',
        line=dict(color='#00d2ff', width=1),
        fill='tozeroy',
        fillcolor='rgba(0,210,255,0.1)'
    ))

    # Add rolling average
    rolling = daily.rolling(window=30).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        mode='lines',
        name='Moving Average (30 hari)',
        line=dict(color='#ff6b6b', width=2.5)
    ))

    fig.update_layout(
        title="📈 Tren Konsumsi Energi Listrik Harian (kW)",
        xaxis_title="Tanggal",
        yaxis_title="Global Active Power (kW)",
        template="plotly",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig


def plot_distribution(df_hourly):
    """Plot distribution of energy consumption."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Distribusi Global Active Power", "Box Plot per Bulan"))

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df_hourly['Global_active_power'],
            nbinsx=60,
            marker_color='#00d2ff',
            opacity=0.7,
            name='Distribusi'
        ), row=1, col=1
    )

    # Box plot per month
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
              'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    for m in range(1, 13):
        data = df_hourly[df_hourly['month'] == m]['Global_active_power']
        fig.add_trace(
            go.Box(y=data, name=months[m-1], marker_color=px.colors.qualitative.Set3[m-1]),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly",
        height=400,
        showlegend=False
    )
    return fig


def plot_hourly_pattern(df_hourly):
    """Plot average consumption by hour."""
    hourly_avg = df_hourly.groupby('hour')['Global_active_power'].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker=dict(
            color=hourly_avg.values,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="kW")
        ),
        name='Rata-rata per Jam'
    ))

    fig.update_layout(
        title="🕐 Pola Konsumsi Rata-rata per Jam",
        xaxis_title="Jam",
        yaxis_title="Global Active Power (kW)",
        template="plotly",
        height=400,
        xaxis=dict(dtick=1)
    )
    return fig


def plot_weekly_pattern(df_hourly):
    """Plot average consumption by day of week."""
    days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    daily_avg = df_hourly.groupby('day_of_week')['Global_active_power'].mean()

    colors = ['#636EFA'] * 5 + ['#EF553B'] * 2  # Weekend highlighted

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=days,
        y=daily_avg.values,
        marker_color=colors,
        name='Rata-rata per Hari'
    ))

    fig.update_layout(
        title="📅 Pola Konsumsi Rata-rata per Hari",
        xaxis_title="Hari",
        yaxis_title="Global Active Power (kW)",
        template="plotly",
        height=400
    )
    return fig


def plot_heatmap(df_hourly):
    """Plot heatmap of consumption by hour and day."""
    pivot = df_hourly.pivot_table(
        values='Global_active_power',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'{h}:00' for h in range(24)],
        y=days,
        colorscale='Turbo',
        colorbar=dict(title="kW")
    ))

    fig.update_layout(
        title="🔥 Heatmap Konsumsi Energi (Jam × Hari)",
        xaxis_title="Jam",
        yaxis_title="Hari",
        template="plotly",
        height=400
    )
    return fig


def plot_correlation(df_hourly):
    """Plot correlation matrix."""
    cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    corr = df_hourly[cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="📊 Matriks Korelasi Antar Variabel",
        template="plotly",
        height=500,
        width=700
    )
    return fig


def plot_model_predictions(results, model_name):
    """Plot actual vs predicted values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(results['y_test']))),
        y=results['y_test'],
        mode='lines',
        name='Aktual',
        line=dict(color='#00d2ff', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(results['y_pred']))),
        y=results['y_pred'],
        mode='lines',
        name='Prediksi',
        line=dict(color='#ff6b6b', width=1.5)
    ))

    fig.update_layout(
        title=f"🎯 Hasil Prediksi {model_name}: Aktual vs Prediksi",
        xaxis_title="Data Point",
        yaxis_title="Global Active Power (kW)",
        template="plotly",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig


def plot_residuals(results, model_name):
    """Plot residual analysis."""
    residuals = results['y_test'] - results['y_pred']

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"Residual Plot {model_name}", f"Distribusi Residual {model_name}"))

    # Scatter residuals
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='markers',
            marker=dict(color='#00d2ff', size=3, opacity=0.5),
            name='Residual'
        ), row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Histogram residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=50,
            marker_color='#ff6b6b',
            opacity=0.7,
            name='Distribusi'
        ), row=1, col=2
    )

    fig.update_layout(
        template="plotly",
        height=380,
        showlegend=False
    )
    return fig


def plot_scatter_actual_vs_pred(results, model_name):
    """Scatter plot of actual vs predicted."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['y_test'],
        y=results['y_pred'],
        mode='markers',
        marker=dict(color='#00d2ff', size=4, opacity=0.3),
        name='Data Points'
    ))

    # Perfect prediction line
    min_val = min(results['y_test'].min(), results['y_pred'].min())
    max_val = max(results['y_test'].max(), results['y_pred'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Garis Ideal (y=x)'
    ))

    fig.update_layout(
        title=f"📐 Scatter Plot: Aktual vs Prediksi ({model_name})",
        xaxis_title="Nilai Aktual (kW)",
        yaxis_title="Nilai Prediksi (kW)",
        template="plotly",
        height=450
    )
    return fig


def plot_comparison_metrics(lstm_results, prophet_results):
    """Plot comparison of metrics between models."""
    metrics = ['MAE', 'RMSE', 'MAPE (%)', 'R² Score']
    lstm_vals = [lstm_results['mae'], lstm_results['rmse'], lstm_results['mape'], lstm_results['r2']]
    prophet_vals = [prophet_results['mae'], prophet_results['rmse'], prophet_results['mape'], prophet_results['r2']]

    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics)

    colors_lstm = '#00d2ff'
    colors_prophet = '#ff6b6b'

    for i, (metric, lv, pv) in enumerate(zip(metrics, lstm_vals, prophet_vals)):
        fig.add_trace(
            go.Bar(x=['LSTM'], y=[lv], marker_color=colors_lstm, name='LSTM', showlegend=(i==0)),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Bar(x=['Prophet'], y=[pv], marker_color=colors_prophet, name='Prophet', showlegend=(i==0)),
            row=1, col=i+1
        )

    fig.update_layout(
        title="⚔️ Perbandingan Metrik Evaluasi: LSTM vs Prophet",
        template="plotly",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig


def plot_training_history(history):
    """Plot LSTM training history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines',
        name='Training Loss', line=dict(color='#00d2ff')
    ))
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(
            y=history['val_loss'], mode='lines',
            name='Validation Loss', line=dict(color='#ff6b6b')
        ))

    fig.update_layout(
        title="📉 LSTM Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        template="plotly",
        height=380
    )
    return fig


# ============================================================================
# HELPER FUNCTIONS: MODEL PERSISTENCE
# ============================================================================
import pickle
import os

def save_trained_models(lstm, prophet):
    os.makedirs('models', exist_ok=True)
    # Save LSTM (Keras model must be saved separately)
    lstm_model = lstm['model']
    lstm_model.save('models/lstm_model.keras')

    # Save LSTM metadata
    lstm['model'] = None # Remove before pickle
    with open('models/lstm_meta.pkl', 'wb') as f:
        pickle.dump(lstm, f)
    lstm['model'] = lstm_model # Restore

    # Save Prophet
    with open('models/prophet_full.pkl', 'wb') as f:
        pickle.dump(prophet, f)

def load_trained_models():
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists('models/lstm_model.keras') and os.path.exists('models/prophet_full.pkl'):
            # Load LSTM
            with open('models/lstm_meta.pkl', 'rb') as f:
                lstm = pickle.load(f)
            lstm['model'] = load_model('models/lstm_model.keras')

            # Load Prophet
            with open('models/prophet_full.pkl', 'rb') as f:
                prophet = pickle.load(f)

            return lstm, prophet
    except Exception as e:
        return None, None
    return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.title("⚡ Prediksi Konsumsi Energi Listrik Rumah Tangga")
    st.markdown("**Perbandingan Algoritma LSTM dan Prophet | UAS Big Data and Visualization**")
    st.caption("Dataset: UCI - Individual Household Electric Power Consumption")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/electricity.png", width=80)
        st.markdown("---")

        page = st.radio(
            "Navigasi",
            ["🏠 Beranda",
             "📊 Explorasi Data (EDA)",
             "🧠 Training Model",
             "📈 Hasil Prediksi",
             "🏆 Perbandingan Model",
             "📖 Tentang Penelitian"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📋 Info Dataset")
        st.markdown("""
        - **Sumber:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
        - **Periode:** Des 2006 - Nov 2010
        - **Resolusi:** Per menit → hourly
        - **Target:** Global Active Power (kW)
        """)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; font-size: 0.8rem;">
            UAS Big Data & Visualization<br>
            Magister Teknik Informatika<br>
            Universitas Pamulang
        </div>
        """, unsafe_allow_html=True)

    # Load data
    with st.spinner("📥 Memuat dataset..."):
        df_raw = load_data()
        df_hourly = preprocess_data(df_raw)

    # Auto-load models if they exist
    if 'lstm_results' not in st.session_state or 'prophet_results' not in st.session_state:
        l_res, p_res = load_trained_models()
        if l_res and p_res:
            st.session_state['lstm_results'] = l_res
            st.session_state['prophet_results'] = p_res

    # ========================================================================
    # PAGE: BERANDA
    # ========================================================================
    if page == "🏠 Beranda":
        st.subheader("📋 Ringkasan Dataset", divider="red")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data Mentah (per menit)", f"{len(df_raw):,}")
        with col2:
            st.metric("Data Setelah Resampling (per jam)", f"{len(df_hourly):,}")
        with col3:
            st.metric("Jumlah Fitur", f"{df_hourly.shape[1]}")
        with col4:
            missing_pct = (df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100
            st.metric("Missing Values", f"{missing_pct:.2f}%")

        st.markdown("")

        # Time series plot
        st.plotly_chart(plot_time_series(df_hourly), use_container_width=True)

        # Dataset info
        st.subheader("📄 Deskripsi Variabel", divider="red")

        variable_info = pd.DataFrame({
            'Variabel': ['Global_active_power', 'Global_reactive_power', 'Voltage',
                         'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
            'Deskripsi': [
                'Daya aktif rata-rata per menit (kilowatt)',
                'Daya reaktif rata-rata per menit (kilowatt)',
                'Tegangan rata-rata per menit (volt)',
                'Intensitas arus rata-rata per menit (ampere)',
                'Sub-pengukuran energi #1 (dapur: mesin cuci piring, oven, microwave)',
                'Sub-pengukuran energi #2 (ruang cuci: mesin cuci, pengering, kulkas, lampu)',
                'Sub-pengukuran energi #3 (pemanas air listrik, AC)'
            ],
            'Satuan': ['kW', 'kW', 'Volt', 'Ampere', 'Watt-hour', 'Watt-hour', 'Watt-hour']
        })
        st.dataframe(variable_info, use_container_width=True, hide_index=True)

        # Preview data
        st.subheader("👀 Preview Data (Hourly)", divider="red")
        st.dataframe(df_hourly.head(20), use_container_width=True)

        # Statistics
        st.subheader("📊 Statistik Deskriptif", divider="red")
        st.dataframe(df_hourly.describe().round(3), use_container_width=True)

    # ========================================================================
    # PAGE: EDA
    # ========================================================================
    elif page == "📊 Explorasi Data (EDA)":
        st.subheader("📊 Exploratory Data Analysis (EDA)", divider="red")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Distribusi", "🕐 Pola Jam", "📅 Pola Mingguan", "🔥 Heatmap", "🔗 Korelasi"
        ])

        with tab1:
            st.plotly_chart(plot_distribution(df_hourly), use_container_width=True)
            st.info("""
            **Insight:** Distribusi konsumsi energi bersifat right-skewed (condong ke kanan),
            menunjukkan bahwa sebagian besar waktu konsumsi rendah, dengan beberapa periode puncak
            konsumsi tinggi.
            """)

        with tab2:
            st.plotly_chart(plot_hourly_pattern(df_hourly), use_container_width=True)
            st.info("""
            **Insight:** Terdapat pola diurnal yang jelas — konsumsi terendah pada pukul 01:00-06:00 (dini hari)
            dan puncak konsumsi pada pukul 18:00-21:00 (malam hari) ketika penghuni rumah beraktivitas.
            """)

        with tab3:
            st.plotly_chart(plot_weekly_pattern(df_hourly), use_container_width=True)
            st.info("""
            **Insight:** Konsumsi energi cenderung lebih tinggi pada akhir pekan (Sabtu-Minggu,
            ditandai warna merah) karena penghuni lebih banyak berada di rumah.
            """)

        with tab4:
            st.plotly_chart(plot_heatmap(df_hourly), use_container_width=True)
            st.info("""
            **Insight:** Heatmap menunjukkan bahwa puncak konsumsi terjadi pada jam 18:00-22:00
            sepanjang minggu, terutama di hari kerja. Akhir pekan menunjukkan konsumsi yang lebih
            merata sepanjang hari.
            """)

        with tab5:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_correlation(df_hourly), use_container_width=True)
            with col2:
                st.markdown("""
                ### 🔍 Interpretasi Korelasi

                - **Global_active_power ↔ Global_intensity**: Korelasi sangat tinggi (~0.99)
                  karena daya aktif berbanding lurus dengan arus
                - **Sub_metering_3 ↔ Global_active_power**: Korelasi tinggi, menunjukkan pemanas
                  air dan AC berkontribusi besar
                - **Voltage** memiliki korelasi negatif lemah dengan konsumsi daya
                """)

    # ========================================================================
    # PAGE: TRAINING MODEL
    # ========================================================================
    elif page == "🧠 Training Model":
        st.subheader("🧠 Training Model", divider="red")

        st.warning("⚠️ Training model membutuhkan waktu beberapa menit. Harap bersabar.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧠 LSTM (Long Short-Term Memory)")
            st.markdown("""
            **Arsitektur:**
            ```
            Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
            ```
            """)
            lookback = st.slider("Lookback Window (jam)", 12, 72, 24, 6)
            epochs = st.slider("Max Epochs", 10, 50, 20, 5)

        with col2:
            st.markdown("### 📊 Prophet (Facebook)")
            st.markdown("""
            **Konfigurasi:**
            ```
            Yearly Seasonality: ON
            Weekly Seasonality: ON
            Seasonality Mode: Multiplicative
            Changepoint Prior: 0.05
            ```
            """)
            st.markdown("*Data di-resample ke harian untuk Prophet*")

        st.markdown("---")

        if st.button("🚀 Mulai Training Kedua Model", type="primary", use_container_width=True):
            # Train LSTM
            with st.spinner("🧠 Training LSTM... (ini bisa memakan waktu beberapa menit)"):
                lstm_results = train_lstm_model(df_hourly, lookback=lookback, epochs=epochs)

            if lstm_results:
                st.success(f"✅ LSTM selesai! R² = {lstm_results['r2']:.4f}")
                st.session_state['lstm_results'] = lstm_results

            # Train Prophet
            with st.spinner("📊 Training Prophet..."):
                prophet_results = train_prophet_model(df_hourly)

            if prophet_results:
                st.success(f"✅ Prophet selesai! R² = {prophet_results['r2']:.4f}")
                st.session_state['prophet_results'] = prophet_results

            # Save to disk
            if lstm_results and prophet_results:
                with st.spinner("💾 Menyimpan model ke penyimpanan permanen..."):
                    save_trained_models(lstm_results, prophet_results)

            st.balloons()
            st.success("🎉 Kedua model berhasil di-training dan disimpan permanen! Buka tab 'Hasil Prediksi' untuk melihat hasil.")

        st.markdown("---")
        col_load, col_force = st.columns(2)
        with col_load:
            if st.button("📂 Muat Model Tersimpan (Cepat)", use_container_width=True):
                with st.spinner("Memuat model dari penyimpanan..."):
                    l_res, p_res = load_trained_models()
                    if l_res and p_res:
                        st.session_state['lstm_results'] = l_res
                        st.session_state['prophet_results'] = p_res
                        st.success("✅ Model berhasil dimuat! Silakan ke halaman Hasil Prediksi.")
                    else:
                        st.error("❌ Belum ada model tersimpan atau file corrupt. Silakan mulai training.")

        with col_force:
            if st.button("♻️ Paksa Hapus Cache (Training Ulang)", use_container_width=True):
                st.cache_resource.clear()
                st.cache_data.clear()
                import shutil
                if os.path.exists('models'):
                    shutil.rmtree('models')
                st.success("✅ Cache dan model tersimpan berhasil dihapus! Silakan klik 'Mulai Training' lagi jika ingin.")

    # ========================================================================
    # PAGE: HASIL PREDIKSI
    # ========================================================================
    elif page == "📈 Hasil Prediksi":
        st.subheader("📈 Hasil Prediksi", divider="red")

        if 'lstm_results' not in st.session_state or 'prophet_results' not in st.session_state:
            st.warning("⚠️ Model belum di-training. Silakan training model terlebih dahulu di halaman '🤖 Training Model'.")
            st.stop()

        lstm_results = st.session_state['lstm_results']
        prophet_results = st.session_state['prophet_results']

        model_choice = st.selectbox("Pilih Model:", ["LSTM", "Prophet", "Keduanya"])

        if model_choice in ["LSTM", "Keduanya"]:
            st.markdown("### 🧠 LSTM Results")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{lstm_results['mae']:.4f} kW")
            c2.metric("RMSE", f"{lstm_results['rmse']:.4f} kW")
            c3.metric("MAPE", f"{lstm_results['mape']:.2f}%")
            c4.metric("R² Score", f"{lstm_results['r2']:.4f}")

            # Predictions plot
            st.plotly_chart(plot_model_predictions(lstm_results, "LSTM"), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_scatter_actual_vs_pred(lstm_results, "LSTM"), use_container_width=True)
            with col2:
                st.plotly_chart(plot_residuals(lstm_results, "LSTM"), use_container_width=True)

            # Training history
            if 'history' in lstm_results:
                st.plotly_chart(plot_training_history(lstm_results['history']), use_container_width=True)

        if model_choice in ["Prophet", "Keduanya"]:
            st.markdown("### 📊 Prophet Results")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{prophet_results['mae']:.4f} kW")
            c2.metric("RMSE", f"{prophet_results['rmse']:.4f} kW")
            c3.metric("MAPE", f"{prophet_results['mape']:.2f}%")
            c4.metric("R² Score", f"{prophet_results['r2']:.4f}")

            # Predictions plot
            st.plotly_chart(plot_model_predictions(prophet_results, "Prophet"), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_scatter_actual_vs_pred(prophet_results, "Prophet"), use_container_width=True)
            with col2:
                st.plotly_chart(plot_residuals(prophet_results, "Prophet"), use_container_width=True)

    # ========================================================================
    # PAGE: PERBANDINGAN
    # ========================================================================
    elif page == "🏆 Perbandingan Model":
        st.subheader("🏆 Perbandingan Model: LSTM vs Prophet", divider="red")

        if 'lstm_results' not in st.session_state or 'prophet_results' not in st.session_state:
            st.warning("⚠️ Model belum di-training. Silakan training model terlebih dahulu.")
            st.stop()

        lstm_results = st.session_state['lstm_results']
        prophet_results = st.session_state['prophet_results']

        # Comparison table
        comparison_df = pd.DataFrame({
            'Metrik': ['MAE (kW)', 'RMSE (kW)', 'MAPE (%)', 'R² Score'],
            'LSTM': [
                f"{lstm_results['mae']:.4f}",
                f"{lstm_results['rmse']:.4f}",
                f"{lstm_results['mape']:.2f}",
                f"{lstm_results['r2']:.4f}"
            ],
            'Prophet': [
                f"{prophet_results['mae']:.4f}",
                f"{prophet_results['rmse']:.4f}",
                f"{prophet_results['mape']:.2f}",
                f"{prophet_results['r2']:.4f}"
            ],
            'Pemenang': [
                'LSTM ✅' if lstm_results['mae'] < prophet_results['mae'] else 'Prophet ✅',
                'LSTM ✅' if lstm_results['rmse'] < prophet_results['rmse'] else 'Prophet ✅',
                'LSTM ✅' if lstm_results['mape'] < prophet_results['mape'] else 'Prophet ✅',
                'LSTM ✅' if lstm_results['r2'] > prophet_results['r2'] else 'Prophet ✅'
            ]
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Visual comparison
        st.plotly_chart(plot_comparison_metrics(lstm_results, prophet_results), use_container_width=True)

        # Determine overall winner
        lstm_wins = sum([
            lstm_results['mae'] < prophet_results['mae'],
            lstm_results['rmse'] < prophet_results['rmse'],
            lstm_results['mape'] < prophet_results['mape'],
            lstm_results['r2'] > prophet_results['r2']
        ])

        winner = "LSTM" if lstm_wins >= 3 else "Prophet"
        winner_emoji = "🧠" if winner == "LSTM" else "📊"

        st.success(f"### {winner_emoji} Model Terbaik: {winner}  \nMemenangkan {max(lstm_wins, 4-lstm_wins)} dari 4 metrik evaluasi")

        # Detailed analysis
        st.subheader("📝 Analisis Detail", divider="red")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧠 LSTM")
            st.markdown(f"""
            - **Arsitektur:** {lstm_results['architecture']}
            - **Lookback:** {lstm_results['params']['lookback']} jam
            - **Epochs:** {lstm_results['params']['epochs']}
            - **Optimizer:** {lstm_results['params']['optimizer']}
            - **Data:** Hourly (per jam)
            """)

        with col2:
            st.markdown("### 📊 Prophet")
            st.markdown(f"""
            - **Arsitektur:** {prophet_results['architecture']}
            - **Seasonality Mode:** {prophet_results['params']['seasonality_mode']}
            - **Yearly Seasonality:** {prophet_results['params']['yearly_seasonality']}
            - **Weekly Seasonality:** {prophet_results['params']['weekly_seasonality']}
            - **Data:** Daily (harian)
            """)

        st.markdown("### 💡 Kesimpulan")
        st.markdown(f"""
        Berdasarkan hasil perbandingan, model **{winner}** menunjukkan performa yang lebih baik
        untuk memprediksi konsumsi energi listrik rumah tangga. Hal ini ditunjukkan oleh nilai
        error yang lebih rendah (MAE, RMSE, MAPE) dan koefisien determinasi (R²) yang lebih tinggi.

        **Catatan penting:**
        - LSTM menggunakan data **per jam** dengan lookback window, cocok untuk prediksi jangka pendek
        - Prophet menggunakan data **harian** dengan dekomposisi seasonal, cocok untuk prediksi tren jangka panjang
        - Kedua model memiliki kelebihan masing-masing tergantung skenario penggunaan
        """)

    # ========================================================================
    # PAGE: TENTANG
    # ========================================================================
    elif page == "📖 Tentang Penelitian":
        st.subheader("📖 Tentang Penelitian", divider="red")

        st.markdown("""
        ### 📖 Judul Penelitian
        **Perbandingan Algoritma LSTM dan Prophet untuk Prediksi Konsumsi Energi Listrik Rumah Tangga**

        ### 🎓 Informasi Akademik
        - **Mata Kuliah:** Big Data and Visualization
        - **Program Studi:** Magister Teknik Informatika
        - **Universitas:** Universitas Pamulang
        - **Dosen:** Dr. Tukiyat, M.Si

        ### 📊 Dataset
        - **Nama:** Individual Household Electric Power Consumption
        - **Sumber:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
        - **Deskripsi:** Pengukuran konsumsi daya listrik rumah tangga dengan resolusi per menit,
          dikumpulkan selama hampir 4 tahun (Desember 2006 - November 2010)
        - **Lokasi:** Sceaux, Prancis (7 km selatan Paris)

        ### 🤖 Algoritma
        | Model | Tipe | Kelebihan |
        |-------|------|-----------|
        | **LSTM** | Deep Learning (RNN) | Menangkap dependensi temporal jangka panjang |
        | **Prophet** | Statistical (Additive/Multiplicative) | Dekomposisi trend + seasonality + holiday |

        ### 📏 Metrik Evaluasi
        - **MAE** (Mean Absolute Error)
        - **RMSE** (Root Mean Squared Error)
        - **MAPE** (Mean Absolute Percentage Error)
        - **R²** (Coefficient of Determination)

        ### 🛠️ Teknologi
        - Python 3.x
        - TensorFlow/Keras (LSTM)
        - Facebook Prophet
        - Streamlit (Dashboard)
        - Plotly (Visualisasi)
        - Pandas, NumPy, Scikit-learn
        """)


if __name__ == "__main__":
    main()
