# 📈 AI Stock Price Predictor

> An intelligent stock price prediction dashboard powered by LSTM Neural Networks,
> built with Python and deployed live on Streamlit Cloud.


---

## 🌍 Live Demo

👉 **[Open the Live App](https://stock-market-prediction-using-ai-jb3zv6ho3tsbzggr2p9hsd.streamlit.app/)**

---

## 📌 What This App Does

- Downloads **real-time stock data** from Yahoo Finance
- Calculates **5 technical indicators** used by professional traders
- Uses a trained **LSTM neural network** to predict tomorrow's closing price
- Displays an **interactive dashboard** with charts, signals, and live prices
- Covers **8 major companies** across India and the US

---

## 🏢 Supported Companies

| # | Company | Ticker | Market | Sector |
|---|---------|--------|--------|--------|
| 1 | 🇮🇳 Reliance Industries | RELIANCE.NS | NSE India | Energy & Retail |
| 2 | 🇮🇳 TCS | TCS.NS | NSE India | IT Services |
| 3 | 🇮🇳 HDFC Bank | HDFCBANK.NS | NSE India | Banking |
| 4 | 🇮🇳 Infosys | INFY.NS | NSE India | IT Services |
| 5 | 🇺🇸 Apple | AAPL | NASDAQ | Technology |
| 6 | 🇺🇸 NVIDIA | NVDA | NASDAQ | Semiconductors |
| 7 | 🇺🇸 Microsoft | MSFT | NASDAQ | Technology |
| 8 | 🇺🇸 Tesla | TSLA | NASDAQ | Electric Vehicles |

---

## 🧠 How the AI Works
Yahoo Finance (real stock prices)

↓

Download 5 years of historical data

↓

Calculate technical indicators

(SMA-20, EMA-50, RSI-14, MACD, MACD Signal)

↓

Scale all values to 0–1 (MinMaxScaler)

↓

Create 60-day sliding windows

(each window = 60 days of 6 features)

↓

LSTM Neural Network learns the patterns

↓

Predicts tomorrow's closing price

↓

Display on live Streamlit dashboard

### Model Architecture
Input Layer     → (60 days × 6 features)

LSTM Layer 1    → 64 neurons, return_sequences=True

Dropout         → 20% (prevents overfitting)

LSTM Layer 2    → 32 neurons, return_sequences=False

Dropout         → 20%

Dense Layer     → 1 neuron (tomorrow's price)

### Model Performance

| Metric | Value |
|--------|-------|
| RMSE | ₹26.60 |
| MAE | ₹21.61 |
| MAPE | 1.53% |
| **Accuracy** | **98.47%** |
| Test samples | 226 unseen days |
| Rating | 🟢 Excellent |

---

## 📊 Technical Indicators Explained

| Indicator | What It Measures | Signal |
|-----------|-----------------|--------|
| **SMA-20** | Average price of last 20 days | Trend direction |
| **EMA-50** | Weighted average (recent days count more) | Medium-term trend |
| **RSI-14** | Momentum — 0 to 100 | >70 overbought, <30 oversold |
| **MACD** | Difference between two EMAs | Momentum direction |
| **MACD Signal** | Smoothed MACD | Crossover = buy/sell signal |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **AI/ML** | TensorFlow 2.17, Keras, scikit-learn |
| **Data** | yfinance (Yahoo Finance API) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Dashboard** | Streamlit |
| **Deployment** | Streamlit Cloud (free) |
| **Version Control** | Git + GitHub |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/ayushsankhyan/Stock-Market-Prediction-Using-AI.git
cd Stock-Market-Prediction-Using-AI
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run app.py
```

**5. Open in browser**
---

## 📁 Project Structure

Stock-Market-Prediction-Using-AI/

│

├── app.py              ← Main Streamlit dashboard

├── explore.py          ← Step 1: Data exploration

├── indicators.py       ← Step 2: Technical indicators

├── prepare_data.py     ← Step 3: Data preparation for LSTM

├── train_model.py      ← Step 4: Model training

├── evaluate.py         ← Step 5: Model evaluation

│

├── stock_lstm.keras    ← Trained LSTM model

├── scaler.pkl          ← MinMaxScaler (for inverse transform)

│

├── requirements.txt    ← Python dependencies

├── runtime.txt         ← Python version for Streamlit Cloud

└── README.md           ← This file

---

## 📈 Dashboard Features

- **5 live metrics** — current price, change %, 52W high/low, volume
- **Price chart** — closing price + SMA-20 + EMA-50
- **RSI chart** — with overbought/oversold zones highlighted
- **MACD chart** — with bullish/bearish zones highlighted
- **AI prediction** — tomorrow's price with direction signal
- **Comparison bar chart** — today vs predicted
- **8-company snapshot table** — live prices for all companies
- **Raw data table** — last 30 days (toggle in sidebar)

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**.

- The model predicts based on historical price patterns
- It cannot account for news, earnings, or market events
- **Never make investment decisions based solely on AI predictions**
- Past performance does not guarantee future results

---

## 👨‍💻 Built By

**Ayush Sankhyan**

- 💼 LinkedIn: www.linkedin.com/in/ayush-sankhyan-81b9b8250
- 🐙 GitHub: [github.com/ayushsankhyan]

---
