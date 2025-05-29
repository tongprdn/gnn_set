
# 📈 FinMamba Thai Stock Prediction

Build a predictive AI system using the **FinMamba (LSTM + GNN hybrid)** model to forecast 3-day stock price movements in the Thai SET market. The system is designed to simulate backtesting in Phase 1 and explore real-world trading automation via the SETTRADE API in Phase 2.

---

## 📁 Project Structure

```
project_root/
├── data/                        # Raw and processed stock data
│   ├── raw/                    # Pulled from SETTRADE API
│   └── processed/              # Cleaned, engineered features
├── models/                     # Trained models, checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Core source code
│   ├── config.py               # Load .env and global configs
│   ├── data_loader.py          # API data fetching
│   ├── preprocessing.py        # Feature engineering + graph builder
│   ├── train.py                # FinMamba training loop
│   └── evaluate.py             # Backtesting and evaluation
├── utils/                      # Utility functions
│   └── logger.py
├── .env                        # Secrets for SETTRADE API (not committed)
├── requirements.txt            # pip dependencies
└── main.py                     # Entrypoint to train/evaluate pipeline
```

---

## 🚀 Implementation Plan

### ✅ Phase 1: Predictive AI Model Training (Backtest)

- Train **FinMamba** on historical OHLCV data for ~30–50 most liquid stocks from SET.
- Create graph edges based on **price correlation** and **industry relations**.
- Use **3-day prediction window** for supervised learning classification.
- Evaluate with both **classification metrics** and **backtested PnL**.
- Benchmark against rule-based bot (e.g. EMA50/EMA200 strategy).

---

### 🧪 Phase 2: Real-World Trading Integration

- Integrate live trading logic via **SETTRADE API**.
- Add **macroeconomic sentiment nodes** from NLP/news sources.
- Apply **macro filter** to avoid bad market conditions.
- Use **ensemble logic** to blend FinMamba with rule-based bots.
- Monitor **model drift** and support dynamic retraining.
- Evaluate with **realized ROI** and **signal precision**.

---

## 🔐 Environment

- Python 3.10 (conda)
- PyTorch (MPS for Mac M2 Pro)
- PyTorch Geometric
- SETTRADE Open API
