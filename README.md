# 🔍 TruthLens — Fake News Detector
**College Capstone Project | Python · Flask · scikit-learn · TF-IDF · NLP**

## 📁 Project Structure
```
fakenews/
├── backend/
│   └── app.py              ← Flask REST API
├── frontend/
│   └── index.html          ← Web UI
├── model/
│   ├── train_model.py      ← ML training script
│   └── fake_news_model.joblib  ← Trained model
├── requirements.txt
├── run.sh
└── README.md
```

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python3 model/train_model.py   # train once
cd backend && python3 app.py   # start server
# Open http://localhost:5000
```

## 📡 API Endpoints
- POST /api/analyze  → verdict + signals + flags
- GET  /api/health   → server status
- GET  /api/samples  → sample articles

## 🧠 ML Pipeline
TF-IDF (10k features, bigrams) → Logistic Regression → 93.8% accuracy

## 📊 Real LIAR Dataset
Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
Uncomment load_liar_dataset() in train_model.py and retrain.
