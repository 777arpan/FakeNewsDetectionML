#!/bin/bash
# TruthLens - Start Script
# ========================
# Run this once to train the model and start the server.

set -e

echo ""
echo "========================================"
echo "  TruthLens - Fake News Detector"
echo "  Flask + scikit-learn Backend"
echo "========================================"
echo ""

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q

# Step 2: Train the model if not already trained
if [ ! -f "model/fake_news_model.joblib" ]; then
  echo "🧠 Training ML model (first-time setup)..."
  python3 model/train_model.py
else
  echo "✅ Model already trained. Skipping..."
fi

# Step 3: Start the Flask server
echo ""
echo "🚀 Starting Flask server at http://localhost:5000"
echo "   Open your browser → http://localhost:5000"
echo ""
cd backend && python3 app.py
