#!/usr/bin/env bash
# ================================================================
#  DeepX Arabic ABSA — Quick Start
#  Run this script from the folder containing your .xlsx files
# ================================================================

echo "🚀 DeepX Arabic ABSA — Setup & Run"
echo "===================================="

# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers==4.40.0 datasets accelerate -q
pip install scikit-learn pandas openpyxl numpy tqdm streamlit plotly -q

echo "✅ Dependencies installed"

# 2. Verify GPU
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU found)')"

# 3. Train
echo ""
echo "🤖 Starting training..."
python3 train.py

# 4. Evaluate on validation set
echo ""
echo "📊 Evaluating on validation set..."
# (After running train.py, also generate val predictions and evaluate)

# 5. Launch demo
echo ""
echo "🌐 Starting Streamlit demo..."
echo "   Open: http://localhost:8501"
streamlit run demo.py
