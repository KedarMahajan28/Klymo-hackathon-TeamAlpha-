#!/bin/bash

# Setup script for Satellite Image Super-Resolution App
# Team Alpha - Klymo Hackathon

echo "🛰️ Setting up Satellite Image Super-Resolution App..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Clone SwinIR if not exists
if [ ! -d "SwinIR" ]; then
    echo "📦 Cloning SwinIR repository..."
    git clone https://github.com/JingyunLiang/SwinIR.git
    echo "   ✅ SwinIR cloned successfully"
else
    echo "✅ SwinIR repository already exists"
fi
echo ""

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt
echo "   ✅ Dependencies installed successfully"
echo ""

# Check for model file
if [ -f "best_model.pth" ]; then
    echo "✅ Model file found: best_model.pth"
else
    echo "⚠️  Model file not found. Please place your .pth file in this directory"
    echo "   You can specify a different path when running the app"
fi
echo ""

echo "🎉 Setup complete!"
echo ""
echo "To run the app:"
echo "   streamlit run app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
