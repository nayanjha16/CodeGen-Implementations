#!/bin/bash
# Local deployment with ngrok (for testing on your Mac)

echo "🚀 Starting local deployment..."

# 1. Install ngrok if not already installed
if ! command -v ngrok &> /dev/null; then
    echo "📥 Installing ngrok..."
    brew install ngrok
fi

# 2. Set the model (change if needed)
export OLLAMA_MODEL="phi3-finetuned"
echo "✅ Using model: $OLLAMA_MODEL"

# 3. Start Streamlit in background
echo "🌐 Starting Streamlit..."
streamlit run ui.py --server.port 8501 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 5

# 4. Create ngrok tunnel
echo "🔗 Creating public URL..."
ngrok http 8501 &
NGROK_PID=$!

echo ""
echo "="
echo "✅ Deployment started!"
echo ""
echo "📱 To get your public URL:"
echo "   1. Open: http://localhost:4040"
echo "   2. Look for the 'https://' URL"
echo "   3. Share that URL with your professor"
echo ""
echo "⚠️  Keep this terminal open!"
echo "="

# Keep running until Ctrl+C
wait
