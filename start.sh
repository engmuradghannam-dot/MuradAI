#!/bin/bash
# MuradAI v4.1 Startup Script

echo "🚀 Starting MuradAI v4.1..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not detected on localhost:11434"
    echo "   Please start Ollama or set OLLAMA_HOST env variable"
fi

# Install dependencies if needed
pip install -q -r requirements.txt 2>/dev/null || true

# Start the server
echo "✅ Starting FastAPI server on http://0.0.0.0:8000"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
