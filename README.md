# 🤖 MuradAI v4.1

Your Personal Self-Learning AI Assistant powered by Ollama.

## Features

- 🧠 **Local LLM Integration** - Connects to Ollama for private AI
- 💬 **Streaming Chat** - Real-time response streaming
- 📝 **Conversation Memory** - Persistent chat history
- 🎨 **Modern UI** - Dark theme React interface
- 🔄 **Multiple Models** - Support for Llama, DeepSeek, Qwen
- 📱 **Responsive Design** - Works on desktop and mobile

## Quick Start

### Option 1: Local Development

```bash
# 1. Install Ollama
# Visit: https://ollama.com

# 2. Pull a model
ollama pull llama3.2

# 3. Start MuradAI
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Open browser
# http://localhost:8000
```

### Option 2: Docker

```bash
# Start everything
docker-compose up -d

# Pull a model
docker exec -it muradai-ollama ollama pull llama3.2
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/api/health` | GET | Health check |
| `/api/models` | GET | List available models |
| `/api/chat` | POST | Send message (non-streaming) |
| `/api/chat/stream` | POST | Send message (streaming) |
| `/api/conversations` | GET | List conversations |
| `/api/conversations/{id}` | GET | Get conversation |
| `/api/conversations/{id}` | DELETE | Delete conversation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `llama3.2` | Default LLM model |
| `MAX_HISTORY` | `50` | Max messages per conversation |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   React UI  │────▶│   FastAPI   │────▶│   Ollama    │
│  (Embedded) │◀────│   Backend   │◀────│    LLM      │
└─────────────┘     └─────────────┘     └─────────────┘
```

## License

MIT License - Created by Murad Ghannam
