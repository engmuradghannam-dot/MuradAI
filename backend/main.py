#!/usr/bin/env python3
"""
MuradAI v4.1 - Full Stack AI Chat
Backend: FastAPI + Ollama
Frontend: React + Dashboard embedded in one file
"""

import os
import json
import aiohttp
from typing import Optional, List, Dict, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "50"))

AVAILABLE_MODELS = [
    {"id": "llama3.2", "name": "Llama 3.2", "size": "2GB", "description": "Fast & efficient"},
    {"id": "deepseek-r1:7b", "name": "DeepSeek R1", "size": "7GB", "description": "Reasoning focused"},
    {"id": "qwen2.5-coder:7b", "name": "Qwen 2.5 Coder", "size": "7GB", "description": "Code generation"},
]

# ═══════════════════════════════════════════════════════════════
# MEMORY STORE
# ═══════════════════════════════════════════════════════════════

class ConversationMemory:
    def __init__(self, max_messages: int = MAX_HISTORY):
        self.conversations: Dict[str, List[Dict]] = {}
        self.max_messages = max_messages

    def get_messages(self, conversation_id: str) -> List[Dict]:
        return self.conversations.get(conversation_id, [])

    def add_message(self, conversation_id: str, role: str, content: str, model: str = None):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model
        }
        self.conversations[conversation_id].append(msg)

        if len(self.conversations[conversation_id]) > self.max_messages:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_messages:]

    def clear_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def get_conversation_list(self) -> List[Dict]:
        result = []
        for conv_id, messages in self.conversations.items():
            if messages:
                first_msg = messages[0]["content"][:50] + "..." if len(messages[0]["content"]) > 50 else messages[0]["content"]
                result.append({
                    "id": conv_id,
                    "title": first_msg,
                    "message_count": len(messages),
                    "last_updated": messages[-1]["timestamp"] if messages else None
                })
        return sorted(result, key=lambda x: x["last_updated"] or "", reverse=True)

memory = ConversationMemory()

# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(default="default")
    model: Optional[str] = Field(default=None)
    stream: bool = Field(default=False)
    system_prompt: Optional[str] = Field(default=None)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    model: str
    timestamp: str

# ═══════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════════════

async def check_ollama_health() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_HOST}/api/tags", timeout=aiohttp.ClientTimeout(total=5)):
                return True
    except:
        return False

async def list_ollama_models() -> List[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_HOST}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                return [m["name"] for m in data.get("models", [])]
    except:
        return []

async def generate_response(message: str, conversation_id: str, model: str = None, system_prompt: str = None) -> str:
    model = model or DEFAULT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history = memory.get_messages(conversation_id)
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9}
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            return data.get("message", {}).get("content", "No response")

async def generate_streaming_response(
    message: str, conversation_id: str, model: str = None, system_prompt: str = None
) -> AsyncGenerator[str, None]:
    model = model or DEFAULT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history = memory.get_messages(conversation_id)
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "top_p": 0.9}
    }

    full_response = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            async for line in resp.content:
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8").strip())
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"

                    if data.get("done", False):
                        memory.add_message(conversation_id, "user", message, model)
                        memory.add_message(conversation_id, "assistant", full_response, model)
                        yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': full_response})}\n\n"
                        break
                except json.JSONDecodeError:
                    continue

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="MuradAI",
    description="Advanced AI Chat with Streaming & Memory",
    version="4.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    ollama_ok = await check_ollama_health()
    models = await list_ollama_models()
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "version": "4.1",
        "ollama": "connected" if ollama_ok else "disconnected",
        "models_available": models,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/models")
async def get_models():
    ollama_models = await list_ollama_models()
    result = []
    for model in AVAILABLE_MODELS:
        model_copy = model.copy()
        model_copy["available"] = model["id"] in ollama_models
        result.append(model_copy)
    return result

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response_text = await generate_response(
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            system_prompt=request.system_prompt
        )

        memory.add_message(request.conversation_id, "user", request.message, request.model or DEFAULT_MODEL)
        memory.add_message(request.conversation_id, "assistant", response_text, request.model or DEFAULT_MODEL)

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            model=request.model or DEFAULT_MODEL,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not request.stream:
        raise HTTPException(status_code=400, detail="Set stream=true for streaming")

    async def event_generator():
        async for chunk in generate_streaming_response(
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            system_prompt=request.system_prompt
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )

@app.get("/api/conversations")
async def list_conversations():
    return {"conversations": memory.get_conversation_list()}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    messages = memory.get_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": messages}

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    memory.clear_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}

@app.post("/api/conversations/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    memory.clear_conversation(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}

# ═══════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

# ═══════════════════════════════════════════════════════════════
# MAIN CHAT PAGE
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=CHAT_HTML)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("/dev/null")

# ═══════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MuradAI v4.1 - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #222;
        }
        .logo {
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #2ecc71;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #111118;
            border: 1px solid #222;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .section {
            background: #111118;
            border: 1px solid #222;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #fff;
        }
        .model-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            background: rgba(46, 204, 113, 0.08);
            border: 1px solid rgba(46, 204, 113, 0.2);
        }
        .model-item.inactive {
            background: rgba(255, 255, 255, 0.02);
            border-color: #222;
        }
        .model-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .model-icon { font-size: 16px; }
        .model-name { font-size: 13px; font-weight: 600; }
        .model-desc { font-size: 11px; color: #666; }
        .badge {
            font-size: 11px;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 600;
        }
        .badge-active { background: #2ecc71; color: white; }
        .badge-download { background: #222; color: #666; }
        .actions-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .action-btn {
            padding: 12px;
            border: 1px solid #222;
            border-radius: 8px;
            background: transparent;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            text-decoration: none;
        }
        .action-btn:hover {
            background: rgba(255,255,255,0.05);
            border-color: #667eea;
        }
        .footer {
            text-align: center;
            font-size: 11px;
            color: #444;
            margin-top: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">MuradAI v4.1</div>
            <div class="status">
                <div class="status-dot"></div>
                <span style="font-size:13px;color:#888;">Connected</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Models Available</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="conv-count">0</div>
                <div class="stat-label">Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">50</div>
                <div class="stat-label">Max History</div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Available Models</div>
            <div class="model-item">
                <div class="model-info">
                    <span class="model-icon">Llama</span>
                    <div>
                        <div class="model-name">Llama 3.2</div>
                        <div class="model-desc">Fast & efficient - 2GB</div>
                    </div>
                </div>
                <span class="badge badge-active">Active</span>
            </div>
            <div class="model-item inactive">
                <div class="model-info">
                    <span class="model-icon">DeepSeek</span>
                    <div>
                        <div class="model-name">DeepSeek R1</div>
                        <div class="model-desc">Reasoning focused - 7GB</div>
                    </div>
                </div>
                <span class="badge badge-download">Download</span>
            </div>
            <div class="model-item inactive">
                <div class="model-info">
                    <span class="model-icon">Qwen</span>
                    <div>
                        <div class="model-name">Qwen 2.5 Coder</div>
                        <div class="model-desc">Code generation - 7GB</div>
                    </div>
                </div>
                <span class="badge badge-download">Download</span>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Quick Actions</div>
            <div class="actions-grid">
                <a href="/" class="action-btn">Chat Interface</a>
                <a href="/api/docs" class="action-btn">API Docs</a>
                <a href="/api/health" class="action-btn">Health Check</a>
                <a href="/api/models" class="action-btn">Models List</a>
            </div>
        </div>

        <div class="footer">
            MuradAI v4.1 - FastAPI + Ollama - GitHub Codespace Ready
        </div>
    </div>

    <script>
        fetch('/api/conversations')
            .then(r => r.json())
            .then(data => {
                document.getElementById('conv-count').textContent = data.conversations?.length || 0;
            })
            .catch(() => {});
    </script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
# CHAT HTML (Complete React UI)
# ═══════════════════════════════════════════════════════════════

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MuradAI v4.1 - Chat</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        #root { height: 100vh; }
        .app-container { display: flex; height: 100vh; }
        .sidebar {
            width: 260px;
            background: #111118;
            border-right: 1px solid #222;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header { padding: 20px; border-bottom: 1px solid #222; }
        .logo {
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .new-chat-btn {
            margin: 15px;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .new-chat-btn:hover { transform: translateY(-2px); }
        .conversations-list { flex: 1; overflow-y: auto; padding: 10px; }
        .conversation-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .conversation-item:hover { background: #1a1a2e; }
        .conversation-item.active { background: #1a1a2e; border-left: 3px solid #667eea; }
        .sidebar-footer {
            padding: 15px;
            border-top: 1px solid #222;
            font-size: 12px;
            color: #555;
            text-align: center;
        }
        .main-area { flex: 1; display: flex; flex-direction: column; background: #0a0a0f; }
        .chat-header {
            padding: 15px 25px;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: #111118;
            border: 1px solid #333;
            border-radius: 8px;
            font-size: 13px;
            color: #aaa;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2ecc71;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            gap: 20px;
        }
        .welcome-logo {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .message {
            display: flex;
            gap: 15px;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }
        .message.user { flex-direction: row-reverse; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        .avatar.ai { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .avatar.user { background: #2c3e50; }
        .message-content {
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 80%;
            line-height: 1.6;
            font-size: 15px;
        }
        .message.ai .message-content {
            background: #111118;
            border: 1px solid #222;
            border-top-left-radius: 4px;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-top-right-radius: 4px;
        }
        .typing-indicator { display: flex; gap: 4px; padding: 20px; }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .input-area { padding: 20px; border-top: 1px solid #222; }
        .input-container {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 10px;
            align-items: flex-end;
            background: #111118;
            border: 1px solid #333;
            border-radius: 16px;
            padding: 12px 16px;
        }
        .message-input {
            flex: 1;
            background: none;
            border: none;
            color: #e0e0e0;
            font-size: 15px;
            resize: none;
            outline: none;
            max-height: 150px;
            min-height: 24px;
            font-family: inherit;
        }
        .send-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect, useRef } = React;
        const API_BASE = window.location.origin;

        function App() {
            const [conversations, setConversations] = useState([]);
            const [currentConv, setCurrentConv] = useState('default');
            const [messages, setMessages] = useState([]);
            const [input, setInput] = useState('');
            const [isLoading, setIsLoading] = useState(false);
            const [selectedModel, setSelectedModel] = useState('llama3.2');
            const [health, setHealth] = useState(null);
            const messagesEndRef = useRef(null);

            useEffect(() => {
                fetchConversations();
                fetchHealth();
                const interval = setInterval(fetchHealth, 30000);
                return () => clearInterval(interval);
            }, []);

            useEffect(() => {
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }, [messages]);

            const fetchHealth = async () => {
                try {
                    const res = await fetch(`${API_BASE}/api/health`);
                    const data = await res.json();
                    setHealth(data);
                } catch (e) {
                    setHealth({ status: 'error', ollama: 'disconnected' });
                }
            };

            const fetchConversations = async () => {
                try {
                    const res = await fetch(`${API_BASE}/api/conversations`);
                    const data = await res.json();
                    setConversations(data.conversations || []);
                } catch (e) {}
            };

            const newConversation = () => {
                const id = 'conv_' + Date.now();
                setCurrentConv(id);
                setMessages([]);
            };

            const sendMessage = async () => {
                if (!input.trim() || isLoading) return;
                const userMessage = input.trim();
                setInput('');
                setIsLoading(true);

                const newMessages = [...messages, {
                    role: 'user', content: userMessage, timestamp: new Date().toISOString()
                }];
                setMessages(newMessages);

                try {
                    const response = await fetch(`${API_BASE}/api/chat/stream`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: userMessage,
                            conversation_id: currentConv,
                            model: selectedModel,
                            stream: true
                        })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullResponse = '';

                    setMessages([...newMessages, {
                        role: 'assistant', content: '', timestamp: new Date().toISOString(), streaming: true
                    }]);

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n\\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.chunk) {
                                        fullResponse += data.chunk;
                                        setMessages(prev => {
                                            const updated = [...prev];
                                            const lastMsg = updated[updated.length - 1];
                                            if (lastMsg && lastMsg.role === 'assistant') {
                                                lastMsg.content = fullResponse;
                                            }
                                            return updated;
                                        });
                                    }
                                    if (data.done) {
                                        setMessages(prev => {
                                            const updated = [...prev];
                                            const lastMsg = updated[updated.length - 1];
                                            if (lastMsg) {
                                                lastMsg.streaming = false;
                                                lastMsg.content = data.full_response || fullResponse;
                                            }
                                            return updated;
                                        });
                                    }
                                } catch (e) {}
                            }
                        }
                    }
                    fetchConversations();
                } catch (e) {
                    setMessages([...newMessages, {
                        role: 'assistant',
                        content: 'Error: Could not connect to AI. Please make sure Ollama is running.',
                        timestamp: new Date().toISOString(),
                        error: true
                    }]);
                } finally {
                    setIsLoading(false);
                }
            };

            const handleKeyDown = (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            };

            const suggestions = [
                "Explain quantum computing",
                "Write a Python function",
                "What are benefits of meditation?",
                "Help me debug this code"
            ];

            return (
                <div className="app-container">
                    <div className="sidebar">
                        <div className="sidebar-header">
                            <div className="logo">MuradAI v4.1</div>
                        </div>
                        <button className="new-chat-btn" onClick={newConversation}>+ New Chat</button>
                        <div className="conversations-list">
                            {conversations.map(conv => (
                                <div key={conv.id}
                                    className={`conversation-item ${conv.id === currentConv ? 'active' : ''}`}
                                    onClick={() => { setCurrentConv(conv.id); setMessages([]); }}>
                                    <div style={{fontSize:'13px'}}>{conv.title || 'New Chat'}</div>
                                </div>
                            ))}
                        </div>
                        <div className="sidebar-footer">
                            <a href="/dashboard" style={{color:'#667eea',textDecoration:'none'}}>Dashboard</a>
                        </div>
                    </div>
                    <div className="main-area">
                        <div className="chat-header">
                            <span style={{fontSize:'16px',fontWeight:'600'}}>MuradAI Chat</span>
                            <div className="model-selector">
                                <span className="status-dot" style={{
                                    background: health?.ollama === 'connected' ? '#2ecc71' : '#e74c3c'
                                }}></span>
                                <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}
                                    style={{background:'none',border:'none',color:'#aaa',outline:'none',cursor:'pointer',fontSize:'13px'}}>
                                    <option value="llama3.2">Llama 3.2</option>
                                    <option value="deepseek-r1:7b">DeepSeek R1</option>
                                    <option value="qwen2.5-coder:7b">Qwen 2.5 Coder</option>
                                </select>
                            </div>
                        </div>
                        <div className="messages-container">
                            {messages.length === 0 ? (
                                <div className="welcome-screen">
                                    <div className="welcome-logo">MuradAI</div>
                                    <div style={{fontSize:'18px',color:'#666',maxWidth:'500px'}}>
                                        Your personal AI assistant powered by Ollama
                                    </div>
                                    <div style={{display:'flex',flexWrap:'wrap',gap:'10px',justifyContent:'center',maxWidth:'600px'}}>
                                        {suggestions.map((s, i) => (
                                            <div key={i} onClick={() => setInput(s)}
                                                style={{padding:'10px 20px',background:'#111118',border:'1px solid #333',borderRadius:'20px',cursor:'pointer',fontSize:'13px'}}>
                                                {s}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                messages.map((msg, i) => (
                                    <div key={i} className={`message ${msg.role}`}>
                                        <div className={`avatar ${msg.role}`}>{msg.role === 'user' ? 'U' : 'AI'}</div>
                                        <div className="message-content">{msg.content}</div>
                                    </div>
                                ))
                            )}
                            {isLoading && (
                                <div className="message ai">
                                    <div className="avatar ai">AI</div>
                                    <div className="typing-indicator">
                                        <span></span><span></span><span></span>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                        <div className="input-area">
                            <div className="input-container">
                                <textarea className="message-input" placeholder="Message MuradAI..."
                                    value={input} onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown} rows={1} />
                                <button className="send-btn" onClick={sendMessage}
                                    disabled={!input.trim() || isLoading}>Send</button>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }
        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
