#!/usr/bin/env python3
"""
MuradAI v4.1 - Full Stack AI Chat
Backend: FastAPI + Ollama
Frontend: React static files served from /static
"""

import os
import json
import aiohttp
from typing import Optional, List, Dict, AsyncGenerator
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import dashboard
from dashboard import router as dashboard_router
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

class HealthResponse(BaseModel):
    status: str
    version: str
    llm: str
    models_available: List[str]
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

# CORS
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
# DASHBOARD ROUTER
# ═══════════════════════════════════════════════════════════════

app.include_router(dashboard_router)

# ═══════════════════════════════════════════════════════════════
# STATIC FRONTEND (Single HTML with embedded React)
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React chat UI"""
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("/dev/null")

# ═══════════════════════════════════════════════════════════════
# EMBEDDED REACT FRONTEND HTML
# ═══════════════════════════════════════════════════════════════

HTML_CONTENT = """<!DOCTYPE html>
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

        .app-container {
            display: flex;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background: #111118;
            border-right: 1px solid #222;
            display: flex;
            flex-direction: column;
            transition: width 0.3s;
        }
        .sidebar.collapsed { width: 0; overflow: hidden; }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #222;
        }
        .logo {
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .logo span { font-weight: 300; font-size: 14px; opacity: 0.7; }

        .new-chat-btn {
            margin: 15px 15px 0;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .new-chat-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(102,126,234,0.4); }

        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
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
        .conv-icon { font-size: 16px; }
        .conv-info { flex: 1; overflow: hidden; }
        .conv-title {
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .conv-meta {
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        .delete-btn {
            opacity: 0;
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s;
        }
        .conversation-item:hover .delete-btn { opacity: 1; }
        .delete-btn:hover { color: #ff4757; background: rgba(255,71,87,0.1); }

        .sidebar-footer {
            padding: 15px;
            border-top: 1px solid #222;
            font-size: 12px;
            color: #555;
            text-align: center;
        }

        /* Main Chat Area */
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0a0a0f;
        }

        .chat-header {
            padding: 15px 25px;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .menu-btn {
            background: none;
            border: none;
            color: #888;
            font-size: 20px;
            cursor: pointer;
            padding: 5px;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .menu-btn:hover { background: #1a1a2e; color: #fff; }
        .header-title {
            font-size: 16px;
            font-weight: 600;
        }
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: #111118;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            color: #aaa;
            transition: all 0.2s;
        }
        .model-selector:hover { border-color: #667eea; }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2ecc71;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

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
            background-clip: text;
        }
        .welcome-subtitle {
            font-size: 18px;
            color: #666;
            max-width: 500px;
        }
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            max-width: 600px;
            margin-top: 10px;
        }
        .chip {
            padding: 10px 20px;
            background: #111118;
            border: 1px solid #333;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .chip:hover {
            border-color: #667eea;
            background: #1a1a2e;
            transform: translateY(-2px);
        }

        .message {
            display: flex;
            gap: 15px;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
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
        .avatar.ai {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .avatar.user {
            background: #2c3e50;
        }
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
        .message-meta {
            font-size: 11px;
            color: #555;
            margin-top: 5px;
            padding: 0 5px;
        }
        .message.user .message-meta { text-align: right; }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 20px;
        }
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
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .input-area {
            padding: 20px;
            border-top: 1px solid #222;
        }
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
            transition: all 0.2s;
        }
        .input-container:focus-within {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
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
        .message-input::placeholder { color: #555; }
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
            transition: all 0.2s;
            flex-shrink: 0;
        }
        .send-btn:hover { transform: scale(1.1); box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .input-footer {
            text-align: center;
            font-size: 12px;
            color: #444;
            margin-top: 10px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #444; }

        /* Code blocks */
        pre {
            background: #0d0d15;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
            margin: 10px 0;
        }
        code {
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
        }
        p { margin: 8px 0; }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar { position: fixed; z-index: 100; height: 100vh; }
            .message-content { max-width: 90%; }
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect, useRef, useCallback } = React;

        // API base URL
        const API_BASE = window.location.origin;

        // Icons
        const Icons = {
            Menu: () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>,
            Send: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>,
            Plus: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>,
            Chat: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
            Trash: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>,
            Bot: () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M12 11V7a4 4 0 0 1 4-4h0"/><path d="M8 7a4 4 0 0 1 4-4h0"/><line x1="9" y1="15" x2="9.01" y2="15"/><line x1="15" y1="15" x2="15.01" y2="15"/></svg>,
            User: () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
        };

        // Markdown-like parser
        function formatMessage(text) {
            if (!text) return '';
            let html = text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Code blocks
            html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
            html = html.replace(/`([^`]+)`/g, '<code style="background:#1a1a2e;padding:2px 6px;border-radius:4px;">$1</code>');

            // Bold
            html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

            // Italic
            html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

            // Links
            html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:#667eea;">$1</a>');

            // Line breaks
            html = html.replace(/\n/g, '<br/>');

            return html;
        }

        function App() {
            const [sidebarOpen, setSidebarOpen] = useState(true);
            const [conversations, setConversations] = useState([]);
            const [currentConv, setCurrentConv] = useState('default');
            const [messages, setMessages] = useState([]);
            const [input, setInput] = useState('');
            const [isLoading, setIsLoading] = useState(false);
            const [models, setModels] = useState([]);
            const [selectedModel, setSelectedModel] = useState('llama3.2');
            const [health, setHealth] = useState(null);
            const messagesEndRef = useRef(null);
            const inputRef = useRef(null);

            // Load conversations on mount
            useEffect(() => {
                fetchConversations();
                fetchModels();
                fetchHealth();
                const interval = setInterval(fetchHealth, 30000);
                return () => clearInterval(interval);
            }, []);

            // Scroll to bottom
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

            const fetchModels = async () => {
                try {
                    const res = await fetch(`${API_BASE}/api/models`);
                    const data = await res.json();
                    setModels(data);
                } catch (e) {
                    console.error('Failed to fetch models');
                }
            };

            const fetchConversations = async () => {
                try {
                    const res = await fetch(`${API_BASE}/api/conversations`);
                    const data = await res.json();
                    setConversations(data.conversations || []);
                } catch (e) {
                    console.error('Failed to fetch conversations');
                }
            };

            const loadConversation = async (convId) => {
                setCurrentConv(convId);
                try {
                    const res = await fetch(`${API_BASE}/api/conversations/${convId}`);
                    if (res.ok) {
                        const data = await res.json();
                        setMessages(data.messages || []);
                    } else {
                        setMessages([]);
                    }
                } catch (e) {
                    setMessages([]);
                }
                if (window.innerWidth < 768) setSidebarOpen(false);
            };

            const newConversation = () => {
                const id = 'conv_' + Date.now();
                setCurrentConv(id);
                setMessages([]);
                fetchConversations();
            };

            const deleteConversation = async (convId, e) => {
                e.stopPropagation();
                try {
                    await fetch(`${API_BASE}/api/conversations/${convId}`, { method: 'DELETE' });
                    if (currentConv === convId) {
                        setCurrentConv('default');
                        setMessages([]);
                    }
                    fetchConversations();
                } catch (e) {
                    console.error('Failed to delete conversation');
                }
            };

            const sendMessage = async () => {
                if (!input.trim() || isLoading) return;

                const userMessage = input.trim();
                setInput('');
                setIsLoading(true);

                // Add user message immediately
                const newMessages = [...messages, {
                    role: 'user',
                    content: userMessage,
                    timestamp: new Date().toISOString()
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

                    // Add placeholder for AI response
                    setMessages([...newMessages, {
                        role: 'assistant',
                        content: '',
                        timestamp: new Date().toISOString(),
                        streaming: true
                    }]);

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\n\n');

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
                        content: '❌ Error: Could not connect to AI. Please make sure Ollama is running.',
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
                "Explain quantum computing in simple terms",
                "Write a Python function to sort a list",
                "What are the benefits of meditation?",
                "Help me debug this code..."
            ];

            return (
                <div className="app-container">
                    {/* Sidebar */}
                    <div className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
                        <div className="sidebar-header">
                            <div className="logo">MuradAI <span>v4.1</span></div>
                        </div>
                        <button className="new-chat-btn" onClick={newConversation}>
                            <Icons.Plus /> New Chat
                        </button>
                        <div className="conversations-list">
                            {conversations.map(conv => (
                                <div
                                    key={conv.id}
                                    className={`conversation-item ${conv.id === currentConv ? 'active' : ''}`}
                                    onClick={() => loadConversation(conv.id)}
                                >
                                    <span className="conv-icon"><Icons.Chat /></span>
                                    <div className="conv-info">
                                        <div className="conv-title">{conv.title || 'New Chat'}</div>
                                        <div className="conv-meta">{conv.message_count} messages</div>
                                    </div>
                                    <button className="delete-btn" onClick={(e) => deleteConversation(conv.id, e)}>
                                        <Icons.Trash />
                                    </button>
                                </div>
                            ))}
                        </div>
                        <div className="sidebar-footer">
                            MuradAI v4.1 © 2025
                        </div>
                    </div>

                    {/* Main Chat Area */}
                    <div className="main-area">
                        <div className="chat-header">
                            <div className="header-left">
                                <button className="menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
                                    <Icons.Menu />
                                </button>
                                <span className="header-title">
                                    {conversations.find(c => c.id === currentConv)?.title || 'New Chat'}
                                </span>
                            </div>
                            <div className="model-selector">
                                <span className="status-dot" style={{
                                    background: health?.ollama === 'connected' ? '#2ecc71' : '#e74c3c'
                                }}></span>
                                <select 
                                    value={selectedModel}
                                    onChange={(e) => setSelectedModel(e.target.value)}
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        color: '#aaa',
                                        outline: 'none',
                                        cursor: 'pointer',
                                        fontSize: '13px'
                                    }}
                                >
                                    {models.map(m => (
                                        <option key={m.id} value={m.id}>
                                            {m.name} {m.available ? '✓' : '(download)'}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="messages-container">
                            {messages.length === 0 ? (
                                <div className="welcome-screen">
                                    <div className="welcome-logo">MuradAI</div>
                                    <div className="welcome-subtitle">
                                        Your personal AI assistant powered by Ollama.
                                        Ask anything, get intelligent responses.
                                    </div>
                                    <div className="suggestion-chips">
                                        {suggestions.map((s, i) => (
                                            <div key={i} className="chip" onClick={() => {
                                                setInput(s);
                                                inputRef.current?.focus();
                                            }}>
                                                {s}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                messages.map((msg, i) => (
                                    <div key={i} className={`message ${msg.role}`}>
                                        <div className={`avatar ${msg.role}`}>
                                            {msg.role === 'user' ? <Icons.User /> : <Icons.Bot />}
                                        </div>
                                        <div>
                                            <div 
                                                className="message-content"
                                                dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                                            />
                                            <div className="message-meta">
                                                {msg.role === 'assistant' && msg.model && `${msg.model} • `}
                                                {new Date(msg.timestamp).toLocaleTimeString()}
                                                {msg.streaming && ' • typing...'}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                            {isLoading && messages[messages.length - 1]?.role !== 'assistant' && (
                                <div className="message ai">
                                    <div className="avatar ai"><Icons.Bot /></div>
                                    <div className="typing-indicator">
                                        <span></span><span></span><span></span>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        <div className="input-area">
                            <div className="input-container">
                                <textarea
                                    ref={inputRef}
                                    className="message-input"
                                    placeholder="Message MuradAI..."
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    rows={1}
                                />
                                <button 
                                    className="send-btn"
                                    onClick={sendMessage}
                                    disabled={!input.trim() || isLoading}
                                >
                                    <Icons.Send />
                                </button>
                            </div>
                            <div className="input-footer">
                                MuradAI can make mistakes. Consider checking important information.
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
