#!/usr/bin/env python3
"""
MuradAI v4.1 - Interactive Dashboard Widget
FastAPI endpoint that serves the Kimi widget dashboard
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MuradAI v4.1 Dashboard</title>
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
            <div class="logo">🤖 MuradAI <span style="font-weight:400;font-size:14px;opacity:0.7;">v4.1</span></div>
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
                    <span class="model-icon">🦙</span>
                    <div>
                        <div class="model-name">Llama 3.2</div>
                        <div class="model-desc">Fast & efficient • 2GB</div>
                    </div>
                </div>
                <span class="badge badge-active">Active</span>
            </div>
            <div class="model-item inactive">
                <div class="model-info">
                    <span class="model-icon">🧠</span>
                    <div>
                        <div class="model-name">DeepSeek R1</div>
                        <div class="model-desc">Reasoning focused • 7GB</div>
                    </div>
                </div>
                <span class="badge badge-download">Download</span>
            </div>
            <div class="model-item inactive">
                <div class="model-info">
                    <span class="model-icon">💻</span>
                    <div>
                        <div class="model-name">Qwen 2.5 Coder</div>
                        <div class="model-desc">Code generation • 7GB</div>
                    </div>
                </div>
                <span class="badge badge-download">Download</span>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Quick Actions</div>
            <div class="actions-grid">
                <a href="/" class="action-btn"><span>💬</span> Chat Interface</a>
                <a href="/api/docs" class="action-btn"><span>📚</span> API Docs</a>
                <a href="/api/health" class="action-btn"><span>🏥</span> Health Check</a>
                <a href="/api/models" class="action-btn"><span>🤖</span> Models List</a>
            </div>
        </div>

        <div class="footer">
            MuradAI v4.1 • FastAPI + Ollama • GitHub Codespace Ready
        </div>
    </div>

    <script>
        // Fetch real conversation count
        fetch('/api/conversations')
            .then(r => r.json())
            .then(data => {
                document.getElementById('conv-count').textContent = data.conversations?.length || 0;
            })
            .catch(() => {});
    </script>
</body>
</html>"""

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)
