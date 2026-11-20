"""
Nukumi Mobile PWA Field Support Server (Tailscale Serve / Funnel Ready).

Zero-external-dependency HTTP server (Python stdlib `http.server.ThreadingHTTPServer`)
that wraps the Hybrid Search (ModernBERT + BM25 RRF) and Ollama RAG assistant into a
mobile-first Progressive Web App (PWA) for field technicians.

Security features:
- Binds to 127.0.0.1:8000 by default (accessible only via local Tailscale Funnel/Serve proxy).
- Passcode / PIN gate stored in `.access_pin` (or `ACCESS_PIN` env var) with HMAC-SHA256 cookies.
- Brute-force rate limiter (locks out after 5 failed PIN attempts for 60 seconds).
- Strict path-traversal protection on `/pdf/<filename>`.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ollama
import torch
from sentence_transformers import SentenceTransformer

from assistant import (
    build_contextual_search_query,
    format_rag_prompt,
    get_manual_count,
    load_modelfile_config,
    log_feedback,
    modelfile_path,
    retrieve_context_hits,
)
from generate_queries import load_domain_config
from optiDataset import find_active_pdf_path
from test_search import (
    BM25Index,
    filter_active_corpus,
    load_or_build_corpus,
    load_or_compute_embeddings,
    model_path,
    pdf_dir,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pin_file_path = os.path.join(project_root, ".access_pin")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Per-session state store (keyed by session_id cookie)
SESSION_LOCK = threading.Lock()
SESSIONS: dict[str, dict] = {}

# Brute-force rate limiting store (keyed by client IP / X-Forwarded-For)
AUTH_FAILURES: dict[str, list[float]] = {}
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60.0

SERVER_SECRET = secrets.token_bytes(32)


def load_or_create_access_pin() -> str:
    """Load the access PIN from ACCESS_PIN env var or .access_pin file (generating a 6-digit PIN if missing)."""
    env_pin = os.environ.get("ACCESS_PIN", "").strip()
    if env_pin:
        return env_pin

    if os.path.exists(pin_file_path):
        with open(pin_file_path, "r", encoding="utf-8") as f:
            pin = f.read().strip()
            if pin:
                return pin

    new_pin = f"{secrets.randbelow(900000) + 100000}"
    with open(pin_file_path, "w", encoding="utf-8") as f:
        f.write(new_pin + "\n")
    os.chmod(pin_file_path, 0o600)
    return new_pin


def make_auth_token(pin: str) -> str:
    return hmac.new(SERVER_SECRET, pin.encode("utf-8"), hashlib.sha256).hexdigest()


def get_session(session_id: str) -> dict:
    with SESSION_LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                "chat_history": [],
                "last_user_queries": [],
                "last_search_query": "",
                "last_hits": [],
                "active_device": None,
                "updated_at": time.time(),
            }
        SESSIONS[session_id]["updated_at"] = time.time()
        return SESSIONS[session_id]


ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <rect width="192" height="192" rx="38" fill="#0f172a"/>
  <rect x="12" y="12" width="168" height="168" rx="30" fill="none" stroke="#38bdf8" stroke-width="6"/>
  <path d="M58 134V58h20l36 48V58h20v76h-20L78 86v48H58z" fill="#f8fafc"/>
  <circle cx="144" cy="48" r="12" fill="#f59e0b"/>
</svg>"""

MANIFEST_JSON = json.dumps({
    "name": "Nukumi Field Support",
    "short_name": "Nukumi",
    "description": "HVAC & BMS Field Technical Support Assistant",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0f172a",
    "theme_color": "#0f172a",
    "icons": [
        {
            "src": "/icon.svg",
            "sizes": "any",
            "type": "image/svg+xml",
            "purpose": "any maskable",
        }
    ],
})

MOBILE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#0f172a" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <link rel="manifest" href="/manifest.json" />
  <link rel="icon" href="/icon.svg" type="image/svg+xml" />
  <link rel="apple-touch-icon" href="/icon.svg" />
  <title>Nukumi Field Support</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-alt: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-dark: #0284c7;
      --warn-bg: #451a03;
      --warn-border: #f59e0b;
      --border: #334155;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100dvh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    /* Header */
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 1rem;
    }
    .brand-sub {
      font-size: 0.72rem;
      color: var(--muted);
      font-weight: 400;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pill-btn {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .pill-btn.active {
      background: rgba(56, 189, 248, 0.18);
      border-color: var(--accent);
      color: var(--accent);
    }
    /* Chat area */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      scroll-behavior: smooth;
    }
    .msg {
      max-width: 92%;
      padding: 11px 14px;
      border-radius: 14px;
      font-size: 0.95rem;
      line-height: 1.48;
      word-wrap: break-word;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--accent-dark);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msg.bot {
      align-self: flex-start;
      background: var(--panel);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
      width: 100%;
    }
    .msg.bot p { margin-bottom: 8px; }
    .msg.bot p:last-child { margin-bottom: 0; }
    .msg.bot ol, .msg.bot ul { margin: 6px 0 8px 20px; }
    .msg.bot li { margin-bottom: 4px; }
    .msg.bot strong { color: #7dd3fc; font-weight: 700; }
    .msg.bot code {
      background: #0f172a;
      padding: 2px 5px;
      border-radius: 4px;
      font-family: ui-monospace, monospace;
      font-size: 0.86em;
      color: #fcd34d;
    }
    .warn-box {
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-border);
      padding: 8px 10px;
      border-radius: 6px;
      margin-bottom: 8px;
      font-weight: 600;
      color: #fde68a;
    }
    .pdf-link {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin-top: 6px;
      padding: 5px 10px;
      background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.4);
      border-radius: 8px;
      color: var(--accent);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 600;
    }
    /* Citations accordion */
    details.citations {
      margin-top: 10px;
      border-top: 1px solid var(--border);
      padding-top: 8px;
      font-size: 0.8rem;
    }
    details.citations summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
      user-select: none;
    }
    .cite-card {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      margin-top: 8px;
    }
    .cite-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 5px;
      font-size: 0.75rem;
      color: var(--muted);
    }
    .cite-text {
      font-size: 0.78rem;
      color: #cbd5e1;
      line-height: 1.4;
    }
    /* Feedback buttons */
    .fb-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      flex-wrap: wrap;
    }
    .fb-btn {
      background: #0f172a;
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.76rem;
      font-weight: 600;
      cursor: pointer;
    }
    .fb-btn.up-done {
      border-color: #22c55e;
      color: #4ade80;
      background: rgba(34, 197, 94, 0.12);
    }
    .fb-btn.down-done {
      border-color: #f59e0b;
      color: #fbbf24;
      background: rgba(245, 158, 11, 0.12);
    }
    /* Footer Input */
    footer {
      background: var(--panel);
      border-top: 1px solid var(--border);
      padding: 10px 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-shrink: 0;
    }
    #q-input {
      flex: 1;
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 11px 15px;
      font-size: 0.98rem;
      outline: none;
    }
    #q-input:focus { border-color: var(--accent); }
    #send-btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 50%;
      width: 42px;
      height: 42px;
      font-weight: 800;
      font-size: 1.05rem;
      cursor: pointer;
      flex-shrink: 0;
    }
    #send-btn:disabled { opacity: 0.4; cursor: default; }
    /* Login Modal */
    #auth-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.96);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 100;
    }
    .auth-box {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      width: 100%;
      max-width: 340px;
      text-align: center;
    }
    .auth-box h2 { margin-bottom: 8px; font-size: 1.2rem; }
    .auth-box p { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }
    .auth-box input {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font-size: 1.2rem;
      text-align: center;
      letter-spacing: 4px;
      margin-bottom: 12px;
    }
    .auth-box button {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: none;
      background: var(--accent);
      color: #0f172a;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
    }
    #auth-err { color: #f87171; font-size: 0.82rem; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand">🛠️ Nukumi <span class="brand-sub" id="manual-count-lbl">Field Support</span></div>
    </div>
    <div class="controls">
      <button class="pill-btn" id="lock-btn" onclick="promptModelLock()">🔓 All Manuals</button>
      <button class="pill-btn" onclick="resetSession()" title="Clear history for new unit">🔄 New Unit</button>
    </div>
  </header>

  <main id="chat">
    <div class="msg bot">
      <p><strong>Field Support Assistant Ready.</strong></p>
      <p>Ask a troubleshooting or wiring question, or tap <strong>🔓 All Manuals</strong> above to lock onto a specific controller (e.g., <code>CIPer 30</code>, <code>KitControl</code>, <code>IRM</code>, <code>WEB-8000</code>).</p>
    </div>
  </main>

  <footer>
    <input id="q-input" type="text" placeholder="Ask a field question or part #..." autocomplete="off" />
    <button id="send-btn" onclick="sendMessage()">➤</button>
  </footer>

  <div id="auth-overlay">
    <div class="auth-box">
      <h2>🔒 Field Access PIN</h2>
      <p>Enter your technician access code to unlock Nukumi.</p>
      <input id="pin-input" type="password" inputmode="numeric" placeholder="••••••" />
      <button onclick="submitPin()">Unlock</button>
      <div id="auth-err"></div>
    </div>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('q-input');
    const sendBtn = document.getElementById('send-btn');
    const lockBtn = document.getElementById('lock-btn');
    const authOverlay = document.getElementById('auth-overlay');
    const pinInput = document.getElementById('pin-input');
    const authErr = document.getElementById('auth-err');
    let currentLock = null;

    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !sendBtn.disabled) sendMessage();
    });
    pinInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submitPin();
    });

    function escapeHtml(str) {
      return str.replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    function renderMarkdown(raw) {
      let text = escapeHtml(raw);
      // Linkify 📄 Source: <filename.pdf> (Page <X>)
      text = text.replace(
        /📄\s*Source:\s*([^\n(]+\.pdf)\s*\(Page\s*(\d+)\)/gi,
        (_, file, page) => {
          const cleanFile = file.trim();
          const url = '/pdf/' + encodeURIComponent(cleanFile) + '#page=' + page;
          return `<a class="pdf-link" href="${url}" target="_blank">📄 ${cleanFile} (Page ${page}) ↗</a>`;
        }
      );
      // Style 🧮 or 📄 Source: Standard Engineering Formula badge
      text = text.replace(
        /(?:🧮|📄)\s*Source:\s*(Standard Engineering Formula[^\n]*)/gi,
        (_, label) => `<span class="pdf-link" style="color:#fcd34d;border-color:rgba(252,211,77,0.4);background:rgba(252,211,77,0.12);">🧮 Source: ${label.trim()}</span>`
      );
      // Clean any raw LaTeX math tokens if emitted by the LLM
      text = text.replace(/\\mathbf\{([^}]*)\}/g, '**$1**');
      text = text.replace(/\\text\{([^}]*)\}/g, '$1');
      text = text.replace(/\\approx/g, '≈').replace(/\\times/g, '×').replace(/\\div/g, '÷');
      text = text.replace(/\$([^$\n]+)\$/g, '$1');
      text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

      const lines = text.split('\n');
      let html = '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (trimmed.startsWith('⚠️')) {
          html += `<div class="warn-box">${trimmed}</div>`;
        } else {
          html += `<p>${trimmed}</p>`;
        }
      }
      return html || '<p>...</p>';
    }

    function renderCitationsDrawer(hits) {
      if (!hits || !hits.length) return '';
      let cards = '';
      hits.forEach((h, i) => {
        const url = '/pdf/' + encodeURIComponent(h.file_name) + '#page=' + h.page;
        cards += `
          <div class="cite-card">
            <div class="cite-meta">
              <a class="pdf-link" style="margin:0;padding:2px 7px;" href="${url}" target="_blank">
                [${i + 1}] 📄 ${escapeHtml(h.file_name)} (p.${h.page}) ↗
              </a>
              <div>
                <span>Hybrid: ${h.hybrid_score.toFixed(2)} | Sem: ${h.semantic_score.toFixed(2)} | BM25: ${h.bm25_score.toFixed(1)}</span>
                <button class="fb-btn" style="margin-left:6px;padding:2px 7px;" onclick="sendFeedback('up', ${i}, this)">🎯 Best Match</button>
              </div>
            </div>
            <div class="cite-text">${escapeHtml(h.paragraph)}</div>
          </div>`;
      });
      return `
        <div class="fb-row">
          <button class="fb-btn" onclick="sendFeedback('up', 0, this)">👍 Helpful</button>
          <button class="fb-btn" onclick="sendFeedback('down', 0, this)">👎 Needs Work</button>
        </div>
        <details class="citations">
          <summary>📚 View ${hits.length} Verbatim Manual Excerpts & Scores</summary>
          ${cards}
        </details>`;
    }

    async function sendFeedback(rating, excerptIdx, btnEl) {
      let note = '';
      if (rating === 'down') {
        const inputNote = prompt('Optional note: What was missing or which controller/manual should this have used?');
        if (inputNote === null) return;
        note = inputNote.trim();
      }
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({rating: rating, excerpt_index: excerptIdx, note: note})
      });
      if (res.ok && btnEl) {
        if (rating === 'up') {
          btnEl.classList.add('up-done');
          btnEl.textContent = '👍 Saved for Training';
        } else {
          btnEl.classList.add('down-done');
          btnEl.textContent = '👎 Logged for Review';
        }
      }
    }

    function updateLockUI(device) {
      currentLock = device;
      if (device) {
        lockBtn.classList.add('active');
        lockBtn.textContent = `🔒 ${device} ×`;
      } else {
        lockBtn.classList.remove('active');
        lockBtn.textContent = '🔓 All Manuals';
      }
    }

    async function checkStatus() {
      const res = await fetch('/api/status');
      if (res.status === 401) {
        authOverlay.style.display = 'flex';
        pinInput.focus();
        return;
      }
      const data = await res.json();
      authOverlay.style.display = 'none';
      if (data.app_name) {
        document.title = `${data.app_name} Field Support`;
      }
      document.getElementById('manual-count-lbl').textContent = `(${data.manual_count} manuals)`;
      updateLockUI(data.active_device);
    }

    async function submitPin() {
      authErr.textContent = '';
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pin: pinInput.value.trim()})
      });
      if (res.ok) {
        pinInput.value = '';
        await checkStatus();
      } else {
        const err = await res.json().catch(() => ({}));
        authErr.textContent = err.error || 'Invalid PIN';
      }
    }

    async function promptModelLock() {
      if (currentLock) {
        // Clear existing lock on tap
        await sendCommand('unlock');
        return;
      }
      const model = prompt('Enter controller or manual name to lock onto (e.g. CIPer 30, KitControl, IRM, WEB-8000):');
      if (model && model.trim()) {
        await sendCommand('lock ' + model.trim());
      }
    }

    async function resetSession() {
      await sendCommand('reset');
      chatEl.innerHTML = `<div class="msg bot"><p>🔄 <strong>Session cleared.</strong> Ready for a new unit.</p></div>`;
    }

    async function sendCommand(cmdText) {
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({command: cmdText})
      });
      if (res.status === 401) {
        authOverlay.style.display = 'flex';
        return;
      }
      const data = await res.json();
      updateLockUI(data.active_device);
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = '';
      sendBtn.disabled = true;

      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = text;
      chatEl.appendChild(userDiv);

      const botDiv = document.createElement('div');
      botDiv.className = 'msg bot';
      const contentDiv = document.createElement('div');
      contentDiv.innerHTML = '<p><em>Searching manuals...</em></p>';
      const citeContainer = document.createElement('div');
      botDiv.appendChild(contentDiv);
      botDiv.appendChild(citeContainer);
      chatEl.appendChild(botDiv);
      chatEl.scrollTop = chatEl.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: text})
        });
        if (res.status === 401) {
          authOverlay.style.display = 'flex';
          sendBtn.disabled = false;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';

        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const parts = buffer.split('\n\n');
          buffer = parts.pop();

          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            const payload = JSON.parse(part.slice(6));
            if (payload.type === 'meta') {
              updateLockUI(payload.active_device);
              citeContainer.innerHTML = renderCitationsDrawer(payload.citations);
            } else if (payload.type === 'token') {
              fullText += payload.text;
              contentDiv.innerHTML = renderMarkdown(fullText);
              chatEl.scrollTop = chatEl.scrollHeight;
            } else if (payload.type === 'error') {
              contentDiv.innerHTML = `<p style="color:#f87171;">${escapeHtml(payload.text)}</p>`;
            }
          }
        }
      } catch (err) {
        contentDiv.innerHTML = `<p style="color:#f87171;">Connection error: ${escapeHtml(String(err))}</p>`;
      } finally {
        sendBtn.disabled = false;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    }

    checkStatus();
  </script>
</body>
</html>
"""


class FieldSupportHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        retriever: SentenceTransformer,
        paragraphs: list[str],
        file_names: list[str],
        pages: list[int],
        paragraph_embeddings: torch.Tensor,
        bm25_index: BM25Index,
        manual_count: int,
        access_pin: str,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.retriever = retriever
        self.paragraphs = paragraphs
        self.file_names = file_names
        self.pages = pages
        self.paragraph_embeddings = paragraph_embeddings
        self.bm25_index = bm25_index
        self.manual_count = manual_count
        self.access_pin = access_pin
        self.expected_token = make_auth_token(access_pin)


class FieldSupportHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Clean, single-line request log without leaking tokens
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {format % args}")

    def _get_client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.client_address[0]

    def _get_cookies(self) -> SimpleCookie:
        cookie = SimpleCookie()
        raw = self.headers.get("Cookie")
        if raw:
            cookie.load(raw)
        return cookie

    def _is_authenticated(self) -> bool:
        cookies = self._get_cookies()
        token_morsel = cookies.get("nukumi_auth")
        if not token_morsel:
            return False
        return hmac.compare_digest(token_morsel.value, self.server.expected_token)

    def _get_or_create_session_id(self) -> tuple[str, bool]:
        cookies = self._get_cookies()
        sid_morsel = cookies.get("nukumi_sid")
        if sid_morsel and len(sid_morsel.value) == 32:
            return sid_morsel.value, False
        return secrets.token_hex(16), True

    def _send_json(self, status: int, data: dict, extra_headers: list[tuple[str, str]] | None = None):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65536:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            body = MOBILE_HTML.encode("utf-8")
            sid, is_new = self._get_or_create_session_id()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if is_new:
                self.send_header(
                    "Set-Cookie",
                    f"nukumi_sid={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
                )
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/manifest.json":
            body = MANIFEST_JSON.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/manifest+json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/icon.svg":
            body = ICON_SVG.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/status":
            if not self._is_authenticated():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"authenticated": False})
                return
            sid, is_new = self._get_or_create_session_id()
            session = get_session(sid)
            headers = []
            if is_new:
                headers.append((
                    "Set-Cookie",
                    f"nukumi_sid={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
                ))
            cfg = load_domain_config()
            self._send_json(
                HTTPStatus.OK,
                {
                    "authenticated": True,
                    "app_name": cfg.get("app_name", "Nukumi"),
                    "manual_count": self.server.manual_count,
                    "active_device": session["active_device"],
                },
                extra_headers=headers,
            )
            return

        if path.startswith("/pdf/"):
            if not self._is_authenticated():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
                return
            raw_name = urllib.parse.unquote(path[len("/pdf/"):])
            safe_name = os.path.basename(raw_name)
            if not safe_name.lower().endswith(".pdf") or safe_name != raw_name:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid PDF filename"})
                return
            full_pdf_path = find_active_pdf_path(pdf_dir, safe_name)
            if not full_pdf_path or not os.path.isfile(full_pdf_path):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "PDF manual not found or excluded"})
                return
            self._serve_pdf_file(full_pdf_path)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _serve_pdf_file(self, full_pdf_path: str):
        """Serve PDF manual with HTTP Range support so mobile browsers can jump directly to #page=X."""
        file_size = os.path.getsize(full_pdf_path)
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            try:
                start_str, end_str = range_header[6:].split("-", 1)
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                if 0 <= start <= end:
                    length = end - start + 1
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(full_pdf_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                    return
            except Exception:
                pass

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with open(full_pdf_path, "rb") as f:
            self.wfile.write(f.read())

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/login":
            client_ip = self._get_client_ip()
            now = time.time()
            recent_fails = [t for t in AUTH_FAILURES.get(client_ip, []) if now - t < LOCKOUT_SECONDS]
            AUTH_FAILURES[client_ip] = recent_fails
            if len(recent_fails) >= MAX_FAILURES:
                self._send_json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "Too many failed attempts. Wait 60 seconds."},
                )
                return

            body = self._read_json_body()
            submitted_pin = str(body.get("pin", "")).strip()
            if hmac.compare_digest(submitted_pin, self.server.access_pin):
                AUTH_FAILURES.pop(client_ip, None)
                sid, _ = self._get_or_create_session_id()
                headers = [
                    (
                        "Set-Cookie",
                        f"nukumi_auth={self.server.expected_token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000",
                    ),
                    (
                        "Set-Cookie",
                        f"nukumi_sid={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
                    ),
                ]
                self._send_json(HTTPStatus.OK, {"ok": True}, extra_headers=headers)
            else:
                AUTH_FAILURES.setdefault(client_ip, []).append(now)
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid access PIN"})
            return

        if not self._is_authenticated():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
            return

        sid, _ = self._get_or_create_session_id()
        session = get_session(sid)

        if path == "/api/command":
            body = self._read_json_body()
            cmd = str(body.get("command", "")).strip()
            cmd_lower = cmd.lower()
            if cmd_lower in {"reset", "new", "clear"}:
                session["chat_history"].clear()
                session["last_user_queries"].clear()
                session["last_hits"].clear()
                session["active_device"] = None
            elif cmd_lower in {"unlock", "unpin"}:
                session["active_device"] = None
            elif cmd_lower.startswith("lock "):
                arg = cmd[5:].strip()
                session["active_device"] = None if arg.lower() in {"off", "clear", "none"} else arg
            self._send_json(HTTPStatus.OK, {"active_device": session["active_device"]})
            return

        if path == "/api/chat":
            body = self._read_json_body()
            user_input = str(body.get("message", "")).strip()
            if not user_input:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty message"})
                return

            self._handle_streaming_chat(session, user_input)
            return

        if path == "/api/feedback":
            body = self._read_json_body()
            rating = str(body.get("rating", "")).strip().lower()
            if rating not in {"up", "down"}:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid rating"})
                return
            hits = session.get("last_hits") or []
            queries = session.get("last_user_queries") or []
            if not hits or not queries:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "No recent query to rate"})
                return

            idx = int(body.get("excerpt_index", 0))
            if idx < 0 or idx >= len(hits):
                idx = 0
            cid = hits[idx]["corpus_id"]
            note = str(body.get("note", "")).strip()
            log_feedback(
                rating=rating,
                user_query=queries[-1],
                search_query=session.get("last_search_query") or queries[-1],
                active_device=session.get("active_device"),
                file_name=self.server.file_names[cid],
                page=self.server.pages[cid],
                paragraph=self.server.paragraphs[cid],
                note=note,
            )
            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _send_sse(self, payload: dict):
        data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _handle_streaming_chat(self, session: dict, user_input: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        cmd = user_input.lower()
        # Handle inline text commands ('unlock', 'lock <model>', 'reset') inside the chat box too
        if cmd in {"unlock", "unpin"}:
            session["active_device"] = None
            self._send_sse({"type": "meta", "active_device": None, "citations": []})
            self._send_sse({
                "type": "token",
                "text": f"🔓 Model lock cleared (searching all {self.server.manual_count} manuals).",
            })
            return

        if (cmd == "lock" or cmd.startswith("lock ")) and not user_input.endswith("?"):
            arg = user_input[4:].strip()
            if len(arg.split()) <= 5:
                session["active_device"] = None if not arg or arg.lower() in {"off", "clear", "none"} else arg
                self._send_sse({"type": "meta", "active_device": session["active_device"], "citations": []})
                msg = (
                    f"🔒 Locked manual filter to **{session['active_device']}**."
                    if session["active_device"]
                    else "🔓 Model lock cleared."
                )
                self._send_sse({"type": "token", "text": msg})
                return

        # 1. Contextualize query and retrieve top 4 Hybrid Search hits
        search_query = build_contextual_search_query(
            user_input, session["active_device"], session["last_user_queries"]
        )
        session["last_search_query"] = search_query
        hits = retrieve_context_hits(
            search_query=search_query,
            model=self.server.retriever,
            paragraph_embeddings=self.server.paragraph_embeddings,
            bm25_index=self.server.bm25_index,
            file_names=self.server.file_names,
            active_device=session["active_device"],
            top_k=4,
        )
        session["last_hits"] = hits

        citations_payload = [
            {
                "file_name": self.server.file_names[h["corpus_id"]],
                "page": int(self.server.pages[h["corpus_id"]]),
                "paragraph": self.server.paragraphs[h["corpus_id"]],
                "hybrid_score": round(float(h["hybrid_score"]), 3),
                "semantic_score": round(float(h["semantic_score"]), 3),
                "bm25_score": round(float(h["bm25_score"]), 1),
            }
            for h in hits
        ]
        self._send_sse({
            "type": "meta",
            "active_device": session["active_device"],
            "citations": citations_payload,
        })

        # 2. Build RAG prompt & re-read Modelfile dynamically
        rag_user_prompt = format_rag_prompt(
            user_message=user_input,
            hits=hits,
            paragraphs=self.server.paragraphs,
            file_names=self.server.file_names,
            pages=self.server.pages,
            active_device=session["active_device"],
        )
        ollama_model, temperature, num_ctx, system_prompt = load_modelfile_config(modelfile_path)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session["chat_history"][-4:])
        messages.append({"role": "user", "content": rag_user_prompt})

        reply_parts = []
        try:
            stream = ollama.chat(
                model=ollama_model,
                messages=messages,
                stream=True,
                options={"temperature": temperature, "num_ctx": num_ctx},
            )
            for chunk in stream:
                token_text = chunk.get("message", {}).get("content", "")
                if token_text:
                    reply_parts.append(token_text)
                    self._send_sse({"type": "token", "text": token_text})
        except Exception as e:
            self._send_sse({"type": "error", "text": f"Ollama error ({ollama_model}): {e}"})
            return

        full_reply = "".join(reply_parts).strip()
        if full_reply:
            session["chat_history"].append({"role": "user", "content": user_input})
            session["chat_history"].append({"role": "assistant", "content": full_reply})
            session["last_user_queries"].append(user_input)


def main():
    if not os.path.exists(model_path):
        print(f"Error: Search model not found at {model_path}. Run train_st.py first.")
        return

    corpus_dataset = load_or_build_corpus()
    if corpus_dataset is None or len(corpus_dataset) == 0:
        print("Error: Search corpus is empty.")
        return

    access_pin = load_or_create_access_pin()

    retriever_device = "cpu" if os.environ.get("CPU_RETRIEVER", "0") == "1" else None
    print(f"Loading ModernBERT search retriever from {model_path}...")
    retriever = SentenceTransformer(model_path, device=retriever_device)

    paragraphs = list(corpus_dataset["paragraph"])
    file_names = list(corpus_dataset["file_name"])
    pages = list(corpus_dataset["page"])
    manual_count = get_manual_count(file_names)

    paragraph_embeddings = load_or_compute_embeddings(retriever, paragraphs)
    if retriever.device.type == "cpu":
        paragraph_embeddings = paragraph_embeddings.cpu()
    paragraphs, file_names, pages, paragraph_embeddings = filter_active_corpus(
        paragraphs, file_names, pages, paragraph_embeddings
    )
    manual_count = get_manual_count(file_names)

    print(f"Building BM25 lexical index over {len(paragraphs)} paragraphs ({manual_count} manuals)...")
    bm25_index = BM25Index(paragraphs, file_names=file_names)

    server = FieldSupportHTTPServer(
        (HOST, PORT),
        FieldSupportHandler,
        retriever=retriever,
        paragraphs=paragraphs,
        file_names=file_names,
        pages=pages,
        paragraph_embeddings=paragraph_embeddings,
        bm25_index=bm25_index,
        manual_count=manual_count,
        access_pin=access_pin,
    )

    print("\n" + "=" * 65)
    print("🚀 NUKUMI MOBILE PWA SERVER READY")
    print("=" * 65)
    print(f"  Local Address : http://{HOST}:{PORT}")
    print(f"  Access PIN    : {access_pin}  (saved in .access_pin — edit anytime)")
    print("-" * 65)
    print("  To expose privately to your Tailscale devices only:")
    print(f"    tailscale serve {PORT}")
    print("  To expose via public HTTPS URL for testers WITHOUT Tailscale app:")
    print(f"    tailscale funnel {PORT}")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Nukumi web server. Goodbye!")
        server.server_close()


if __name__ == "__main__":
    main()
