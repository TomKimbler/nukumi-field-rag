"""
Nukumi Mobile & Tablet PWA Field Support Server (Tailscale Serve / Funnel Ready).

Zero-external-dependency HTTP server (Python stdlib `http.server.ThreadingHTTPServer`)
that wraps the Hybrid Search (ModernBERT + BM25 RRF) and Ollama RAG assistant into a
Progressive Web App (PWA) with hot-swappable HTML themes in `themes/*.html`.

Security features:
- Binds to 127.0.0.1:8000 by default (accessible only via local Tailscale Funnel/Serve proxy).
- Passcode / PIN gate stored in `.access_pin` (or `ACCESS_PIN` env var) with HMAC-SHA256 cookies.
- Brute-force rate limiter (locks out after 5 failed PIN attempts for 60 seconds).
- Strict path-traversal protection on `/pdf/<filename>` and `themes/<theme>.html`.
"""

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
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
from optiDataset import find_active_pdf_path, is_excluded_name, iter_active_pdf_files
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
secret_file_path = os.path.join(project_root, ".session_secret")
themes_dir = os.path.join(project_root, "themes")
DEFAULT_THEME = "slate-dark"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Per-session state store (keyed by session_id cookie)
SESSION_LOCK = threading.Lock()
SESSIONS: dict[str, dict] = {}
SESSION_TTL_SECONDS = 86400.0  # Prune idle sessions after 24 hours
MAX_SAVED_TURNS = 25

# Brute-force rate limiting store (keyed by client IP / X-Forwarded-For)
AUTH_FAILURES: dict[str, list[float]] = {}
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60.0


def load_or_create_session_secret() -> bytes:
    """Persist the HMAC signing secret in .session_secret so logins survive server restarts."""
    if os.path.exists(secret_file_path):
        try:
            with open(secret_file_path, "r", encoding="utf-8") as f:
                hex_key = f.read().strip()
                if len(hex_key) >= 64:
                    return bytes.fromhex(hex_key)
        except Exception:
            pass

    new_secret = secrets.token_bytes(32)
    with open(secret_file_path, "w", encoding="utf-8") as f:
        f.write(new_secret.hex() + "\n")
    os.chmod(secret_file_path, 0o600)
    return new_secret


SERVER_SECRET = load_or_create_session_secret()


def list_available_themes() -> list[dict[str, str]]:
    """
    Scan `themes/*.html` for available HTML themes (ignoring files prefixed with
    `ignore_`, `exclude_`, or `.`).
    """
    if not os.path.exists(themes_dir):
        return []

    theme_ids = []
    for fname in sorted(os.listdir(themes_dir)):
        if is_excluded_name(fname):
            continue
        if fname.lower().endswith(".html"):
            theme_ids.append(os.path.splitext(fname)[0])

    # Keep DEFAULT_THEME first if present
    if DEFAULT_THEME in theme_ids:
        theme_ids.remove(DEFAULT_THEME)
        theme_ids.insert(0, DEFAULT_THEME)

    return [
        {"id": tid, "label": tid.replace("-", " ").replace("_", " ").title()}
        for tid in theme_ids
    ]


def resolve_theme_html(requested_theme: str | None) -> tuple[str, bytes]:
    """
    Resolve a requested theme ID to a valid `themes/<id>.html` file on disk,
    falling back to `slate-dark.html` or the first available `.html` theme.
    """
    available = [t["id"] for t in list_available_themes()]
    if not available:
        return "none", b"<h1>Error: No .html themes found in themes/ directory.</h1>"

    chosen = requested_theme if (requested_theme in available) else available[0]
    theme_path = os.path.join(themes_dir, f"{chosen}.html")
    with open(theme_path, "rb") as f:
        return chosen, f.read()


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
    now = time.time()
    with SESSION_LOCK:
        # Prune stale sessions to prevent unbounded memory growth on long-running services
        expired = [sid for sid, s in SESSIONS.items() if now - s.get("updated_at", now) > SESSION_TTL_SECONDS]
        for sid in expired:
            SESSIONS.pop(sid, None)

        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                "chat_history": [],
                "last_user_queries": [],
                "last_search_query": "",
                "last_hits": [],
                "turns": {},
                "active_device": None,
                "theme": DEFAULT_THEME,
                "updated_at": now,
            }
        SESSIONS[session_id]["updated_at"] = now
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
        # Cache active PDF paths at startup so HTTP Range requests don't re-walk pdfFiles/
        self.active_pdf_map = {fname: fpath for fname, fpath in iter_active_pdf_files(pdf_dir)}


class FieldSupportHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
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
            try:
                cookie.load(raw)
            except CookieError:
                pass
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

    def _get_preferred_theme(self, query_params: dict[str, list[str]], session: dict) -> str:
        if "theme" in query_params and query_params["theme"]:
            return query_params["theme"][0].strip()
        cookies = self._get_cookies()
        theme_morsel = cookies.get("nukumi_theme")
        if theme_morsel and theme_morsel.value:
            return urllib.parse.unquote(theme_morsel.value)
        return session.get("theme", DEFAULT_THEME)

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
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return {}
        if length <= 0 or length > 65536:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            sid, is_new = self._get_or_create_session_id()
            session = get_session(sid)
            preferred_theme = self._get_preferred_theme(query_params, session)
            active_theme, body = resolve_theme_html(preferred_theme)
            session["theme"] = active_theme

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            if is_new:
                self.send_header(
                    "Set-Cookie",
                    f"nukumi_sid={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
                )
            if "theme" in query_params:
                self.send_header(
                    "Set-Cookie",
                    f"nukumi_theme={active_theme}; Path=/; SameSite=Strict; Max-Age=31536000",
                )
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/app.js":
            js_path = os.path.join(themes_dir, "app.js")
            if not os.path.isfile(js_path):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "app.js not found"})
                return
            with open(js_path, "rb") as f:
                body = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
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
            available_themes = list_available_themes()
            preferred = self._get_preferred_theme(query_params, session)
            valid_ids = {t["id"] for t in available_themes}
            active_theme = preferred if preferred in valid_ids else (available_themes[0]["id"] if available_themes else DEFAULT_THEME)

            self._send_json(
                HTTPStatus.OK,
                {
                    "authenticated": True,
                    "app_name": cfg.get("app_name", "Nukumi"),
                    "manual_count": self.server.manual_count,
                    "active_device": session["active_device"],
                    "active_theme": active_theme,
                    "themes": available_themes,
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
            full_pdf_path = self.server.active_pdf_map.get(safe_name)
            if not full_pdf_path or not os.path.isfile(full_pdf_path):
                full_pdf_path = find_active_pdf_path(pdf_dir, safe_name)
                if full_pdf_path:
                    self.server.active_pdf_map[safe_name] = full_pdf_path
            if not full_pdf_path or not os.path.isfile(full_pdf_path):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "PDF manual not found or excluded"})
                return
            self._serve_pdf_file(full_pdf_path)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _serve_pdf_file(self, full_pdf_path: str):
        """Serve PDF manual with HTTP Range support so mobile/tablet browsers can jump directly to #page=X."""
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
                session["turns"].clear()
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

            turn_id = str(body.get("turn_id", "")).strip()
            turn = session.get("turns", {}).get(turn_id) if turn_id else None
            if turn:
                hits = turn["hits"]
                user_query = turn["user_query"]
                search_query = turn["search_query"]
                active_device = turn["active_device"]
            else:
                hits = session.get("last_hits") or []
                queries = session.get("last_user_queries") or []
                if not hits or not queries:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "No recent query to rate"})
                    return
                user_query = queries[-1]
                search_query = session.get("last_search_query") or user_query
                active_device = session.get("active_device")

            is_formula = bool(turn.get("is_formula")) if turn else False
            explicit_excerpt = body.get("explicit_excerpt", False)
            try:
                idx = int(body.get("excerpt_index", 0))
            except (ValueError, TypeError):
                idx = 0
            if idx < 0 or idx >= len(hits):
                idx = 0
            note = str(body.get("note", "")).strip()

            # Guard retriever training: a thumbs-up on a pure engineering-formula answer
            # should not pair a math question with an unrelated manual excerpt.
            if rating == "up" and is_formula and not explicit_excerpt:
                log_feedback(
                    rating="up_formula",
                    user_query=user_query,
                    search_query=search_query,
                    active_device=active_device,
                    file_name="Standard Engineering Formula",
                    page="",
                    paragraph="",
                    note=note,
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "formula_only": True})
                return

            cid = hits[idx]["corpus_id"]
            log_feedback(
                rating=rating,
                user_query=user_query,
                search_query=search_query,
                active_device=active_device,
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
        turn_id = secrets.token_hex(6)
        turns = session["turns"]
        turns[turn_id] = {
            "user_query": user_input,
            "search_query": search_query,
            "active_device": session["active_device"],
            "hits": hits,
        }
        while len(turns) > MAX_SAVED_TURNS:
            turns.pop(next(iter(turns)))

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
            "turn_id": turn_id,
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
            if turn_id in turns:
                turns[turn_id]["is_formula"] = "Standard Engineering Formula" in full_reply
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

    paragraph_embeddings = load_or_compute_embeddings(retriever, paragraphs)
    if retriever.device.type == "cpu":
        paragraph_embeddings = paragraph_embeddings.cpu()
    paragraphs, file_names, pages, paragraph_embeddings = filter_active_corpus(
        paragraphs, file_names, pages, paragraph_embeddings
    )
    manual_count = get_manual_count(file_names)

    print(f"Building BM25 lexical index over {len(paragraphs)} paragraphs ({manual_count} manuals)...")
    bm25_index = BM25Index(paragraphs, file_names=file_names)

    themes = [t["id"] for t in list_available_themes()]
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
    print("🚀 NUKUMI MOBILE & TABLET PWA SERVER READY")
    print("=" * 65)
    print(f"  Local Address : http://{HOST}:{PORT}")
    print(f"  Access PIN    : {access_pin}  (saved in .access_pin — edit anytime)")
    print(f"  HTML Themes   : {', '.join(themes)}  (from themes/*.html)")
    print("-" * 65)
    print("  To expose privately to your Tailscale devices only:")
    print(f"    tailscale serve {PORT}        (add --bg to run in background)")
    print("  To expose via public HTTPS URL for testers WITHOUT Tailscale app:")
    print(f"    tailscale funnel {PORT}       (add --bg to run in background)")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Nukumi web server. Goodbye!")
        server.server_close()


if __name__ == "__main__":
    main()
