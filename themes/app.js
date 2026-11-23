// Shared Frontend Logic for Nukumi HTML Themes (themes/*.html)
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('q-input');
const sendBtn = document.getElementById('send-btn');
const lockBtn = document.getElementById('lock-btn');
const drawerLockBtn = document.getElementById('drawer-lock-btn');
const authOverlay = document.getElementById('auth-overlay');
const pinInput = document.getElementById('pin-input');
const authErr = document.getElementById('auth-err');
const drawerEl = document.getElementById('drawer');
const drawerBackdrop = document.getElementById('drawer-backdrop');
const themeSelect = document.getElementById('theme-select');
let currentLock = null;

if (inputEl) {
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !sendBtn.disabled) sendMessage();
  });
}
if (pinInput) {
  pinInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitPin();
  });
}

function toggleDrawer(forceOpen) {
  if (!drawerEl || !drawerBackdrop) return;
  const isOpen = forceOpen !== undefined ? forceOpen : !drawerEl.classList.contains('open');
  drawerEl.classList.toggle('open', isOpen);
  drawerBackdrop.classList.toggle('open', isOpen);
}

function changeTheme(themeName) {
  if (!themeName) return;
  document.cookie = `nukumi_theme=${encodeURIComponent(themeName)}; Path=/; SameSite=Strict; Max-Age=31536000`;
  window.location.href = '/?theme=' + encodeURIComponent(themeName);
}

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
      return `<a class="pdf-link" href="${url}" target="_blank" rel="noopener">📄 ${cleanFile} (Page ${page}) ↗</a>`;
    }
  );
  // Style 🧮 or 📄 Source: Standard Engineering Formula badge
  text = text.replace(
    /(?:🧮|📄)\s*Source:\s*(Standard Engineering Formula[^\n]*)/gi,
    (_, label) => `<span class="pdf-link formula-badge">🧮 Source: ${label.trim()}</span>`
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
      html += `<div class="warn-box" role="alert">${trimmed}</div>`;
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
          <a class="pdf-link" style="margin:0;padding:3px 8px;" href="${url}" target="_blank" rel="noopener">
            [${i + 1}] 📄 ${escapeHtml(h.file_name)} (p.${h.page}) ↗
          </a>
          <div class="cite-scores">
            <span>Hybrid: ${h.hybrid_score.toFixed(2)} | Sem: ${h.semantic_score.toFixed(2)} | BM25: ${h.bm25_score.toFixed(1)}</span>
            <button class="fb-btn" style="margin-left:6px;padding:3px 8px;" onclick="sendFeedback('up', ${i}, this)">🎯 Best Match</button>
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
      <summary>📚 View ${hits.length} Verbatim Manual Excerpts &amp; Scores</summary>
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
  if (lockBtn) {
    if (device) {
      lockBtn.classList.add('active');
      lockBtn.textContent = `🔒 ${device} ×`;
    } else {
      lockBtn.classList.remove('active');
      lockBtn.textContent = '🔓 All Manuals';
    }
  }
  if (drawerLockBtn) {
    drawerLockBtn.textContent = device ? `🔒 Unlock (${device})` : '🔒 Lock Controller Model...';
  }
}

async function checkStatus() {
  const res = await fetch('/api/status');
  if (res.status === 401) {
    if (authOverlay) authOverlay.style.display = 'flex';
    if (pinInput) pinInput.focus();
    return;
  }
  const data = await res.json();
  if (authOverlay) authOverlay.style.display = 'none';
  if (data.app_name) {
    document.title = `${data.app_name} Field Support`;
    const brandNameEl = document.getElementById('brand-name');
    if (brandNameEl) brandNameEl.textContent = data.app_name;
  }
  const manualLbl = document.getElementById('manual-count-lbl');
  if (manualLbl) manualLbl.textContent = `(${data.manual_count} manuals)`;

  if (themeSelect && Array.isArray(data.themes)) {
    themeSelect.innerHTML = '';
    data.themes.forEach((t) => {
      const opt = document.createElement('option');
      opt.value = t.id;
      opt.textContent = t.label;
      if (t.id === data.active_theme) opt.selected = true;
      themeSelect.appendChild(opt);
    });
  }
  updateLockUI(data.active_device);
}

async function submitPin() {
  if (authErr) authErr.textContent = '';
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
    if (authErr) authErr.textContent = err.error || 'Invalid PIN';
  }
}

async function promptModelLock() {
  toggleDrawer(false);
  if (currentLock) {
    await sendCommand('unlock');
    return;
  }
  const model = prompt('Enter controller or manual name to lock onto (e.g. CIPer 30, KitControl, IRM, WEB-8000):');
  if (model && model.trim()) {
    await sendCommand('lock ' + model.trim());
  }
}

async function resetSession() {
  toggleDrawer(false);
  await sendCommand('reset');
  chatEl.innerHTML = `<article class="msg bot"><p>🔄 <strong>Session cleared.</strong> Ready for a new unit.</p></article>`;
}

async function sendCommand(cmdText) {
  const res = await fetch('/api/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({command: cmdText})
  });
  if (res.status === 401) {
    if (authOverlay) authOverlay.style.display = 'flex';
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

  const userDiv = document.createElement('article');
  userDiv.className = 'msg user';
  userDiv.textContent = text;
  chatEl.appendChild(userDiv);

  const botDiv = document.createElement('article');
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
      if (authOverlay) authOverlay.style.display = 'flex';
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
