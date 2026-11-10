/* ==========================================================================
   Research Desk — frontend logic (no build step, no dependencies)

   How it talks to the backend (all endpoints already exist in src/app.py):
     GET    /health                       service status
     GET    /documents                    report library
     POST   /documents/upload             add a PDF (progress is real, via XHR)
     DELETE /documents/{id}               remove a report
     POST   /documents/reindex            rebuild the vector index
     GET    /documents/{id}/pages/{n}     extracted text behind a citation
     GET    /documents/{id}/file          the original PDF
     GET    /chat/stream                  ask a question (Server-Sent Events)
     GET    /conversations                saved threads
     GET    /conversations/{id}/messages  one thread (added in this change)
     POST   /feedback                     thumbs up / down

   Sections: 1 Setup · 2 Helpers · 3 Library · 4 Conversations · 5 Chat
             6 Answer rendering · 7 Inspector · 8 Wiring
   ========================================================================== */
(() => {
  'use strict';

  /* 1. Setup ---------------------------------------------------------------- */

  // Same origin by default. To use a different backend: open /?api=http://host:8000
  const API = (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '');

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const PATHS = {
    plus: 'M12 5v14M5 12h14',
    upload: 'M12 15V4M7 9l5-5 5 5M4 15v4a1 1 0 001 1h14a1 1 0 001-1v-4',
    trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6',
    file: 'M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8zM14 3v5h5',
    scope: 'M12 3v4M12 17v4M3 12h4M17 12h4M12 8.5a3.5 3.5 0 100 7 3.5 3.5 0 000-7z',
    send: 'M12 19V5M6 11l6-6 6 6',
    stop: 'M8 7h8a1 1 0 011 1v8a1 1 0 01-1 1H8a1 1 0 01-1-1V8a1 1 0 011-1z',
    copy: 'M9 9h10a1 1 0 011 1v10a1 1 0 01-1 1H9a1 1 0 01-1-1V10a1 1 0 011-1zM5 15V5a1 1 0 011-1h10',
    up: 'M7 11v9H4v-9zM7 11l4-8a2 2 0 012 2v4h5.5a2 2 0 012 2.3l-1 6A2 2 0 0117.5 20H7',
    down: 'M17 13V4h3v9zM17 13l-4 8a2 2 0 01-2-2v-4H5.5a2 2 0 01-2-2.3l1-6A2 2 0 016.5 4H17',
    external: 'M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5',
    chevron: 'M9 6l6 6-6 6',
    left: 'M15 6l-6 6 6 6',
    x: 'M6 6l12 12M18 6L6 18',
    refresh: 'M20 11a8 8 0 10-2.3 5.7M20 5v6h-6',
    menu: 'M4 7h16M4 12h16M4 17h16',
    panel: 'M4 5h16v14H4zM15 5v14',
    check: 'M5 12l5 5 9-10',
    alert: 'M12 8v5M12 16.5h.01M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z'
  };
  const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="${PATHS[name]}"/></svg>`;

  const SUGGESTIONS = [
    'What was the total power generation capacity of RWE AG?',
    'Which leadership positions changed at Origin Bancorp, Inc.?',
    'Did HCA Healthcare announce any changes to its dividend policy?',
    'Which had the lowest net income in EUR: Atreca, INMUNE BIO, Datalogic, NuCana or RWE?'
  ];

  const S = {
    user: userId(),
    mode: load('rd_mode', 'agent'),
    scope: null,                 // document id to search in, or null for all
    docs: [], docsError: null,
    uploads: [],                 // files being uploaded right now
    convs: [],
    convId: newConvId(),
    title: 'New conversation',
    msgs: [],                    // {role, content, citations, steps, usage, id, ...}
    busy: false, abort: null, timer: null,
    active: null,                // index of the answer shown in the inspector
    tab: 'source',
    source: null,                // {doc, page, view, passages, error, loading}
    health: null, healthError: false,
    filter: ''
  };

  /* 2. Helpers -------------------------------------------------------------- */

  function load(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch { return fallback; } }
  function save(key, value) { try { localStorage.setItem(key, value); } catch { /* storage may be blocked */ } }
  function userId() {
    let id = load('rd_user', '');
    if (!id) { id = 'web_' + Math.random().toString(36).slice(2, 10); save('rd_user', id); }
    return id;
  }
  function newConvId() { return 'conv_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5); }

  const enc = encodeURIComponent;
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtInt = (n) => Number(n || 0).toLocaleString('en-US');
  const fmtUsd = (n) => '$' + Number(n || 0).toFixed(Number(n) >= 1 ? 2 : 4);
  const fmtSec = (s) => (s < 10 ? s.toFixed(1) : Math.round(s)) + 's';
  const fmtBytes = (b) => !b ? '' : b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB';
  const shortDoc = (name, max = 26) => {
    const base = String(name).replace(/\.pdf$/i, '');
    return base.length > max ? base.slice(0, max - 1).trimEnd() + '…' : base;
  };
  function relTime(ts) {
    if (!ts) return '';
    const d = new Date(String(ts).replace(' ', 'T') + (String(ts).includes('Z') ? '' : 'Z'));
    if (isNaN(d)) return '';
    const mins = Math.round((Date.now() - d) / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return mins + ' min ago';
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return hrs + ' h ago';
    const days = Math.round(hrs / 24);
    return days === 1 ? 'Yesterday' : days < 30 ? days + ' days ago' : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  }

  class ApiError extends Error { constructor(status, message) { super(message); this.status = status; } }

  async function api(path, opts = {}) {
    let res;
    try {
      res = await fetch(API + path, { ...opts, headers: { 'X-User-Id': S.user, ...(opts.headers || {}) } });
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      throw new ApiError(0, 'Cannot reach the server. Check that the backend is running.');
    }
    if (!res.ok) throw await httpError(res);
    return res.status === 204 ? null : res.json();
  }
  async function httpError(res) {
    let detail = '';
    try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j); }
    catch { detail = res.statusText; }
    return new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  function friendly(e) {
    if (!(e instanceof ApiError)) return e.message || 'Something went wrong.';
    if (e.status === 0) return e.message;
    if (e.status === 429) return `${e.message} Raise MAX_USER_BUDGET_USD in .env to continue.`;
    if (e.status === 503) return `The language model is unavailable. Check GEMINI_API_KEY or GROQ_API_KEY in .env, then try again. (${e.message})`;
    return e.message;
  }

  function toast(message, type = 'ok') {
    const el = document.createElement('div');
    el.className = 'toast' + (type === 'err' ? ' err' : '');
    el.innerHTML = icon(type === 'err' ? 'alert' : 'check') + `<span>${esc(message)}</span>`;
    $('#toasts').append(el);
    setTimeout(() => el.remove(), type === 'err' ? 7000 : 3800);
  }

  function confirmDialog({ title, body, ok }) {
    const dlg = $('#confirmDialog');
    $('#confirmTitle').textContent = title;
    $('#confirmBody').textContent = body;
    $('#confirmOk').textContent = ok;
    return new Promise((resolve) => {
      dlg.returnValue = '';
      dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
      dlg.showModal();
    });
  }

  const readyDocs = () => S.docs.filter((d) => d.status === 'ready');
  const findDoc = (id) => S.docs.find((d) => d.id === id);

  /* 3. Library -------------------------------------------------------------- */

  let pollTimer = null;

  async function loadDocs() {
    try { S.docs = await api('/documents'); S.docsError = null; }
    catch (e) { S.docsError = friendly(e); }
    // An upload is done tracking once the backend lists that file.
    S.uploads = S.uploads.filter((u) => !(u.state === 'queued' && S.docs.some((d) => d.filename === u.name)));
    if (S.scope && !findDoc(S.scope)) S.scope = null;
    renderDocs(); renderScope(); renderHello();
    clearTimeout(pollTimer);
    const working = S.docs.some((d) => d.status === 'pending' || d.status === 'processing') || S.uploads.some((u) => u.state === 'queued');
    if (working) pollTimer = setTimeout(loadDocs, 2500);
  }

  function renderDocs() {
    const list = $('#docList');
    const total = S.docs.length;
    $('#libCount').textContent = total ? `${readyDocs().length} of ${total} ready` : '';
    $('#docFilterWrap').hidden = total < 9;

    if (S.docsError) { list.innerHTML = `<li class="list-empty">${esc(S.docsError)}</li>`; return; }

    const q = S.filter.trim().toLowerCase();
    const items = [];

    S.uploads.forEach((u) => {
      const status = u.state === 'failed' ? 'failed' : u.state === 'queued' ? 'pending' : 'uploading';
      const label = u.state === 'failed' ? esc(u.error || 'Upload failed') : u.state === 'queued' ? 'Queued for processing' : `Uploading ${u.pct}%`;
      items.push(`<li class="doc">
        <span class="doc-ico">${icon('file')}</span>
        <span class="doc-name" title="${esc(u.name)}">${esc(u.name)}</span>
        <span class="doc-meta"><i class="dot ${status}"></i>${label}</span>
        ${u.state === 'failed' ? '' : `<div class="bar ${u.state === 'queued' ? 'indeterminate' : ''}"><i style="width:${u.state === 'queued' ? 40 : u.pct}%"></i></div>`}
      </li>`);
    });

    S.docs.filter((d) => !q || d.filename.toLowerCase().includes(q)).forEach((d) => {
      const ready = d.status === 'ready';
      const meta = ready
        ? `${fmtInt(d.page_count)} pages${d.file_size_bytes ? ' · ' + fmtBytes(d.file_size_bytes) : ''}`
        : d.status === 'failed' ? 'Processing failed' : d.status === 'processing' ? 'Reading pages…' : 'Waiting to process';
      const current = S.scope === d.id;
      items.push(`<li class="doc" data-id="${esc(enc(d.id))}" ${current ? 'aria-current="true"' : ''}>
        <span class="doc-ico">${icon('file')}</span>
        <span class="doc-name" title="${esc(d.filename)}">${esc(d.filename)}</span>
        <span class="doc-actions">
          ${ready ? `<button class="icon-btn ${current ? 'on' : ''}" data-act="scope" aria-pressed="${current}" aria-label="Search only in ${esc(d.filename)}" title="Search only in this report">${icon('scope')}</button>
          <a class="icon-btn" href="${API}/documents/${enc(d.id)}/file" target="_blank" rel="noopener" aria-label="Open ${esc(d.filename)}" title="Open PDF">${icon('external')}</a>` : ''}
          <button class="icon-btn" data-act="delete" aria-label="Delete ${esc(d.filename)}" title="Delete report">${icon('trash')}</button>
        </span>
        <span class="doc-meta"><i class="dot ${esc(d.status)}" role="img" aria-label="${esc(d.status)}" title="${esc(d.status)}"></i>${esc(meta)}</span>
        ${d.status === 'failed' && d.error_message ? `<span class="doc-error">${esc(d.error_message)}</span>` : ''}
      </li>`);
    });

    list.innerHTML = items.length ? items.join('') : '<li class="list-empty">No reports yet. Add a PDF to begin.</li>';
  }

  function uploadFile(file) {
    if (!/\.pdf$/i.test(file.name)) { toast(`${file.name} is not a PDF.`, 'err'); return; }
    const up = { name: file.name, pct: 0, state: 'uploading' };
    S.uploads.push(up);
    renderDocs();

    const xhr = new XMLHttpRequest();
    xhr.open('POST', API + '/documents/upload');
    xhr.setRequestHeader('X-User-Id', S.user);
    let frame = 0;
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable || frame) return;
      frame = requestAnimationFrame(() => { frame = 0; up.pct = Math.round((e.loaded / e.total) * 100); renderDocs(); });
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        up.state = 'queued'; up.pct = 100;
        toast(`${file.name} uploaded. Reading pages in the background.`);
        loadDocs();
      } else {
        let detail = 'Upload failed.';
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* keep default */ }
        up.state = 'failed'; up.error = detail; renderDocs();
        toast(`${file.name}: ${detail}`, 'err');
      }
    };
    xhr.onerror = () => { up.state = 'failed'; up.error = 'Cannot reach the server.'; renderDocs(); toast('Upload failed: cannot reach the server.', 'err'); };
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  }

  async function deleteDoc(id) {
    const doc = findDoc(id);
    if (!doc) return;
    const ok = await confirmDialog({
      title: 'Delete this report?',
      body: `${doc.filename} will be removed from the library, the search index and the server. This cannot be undone.`,
      ok: 'Delete report'
    });
    if (!ok) return;
    try {
      await api(`/documents/${enc(id)}`, { method: 'DELETE' });
      if (S.scope === id) S.scope = null;
      if (S.source && S.source.doc === id) { S.source = null; renderSource(); }
      toast('Report deleted.');
      loadDocs();
    } catch (e) { toast(friendly(e), 'err'); }
  }

  async function reindex() {
    const btn = $('#reindexBtn');
    btn.disabled = true;
    try { await api('/documents/reindex', { method: 'POST' }); toast('Reindex started. This runs in the background.'); }
    catch (e) { toast(friendly(e), 'err'); }
    finally { setTimeout(() => { btn.disabled = false; }, 2500); }
  }

  function setScope(id) {
    S.scope = id || null;
    renderDocs(); renderScope();
  }

  function renderScope() {
    const sel = $('#scopeSelect');
    sel.innerHTML = '<option value="">All reports</option>' + readyDocs().map((d) =>
      `<option value="${esc(d.id)}" ${S.scope === d.id ? 'selected' : ''}>${esc(shortDoc(d.filename, 34))}</option>`).join('');
    sel.value = S.scope || '';
    const line = $('#scopeLine');
    if (S.scope) {
      line.innerHTML = `Searching only ${esc(shortDoc(S.scope, 40))} <button type="button" id="clearScope">Search all reports</button>`;
    } else {
      const n = readyDocs().length;
      line.textContent = n ? `Searching all ${n} report${n === 1 ? '' : 's'}` : 'No reports ready yet';
    }
  }

  function renderHealth() {
    const el = $('#health');
    if (!S.health && !S.healthError) {
      el.className = 'health';
      el.innerHTML = '<i class="dot pending"></i><span>Connecting…</span>';
      return;
    }
    if (S.healthError || !S.health) {
      el.className = 'health off';
      el.innerHTML = '<i class="dot failed"></i><span>Server unreachable</span>';
      return;
    }
    const h = S.health;
    el.className = 'health';
    el.innerHTML = `<i class="dot ready"></i><span>Connected · ${fmtInt(h.ready_documents)} reports${h.vector_indexed_chunks != null ? ' · ' + fmtInt(h.vector_indexed_chunks) + ' chunks' : ''}</span>`;
  }
  async function pollHealth() {
    try { S.health = await api('/health'); S.healthError = false; } catch { S.healthError = true; }
    renderHealth();
    setTimeout(pollHealth, 20000);
  }

  /* 4. Conversations -------------------------------------------------------- */

  async function loadConvs() {
    try { S.convs = await api(`/conversations?user_id=${enc(S.user)}`); } catch { /* keep the old list */ }
    renderConvs();
  }

  function renderConvs() {
    const list = $('#convList');
    if (!S.convs.length) { list.innerHTML = '<li class="list-empty">Your questions will be saved here.</li>'; return; }
    list.innerHTML = S.convs.map((c) => `<li><button class="conv" data-id="${esc(c.id)}" ${c.id === S.convId ? 'aria-current="true"' : ''}>
      <b>${esc(c.title || 'Untitled')}</b><small>${esc(relTime(c.updated_at))}</small></button></li>`).join('');
  }

  async function openConv(id) {
    if (S.busy) return;
    let rows;
    try { rows = (await api(`/conversations/${enc(id)}/messages`)).messages; }
    catch (e) {
      if (e.status !== 404 && e.status !== 405) { toast(friendly(e), 'err'); return; }
      try { rows = (await api(`/conversations/${enc(id)}`)).messages; } catch (e2) { toast(friendly(e2), 'err'); return; }
    }
    S.convId = id;
    S.msgs = rows.map((r) => ({
      role: r.role, content: r.content, id: r.id || null,
      citations: r.citations || [], steps: r.agent_steps || [],
      usage: r.usage || null
    }));
    const first = S.convs.find((c) => c.id === id);
    S.title = (first && first.title) || (S.msgs[0] && S.msgs[0].content.slice(0, 60)) || 'Conversation';
    S.active = lastAnswerIndex();
    closeDrawers();
    renderAll();
  }

  function newConversation() {
    if (S.busy) return;
    S.convId = newConvId(); S.msgs = []; S.title = 'New conversation'; S.active = null; S.source = null;
    closeDrawers();
    renderAll();
    $('#composerInput').focus();
  }

  const lastAnswerIndex = () => { for (let i = S.msgs.length - 1; i >= 0; i--) if (S.msgs[i].role === 'assistant') return i; return null; };

  /* 5. Chat ----------------------------------------------------------------- */

  async function ask(question) {
    question = question.trim();
    if (!question || S.busy) return;

    if (!S.msgs.length) S.title = question.length > 60 ? question.slice(0, 57) + '…' : question;
    S.msgs.push({ role: 'user', content: question });
    const answer = { role: 'assistant', content: '', citations: [], steps: [], usage: null, pending: true, started: Date.now() };
    S.msgs.push(answer);
    const index = S.msgs.length - 1;
    S.active = index;
    setBusy(true);
    renderAll();
    scrollToEnd(true);

    const params = new URLSearchParams({ question, conversation_id: S.convId, mode: S.mode });
    if (S.scope) params.set('document_filter', S.scope);
    S.abort = new AbortController();

    try {
      const res = await fetch(`${API}/chat/stream?${params}`, { headers: { 'X-User-Id': S.user }, signal: S.abort.signal })
        .catch((e) => { if (e.name === 'AbortError') throw e; throw new ApiError(0, 'Cannot reach the server. Check that the backend is running.'); });
      if (!res.ok) throw await httpError(res);
      await readEvents(res, (ev) => onEvent(ev, answer, index));
      if (!answer.content.trim() && !answer.error) throw new ApiError(500, 'The service returned an empty answer.');
    } catch (e) {
      if (e.name === 'AbortError') {
        if (answer.content.trim()) answer.stopped = true; else answer.error = 'You stopped this request before an answer was ready.';
      } else {
        answer.error = friendly(e);
      }
    } finally {
      answer.pending = false;
      answer.content = answer.content.trim();
      answer.elapsed = (Date.now() - answer.started) / 1000;
      setBusy(false);
      refreshMessage(index);
      renderInspector();
      scrollToEnd(false);
      loadConvs();
    }
  }

  async function readEvents(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        block.split('\n').forEach((line) => {
          if (!line.startsWith('data:')) return;
          let ev; try { ev = JSON.parse(line.slice(5).trim()); } catch { return; }
          onEvent(ev);
        });
      }
    }
  }

  let raf = 0;
  function onEvent(ev, answer, index) {
    switch (ev.event) {
      case 'step':
        answer.steps.push(ev.step);
        break;
      case 'token':
        answer.content += ev.token;
        break;
      case 'done':
        answer.citations = ev.citations || [];
        answer.usage = ev.usage || null;
        answer.id = ev.message_id || null;
        answer.pending = false;
        break;
      case 'error':
        throw new ApiError(ev.status || 500, ev.detail || 'The chat service failed.');
      default: return;
    }
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; refreshMessage(index); if (S.tab === 'activity') renderActivity(); scrollToEnd(false); });
  }

  function setBusy(on) {
    S.busy = on;
    const btn = $('#sendBtn');
    btn.classList.toggle('stop', on);
    btn.innerHTML = icon(on ? 'stop' : 'send');
    btn.setAttribute('aria-label', on ? 'Stop' : 'Send question');
    btn.disabled = false;
    $('#composerHint').textContent = on ? 'Working on your question' : 'Enter to send · Shift+Enter for a new line';
    clearInterval(S.timer);
    if (on) S.timer = setInterval(() => {
      const el = $('[data-elapsed]');
      const a = S.msgs[S.msgs.length - 1];
      if (el && a) el.textContent = Math.round((Date.now() - a.started) / 1000) + 's';
    }, 1000);
  }

  function scrollToEnd(force) {
    const sc = $('#scroller');
    if (force || sc.scrollHeight - sc.scrollTop - sc.clientHeight < 180) sc.scrollTo({ top: sc.scrollHeight, behavior: 'auto' });
  }

  async function sendFeedback(index, score) {
    const m = S.msgs[index];
    if (!m || !m.id || m.feedback === score) return;
    try {
      await api('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message_id: m.id, score }) });
      m.feedback = score; refreshMessage(index);
      toast('Thanks, feedback saved.');
    } catch (e) { toast(friendly(e), 'err'); }
  }

  /* 6. Answer rendering ----------------------------------------------------- */

  // Citations look like [RWE AG.pdf, p.12] or [Report.pdf, p.3, p.5] in the answer text.
  const CITE_RE = /\[([^\[\]\n]+?),\s*((?:pp?\.?\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*)?)+)\s*\]/g;

  const chip = (doc, page, long) =>
    `<button type="button" class="cite" data-doc="${esc(enc(doc))}" data-page="${page}" aria-pressed="false" title="${esc(doc)}, page ${page}">${esc(shortDoc(doc, long ? 34 : 22))} · p.${page}</button>`;

  function renderAnswer(text) {
    const chips = [];
    let src = text.replace(CITE_RE, (_, doc, pages) => {
      const nums = [...pages.matchAll(/(\d+)(?:\s*[-–]\s*\d+)?/g)].map((m) => m[1]);
      chips.push(nums.map((n) => chip(doc.trim(), n)).join(' '));
      return `\u0001${chips.length - 1}\u0001`;
    });
    src = esc(src);

    const inline = (s) => s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)\*(?!\w)/g, '$1<em>$2</em>');

    const lines = src.split('\n');
    const out = [];
    let para = [], list = null;
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join('')}</${list.tag}>`); list = null; } };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();
      if (!trimmed) { flushPara(); flushList(); continue; }

      const head = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (head) { flushPara(); flushList(); out.push(`<h${head[1].length < 3 ? 3 : 4}>${inline(head[2])}</h${head[1].length < 3 ? 3 : 4}>`); continue; }

      if (trimmed.startsWith('|') && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1]) && lines[i + 1].includes('-')) {
        flushPara(); flushList();
        const cells = (row) => row.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
        const head = cells(trimmed);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].trim().startsWith('|')) { rows.push(cells(lines[i])); i++; }
        i--;
        const isNum = (v) => /^[\s$€£(−-]*[\d][\d.,\s]*%?\)?\s*[A-Za-z%]{0,4}$/.test(v.replace(/\u0001\d+\u0001/g, ''));
        out.push(`<table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) =>
          `<tr>${r.map((c) => `<td${isNum(c) ? ' class="num"' : ''}>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
        continue;
      }

      const ul = trimmed.match(/^[-*•]\s+(.*)$/);
      const ol = trimmed.match(/^\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        const tag = ul ? 'ul' : 'ol';
        if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
        list.items.push((ul || ol)[1]);
        continue;
      }
      flushList();
      para.push(trimmed);
    }
    flushPara(); flushList();
    return out.join('').replace(/\u0001(\d+)\u0001([.,;:]?)/g, (_, n, punct) =>
      punct ? `<span class="nb">${chips[+n]}${punct}</span>` : chips[+n]);
  }

  const NONE_RE = /(cannot answer|can't answer|not found in|no relevant|insufficient|does not contain|not available|not mentioned|no information)/i;

  function uniqueCitations(list) {
    const seen = new Set();
    return (list || []).filter((c) => {
      const doc = c.document || c.doc, key = doc + '|' + c.page;
      if (!doc || seen.has(key)) return false;
      seen.add(key); return true;
    });
  }

  function msgNode(m, i) {
    const el = document.createElement('article');
    el.dataset.i = i;
    if (m.role === 'user') {
      el.className = 'msg-user';
      el.innerHTML = `<div class="bubble">${esc(m.content)}</div>`;
      return el;
    }
    el.className = 'msg-ai';
    const sheet = document.createElement('div');
    sheet.className = 'sheet';
    const cites = uniqueCitations(m.citations);
    const live = m.pending || S.busy && i === S.msgs.length - 1;

    if (m.error) {
      sheet.classList.add('is-error');
      sheet.innerHTML = `<div class="none-note" style="color:var(--bad)">${icon('alert')}This request did not finish</div>
        <p>${esc(m.error)}</p>
        <button class="btn btn-sm retry" data-act="retry" data-i="${i}">${icon('refresh')}Try again</button>`;
      el.append(sheet);
      return el;
    }

    if (m.pending && !m.content) {
      const agent = S.mode === 'agent';
      const secs = Math.round((Date.now() - m.started) / 1000);
      sheet.innerHTML = `<div class="working" role="status">
        <div><strong>${agent ? 'Searching the reports and reading pages' : 'Retrieving passages and writing the answer'}</strong>
        <small><span data-elapsed>${secs}s</span> elapsed${agent ? ' · agent answers can take up to a minute' : ''}</small></div>
        <div class="bar" aria-hidden="true"><i></i></div></div>`;
      el.append(sheet);
      return el;
    }

    const none = !m.pending && !cites.length && NONE_RE.test(m.content);
    if (none) sheet.classList.add('is-none');
    sheet.innerHTML = `${none ? `<div class="none-note">${icon('alert')}No supporting evidence found in the reports</div>` : ''}
      <div class="answer ${live && m.pending !== false && S.busy ? 'streaming' : ''}">${renderAnswer(m.content)}</div>
      ${cites.length ? `<div class="sources"><span>Sources</span>${cites.map((c) => chip(c.document || c.doc, c.page, true)).join('')}</div>` : ''}
      ${m.stopped ? '<p class="note">Stopped before the answer finished.</p>' : ''}`;
    el.append(sheet);

    if (!(S.busy && i === S.msgs.length - 1)) {
      const u = m.usage || {};
      const tokens = (u.prompt_tokens || 0) + (u.completion_tokens || 0);
      const foot = document.createElement('div');
      foot.className = 'msg-foot';
      foot.innerHTML = `
        <button class="icon-btn" data-act="copy" data-i="${i}" aria-label="Copy answer" title="Copy answer">${icon('copy')}</button>
        ${m.id ? `<button class="icon-btn ${m.feedback === 1 ? 'on' : ''}" data-act="up" data-i="${i}" aria-label="Good answer" title="Good answer">${icon('up')}</button>
        <button class="icon-btn ${m.feedback === -1 ? 'on' : ''}" data-act="down" data-i="${i}" aria-label="Poor answer" title="Poor answer">${icon('down')}</button>` : ''}
        ${m.steps && m.steps.length ? `<button class="linkish" data-act="steps" data-i="${i}">Research steps (${m.steps.length})</button>` : ''}
        <span class="sep"></span>
        <span class="stat">${[tokens ? fmtInt(tokens) + ' tokens' : '', u.usd != null && tokens ? fmtUsd(u.usd) : '', m.elapsed ? fmtSec(m.elapsed) : ''].filter(Boolean).join(' · ')}</span>`;
      el.append(foot);
    }
    return el;
  }

  function renderHello() {
    if (S.msgs.length) return;
    const n = readyDocs().length;
    $('#thread').innerHTML = `<section class="hello">
      <h2>Ask a question. See the page it came from.</h2>
      <p>${n ? `Research Desk searches ${n} annual report${n === 1 ? '' : 's'} and cites the exact page behind every answer.` : 'Add a PDF report from the sidebar, then ask about its financials, leadership or risks.'}
      Select a citation to read the passage or open the original PDF.</p>
      <div class="hello-list">${SUGGESTIONS.map((s) => `<button type="button" data-ask="${esc(s)}"><span>${esc(s)}</span>${icon('chevron')}</button>`).join('')}</div>
    </section>`;
  }

  function renderThread() {
    const t = $('#thread');
    if (!S.msgs.length) { renderHello(); return; }
    t.replaceChildren(...S.msgs.map(msgNode));
    markPressed();
  }
  function refreshMessage(i) {
    const t = $('#thread');
    const old = t.children[i];
    if (!old) return renderThread();
    old.replaceWith(msgNode(S.msgs[i], i));
    markPressed();
  }

  function renderAll() {
    $('#threadTitle').textContent = S.title;
    document.title = S.msgs.length ? `${S.title} · Research Desk` : 'Research Desk';
    renderThread(); renderConvs(); renderScope(); renderDocs(); renderInspector();
    $$('#modeSwitch button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === S.mode)));
  }

  /* 7. Inspector ------------------------------------------------------------ */

  function setTab(name) {
    S.tab = name;
    $$('.tabs [role="tab"]').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    $$('.panel').forEach((p) => { p.hidden = p.id !== 'panel-' + name; });
    renderInspector();
  }
  function renderInspector() {
    if (S.tab === 'source') renderSource();
    else if (S.tab === 'activity') renderActivity();
    else renderUsage();
  }

  let sourceToken = 0;
  async function openSource(doc, page) {
    const keepView = S.source ? S.source.view : 'text';
    S.source = { doc, page, view: keepView, passages: null, error: null, loading: true };
    const token = ++sourceToken;
    setTab('source'); openRight();
    markPressed();
    try {
      const data = await api(`/documents/${enc(doc)}/pages/${page}`);
      if (token !== sourceToken) return;
      S.source.passages = data.passages || [];
    } catch (e) {
      if (token !== sourceToken) return;
      S.source.error = e.status === 404 ? 'No extracted text is stored for this page. Try the original page view.' : friendly(e);
    }
    S.source.loading = false;
    renderSource();
  }

  function renderSource() {
    const el = $('#panel-source');
    const s = S.source;
    if (!s) {
      el.innerHTML = `<div class="panel-empty"><strong>Check any answer against its source</strong>
        Select a citation in an answer to read the extracted passage here, or view the original PDF page.</div>`;
      return;
    }
    const doc = findDoc(s.doc);
    const pages = doc ? doc.page_count : null;
    const url = `${API}/documents/${enc(s.doc)}/file#page=${s.page}`;
    let body;
    if (s.view === 'pdf') {
      body = `<iframe class="pdf-frame" title="${esc(s.doc)}, page ${s.page}" src="${url}"></iframe>
        <p class="note">If the page does not appear, your browser may not preview PDFs inline. Use “Open PDF” instead.</p>`;
    } else if (s.loading) {
      body = '<p class="note">Loading passage…</p>';
    } else if (s.error) {
      body = `<p class="note">${esc(s.error)}</p>`;
    } else {
      body = `<div class="passages">${s.passages.map((p) => `<div class="passage">${esc(p)}</div>`).join('')}</div>`;
    }
    el.innerHTML = `<div class="src-head"><h3>${esc(s.doc)}</h3><p>Page ${s.page}${pages ? ' of ' + fmtInt(pages) : ''}</p></div>
      <div class="src-bar">
        <span class="pager">
          <button class="icon-btn" data-act="page" data-d="-1" aria-label="Previous page" ${s.page <= 1 ? 'disabled' : ''}>${icon('left')}</button>
          <span>${s.page}</span>
          <button class="icon-btn" data-act="page" data-d="1" aria-label="Next page" ${pages && s.page >= pages ? 'disabled' : ''}>${icon('chevron')}</button>
        </span>
        <span class="segmented" role="radiogroup" aria-label="View">
          <button role="radio" aria-checked="${s.view === 'text'}" data-act="view" data-v="text">Extracted text</button>
          <button role="radio" aria-checked="${s.view === 'pdf'}" data-act="view" data-v="pdf">Original page</button>
        </span>
        <span class="grow"></span>
        <a class="btn btn-sm" href="${url}" target="_blank" rel="noopener">${icon('external')}Open PDF</a>
      </div>${body}`;
  }

  const STEP_TITLES = {
    search_documents: 'Searched the reports', read_page: 'Read a page', calculate: 'Calculated',
    list_documents: 'Listed available reports', 'Final Answer': 'Wrote the answer', llm_call: 'Model call',
    force_synthesis: 'Wrote the answer from gathered evidence', offline_fallback: 'Used offline fallback',
    invalid_final_answer: 'Answer rejected, retrying'
  };
  const humanize = (a) => STEP_TITLES[a] || (String(a || 'step').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()));

  function renderActivity() {
    const el = $('#panel-activity');
    const m = S.active != null ? S.msgs[S.active] : null;
    if (!m || m.role !== 'assistant') {
      el.innerHTML = `<div class="panel-empty"><strong>Nothing researched yet</strong>Ask a question and every search, page read and calculation the agent makes will be listed here.</div>`;
      return;
    }
    const steps = m.steps || [];
    if (!steps.length) {
      el.innerHTML = m.pending
        ? `<div class="panel-empty"><strong>Working…</strong>Steps appear here as soon as the answer is ready.</div>`
        : `<div class="panel-empty"><strong>No intermediate steps</strong>Single pass mode makes one retrieval and one model call, so there is nothing to trace for this answer.</div>`;
      return;
    }
    const tools = steps.filter((s) => s.action && !['llm_call', 'Final Answer'].includes(s.action)).length;
    const models = steps.filter((s) => s.action === 'llm_call').length;
    const time = steps.reduce((t, s) => t + ((s.usage && s.usage.latency_seconds) || 0), 0);
    el.innerHTML = `<div class="sum">
        <div><b>${tools}</b><span>tool calls</span></div>
        <div><b>${models}</b><span>model calls</span></div>
        <div><b>${time ? fmtSec(time) : '–'}</b><span>model time</span></div></div>
      <div class="timeline">${steps.map(stepHtml).join('')}</div>`;
  }

  function stepHtml(s) {
    const u = s.usage || {};
    const isModel = s.action === 'llm_call';
    const isFinal = s.action === 'Final Answer';
    const meta = [u.model || (u.models && u.models[0]) || '', u.latency_seconds != null ? fmtSec(u.latency_seconds) : '',
      (u.prompt_tokens || u.completion_tokens) ? `${fmtInt(u.prompt_tokens)} in / ${fmtInt(u.completion_tokens)} out` : '',
      u.cost_usd != null ? fmtUsd(u.cost_usd) : ''].filter(Boolean).join(' · ');
    const args = s.action_args && typeof s.action_args === 'object' ? Object.entries(s.action_args).filter(([, v]) => v !== '' && v != null) : [];
    return `<div class="step ${isModel ? 'model' : ''} ${isFinal ? 'final' : ''}">
      <h4>${esc(humanize(s.action))}</h4>
      ${s.thought ? `<p class="thought">${esc(String(s.thought).slice(0, 600))}</p>` : ''}
      ${args.length ? `<div class="args">${args.map(([k, v]) => `<span class="arg"><b>${esc(k)}</b> ${esc(String(typeof v === 'object' ? JSON.stringify(v) : v).slice(0, 140))}</span>`).join('')}</div>` : ''}
      ${meta ? `<div class="meta">${esc(meta)}</div>` : ''}
      ${s.observation ? `<details><summary>Result</summary><pre>${esc(String(s.observation).slice(0, 3000))}</pre></details>` : ''}
    </div>`;
  }

  function renderUsage() {
    const el = $('#panel-usage');
    const rows = [];
    S.msgs.forEach((m, i) => { if (m.role === 'assistant' && m.usage && !m.error) rows.push({ q: (S.msgs[i - 1] || {}).content || '', u: m.usage }); });
    if (!rows.length) {
      el.innerHTML = `<div class="panel-empty"><strong>No usage yet</strong>Token counts and estimated cost for this conversation appear after the first answer.</div>`;
      return;
    }
    const sum = (k) => rows.reduce((t, r) => t + (r.u[k] || 0), 0);
    const models = [...new Set(rows.flatMap((r) => r.u.models || (r.u.model ? [r.u.model] : [])))];
    el.innerHTML = `<div class="cards">
        <div class="stat-card"><b>${fmtUsd(sum('usd'))}</b><span>estimated cost</span></div>
        <div class="stat-card"><b>${fmtInt(rows.length)}</b><span>answer${rows.length === 1 ? '' : 's'}</span></div>
        <div class="stat-card"><b>${fmtInt(sum('prompt_tokens'))}</b><span>tokens in</span></div>
        <div class="stat-card"><b>${fmtInt(sum('completion_tokens'))}</b><span>tokens out</span></div></div>
      ${models.length ? `<p class="note">Model${models.length === 1 ? '' : 's'}: ${esc(models.join(', '))}</p>` : ''}
      <table class="usage-table"><caption>By question</caption><thead><tr><th>Question</th><th>Tokens</th><th>Cost</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td title="${esc(r.q)}">${esc(r.q)}</td><td>${fmtInt((r.u.prompt_tokens || 0) + (r.u.completion_tokens || 0))}</td><td>${fmtUsd(r.u.usd)}</td></tr>`).join('')}</tbody></table>`;
  }

  // Keeps the highlighter state on citation chips in sync with the open source.
  function markPressed() {
    $$('.cite').forEach((c) => {
      const on = S.source && decodeURIComponent(c.dataset.doc) === S.source.doc && Number(c.dataset.page) === S.source.page;
      c.setAttribute('aria-pressed', String(!!on));
    });
  }

  /* 8. Wiring --------------------------------------------------------------- */

  const app = $('#app');
  function openLeft() { $('#railLeft').classList.add('is-shown'); $('#scrim').hidden = false; }
  function openRight() { if (window.matchMedia('(max-width: 1220px)').matches) { $('#inspector').classList.add('is-shown'); $('#scrim').hidden = false; } }
  function closeDrawers() { $('#railLeft').classList.remove('is-shown'); $('#inspector').classList.remove('is-shown'); $('#scrim').hidden = true; }

  function autosize() {
    const t = $('#composerInput');
    t.style.height = 'auto';
    t.style.height = Math.min(t.scrollHeight, 180) + 'px';
  }

  function init() {
    // Static icons
    $('#newChat').innerHTML = icon('plus') + 'New conversation';
    $('#dzIcon').innerHTML = icon('upload');
    $('#sendBtn').innerHTML = icon('send');
    $('#reindexBtn').innerHTML = icon('refresh') + 'Reindex';
    $('#openLeft').innerHTML = icon('menu');
    $('#openRight').innerHTML = icon('panel');
    $('#closeLeft').innerHTML = icon('x');
    $('#closeRight').innerHTML = icon('x');

    // Header, composer
    $('#modeSwitch').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-mode]');
      if (!b) return;
      S.mode = b.dataset.mode; save('rd_mode', S.mode);
      $$('#modeSwitch button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    });
    $('#scopeLine').addEventListener('click', (e) => { if (e.target.id === 'clearScope') setScope(null); });
    $('#scopeSelect').addEventListener('change', (e) => setScope(e.target.value));

    const input = $('#composerInput');
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#composer').requestSubmit(); }
    });
    $('#composer').addEventListener('submit', (e) => {
      e.preventDefault();
      if (S.busy) { if (S.abort) S.abort.abort(); return; }
      const q = input.value;
      if (!q.trim()) return;
      input.value = ''; autosize();
      ask(q);
    });

    // Thread: suggestions, citations, message actions
    $('#thread').addEventListener('click', (e) => {
      const sug = e.target.closest('[data-ask]');
      if (sug) { ask(sug.dataset.ask); return; }
      const c = e.target.closest('.cite');
      if (c) { openSource(decodeURIComponent(c.dataset.doc), Number(c.dataset.page)); return; }
      const b = e.target.closest('[data-act]');
      if (!b) return;
      const i = Number(b.dataset.i);
      const m = S.msgs[i];
      if (b.dataset.act === 'copy') {
        (navigator.clipboard ? navigator.clipboard.writeText(m.content) : Promise.reject()).then(() => toast('Answer copied.'), () => toast('Copy is blocked by the browser.', 'err'));
      } else if (b.dataset.act === 'up') sendFeedback(i, 1);
      else if (b.dataset.act === 'down') sendFeedback(i, -1);
      else if (b.dataset.act === 'steps') { S.active = i; setTab('activity'); openRight(); }
      else if (b.dataset.act === 'retry') {
        const q = (S.msgs[i - 1] || {}).content;
        S.msgs.splice(i - 1, 2);
        renderThread();
        if (q) ask(q);
      }
    });

    // Library
    $('#docList').addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      const li = e.target.closest('.doc[data-id]');
      if (!b || !li) return;
      const id = decodeURIComponent(li.dataset.id);
      if (b.dataset.act === 'scope') setScope(S.scope === id ? null : id);
      if (b.dataset.act === 'delete') deleteDoc(id);
    });
    $('#docFilter').addEventListener('input', (e) => { S.filter = e.target.value; renderDocs(); });
    $('#convList').addEventListener('click', (e) => { const b = e.target.closest('.conv'); if (b) openConv(b.dataset.id); });
    $('#newChat').addEventListener('click', newConversation);
    $('#reindexBtn').addEventListener('click', reindex);

    const dz = $('#dropzone');
    $('#fileInput').addEventListener('change', (e) => { [...e.target.files].forEach(uploadFile); e.target.value = ''; });
    ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
    dz.addEventListener('drop', (e) => [...e.dataTransfer.files].forEach(uploadFile));
    // Dropping a PDF anywhere else on the page should not navigate away from the app.
    ['dragover', 'drop'].forEach((t) => window.addEventListener(t, (e) => { if (!dz.contains(e.target)) e.preventDefault(); }));

    // Inspector
    $('.tabs').addEventListener('click', (e) => { const t = e.target.closest('[role="tab"]'); if (t) setTab(t.dataset.tab); });
    $('#panel-source').addEventListener('click', (e) => {
      const b = e.target.closest('[data-act]');
      if (!b || !S.source) return;
      if (b.dataset.act === 'page') openSource(S.source.doc, Math.max(1, S.source.page + Number(b.dataset.d)));
      if (b.dataset.act === 'view') { S.source.view = b.dataset.v; renderSource(); }
    });

    // Drawers (small screens)
    $('#openLeft').addEventListener('click', openLeft);
    $('#closeLeft').addEventListener('click', closeDrawers);
    $('#openRight').addEventListener('click', () => { $('#inspector').classList.add('is-shown'); $('#scrim').hidden = false; });
    $('#closeRight').addEventListener('click', closeDrawers);
    $('#scrim').addEventListener('click', closeDrawers);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawers(); });

    renderAll();
    renderHealth();
    loadDocs(); loadConvs(); pollHealth();
    setTab('source');
    input.focus();
  }

  init();
})();
