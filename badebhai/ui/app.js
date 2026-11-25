// BadeBhai web UI: streams /ask/stream (SSE over fetch), renders cited answers, trace and memories.
// All repository-derived text is untrusted: it is HTML-escaped before any formatting is applied.
"use strict";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

const state = { citations: [], busy: false, traceEl: null, repos: [], setup: {}, watching: null };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function safeUrl(u) { return /^https?:\/\//i.test(u || "") ? u : null; }

// ---------------------------------------------------------------- markdown (minimal, escape-first)
function inline(text, citeIds) {
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[(\d{1,3})\]/g, (m, n) => (citeIds.has(Number(n)) ? `<button type="button" class="cite" data-n="${n}">${n}</button>` : m));
  s = s.replace(/⚠️\((unverified|reference not found)\)/g, '<span class="flag" title="Removed or unverifiable citation">⚠ $1</span>');
  return s;
}
function renderMarkdown(md, citations) {
  const ids = new Set(citations.map((c) => c.id));
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let list = null, para = [], code = null;
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "), ids)}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i, ids)}</li>`).join("")}</${list.tag}>`); list = null; } };
  for (const line of lines) {
    if (code !== null) {
      if (/^\s*(```|~~~)/.test(line)) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = null; }
      else code.push(line);
      continue;
    }
    if (/^\s*(```|~~~)/.test(line)) { flushPara(); flushList(); code = []; continue; }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (bullet || ordered) {
      flushPara();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((bullet || ordered)[1]);
    } else if (heading) {
      flushPara(); flushList(); out.push(`<p><strong>${inline(heading[1], ids)}</strong></p>`);
    } else if (!line.trim()) {
      flushPara(); flushList();
    } else {
      if (list) flushList();
      para.push(line.trim());
    }
  }
  if (code !== null) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  flushPara(); flushList();
  return out.join("");
}

// ---------------------------------------------------------------- setup
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) { const b = await r.json().catch(() => ({})); throw new Error(b.error || b.detail || r.statusText); }
  return r.json();
}
function threadKey() { return `badebhai.thread.${$("#repo").value}.${$("#user").value}`; }

async function init() {
  const user = store.get("badebhai.user", null) || `user-${Math.random().toString(36).slice(2, 7)}`;
  $("#user").value = user;
  store.set("badebhai.user", user);
  $("#mode").value = store.get("badebhai.mode", "agent");
  try { state.setup = await api("/setup"); } catch { state.setup = {}; }
  await loadRepos(store.get("badebhai.repo", null));
  loadMemories();
  initAddRepo();
  try {  // keep showing progress for repos still being added (e.g. after a page reload)
    const running = (await api("/jobs?status=running")).filter((j) => j.repo);
    if (running.length) watchJob(running[0].id, running[0].repo, running[0].kind, false);
  } catch { /* older server */ }

  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  document.querySelectorAll(".examples button").forEach((b) => b.addEventListener("click", () => { $("#question").value = b.textContent; ask(); }));
  $("#repo").addEventListener("change", () => { store.set("badebhai.repo", $("#repo").value); updateIntro(); loadMemories(); });
  $("#user").addEventListener("change", () => { store.set("badebhai.user", $("#user").value.trim() || user); loadMemories(); });
  $("#mode").addEventListener("change", () => store.set("badebhai.mode", $("#mode").value));
  $("#new-thread").addEventListener("click", () => { store.del(threadKey()); $("#messages").innerHTML = ""; setStatus("Started a new thread."); });
  $("#clear-memory").addEventListener("click", clearMemories);
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#messages").addEventListener("click", (e) => { const b = e.target.closest(".cite"); if (b) focusCitation(Number(b.dataset.n)); });
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
}
function setStatus(text, isError = false) { const s = $("#status"); s.textContent = text; s.classList.toggle("error", isError); }

// ---------------------------------------------------------------- asking (SSE over fetch)
async function ask() {
  const question = $("#question").value.trim();
  if (!question || state.busy) return;
  const repo = $("#repo").value;
  if (!repo) { setStatus("Add a repository first.", true); openAdd(); return; }
  state.busy = true; $("#send").disabled = true; $("#question").value = "";
  $("#empty")?.remove();
  const msgs = $("#messages");
  msgs.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${esc(question)}</div></div>`);
  const holder = document.createElement("div");
  holder.className = "msg assistant";
  holder.innerHTML = `<div class="bubble streaming">…</div>`;
  msgs.appendChild(holder);
  msgs.scrollTop = msgs.scrollHeight;
  const bubble = holder.querySelector(".bubble");
  resetTrace();
  setStatus("Thinking…");

  const body = { question, repo, mode: $("#mode").value, user_id: $("#user").value.trim() || "anonymous",
                 thread_id: store.get(threadKey(), null) };
  let streamed = "";
  try {
    const resp = await fetch("/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || resp.statusText);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx).replace(/^\r?\n\r?\n/, "");
        const ev = parseSse(block);
        if (!ev) continue;
        if (ev.event === "start") store.set(threadKey(), ev.data.thread_id);
        else if (ev.event === "step") addStep(ev.data);
        else if (ev.event === "tool") addTool(ev.data);
        else if (ev.event === "token") { streamed += ev.data.text; bubble.textContent = streamed; msgs.scrollTop = msgs.scrollHeight; }
        else if (ev.event === "final") renderFinal(holder, ev.data.response);
        else if (ev.event === "error") throw new Error(ev.data.error);
      }
    }
  } catch (e) {
    bubble.classList.remove("streaming");
    bubble.innerHTML = `<div class="notice error">${esc(e.message || String(e))}</div>`;
    setStatus("Request failed.", true);
  } finally {
    state.busy = false; $("#send").disabled = false; $("#question").focus();
  }
}

function parseSse(block) {
  let event = "message"; const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join("\n")) }; } catch { return null; }
}

function renderFinal(holder, r) {
  state.citations = r.citations || [];
  const bubble = holder.querySelector(".bubble");
  bubble.classList.remove("streaming");
  bubble.innerHTML = renderMarkdown(r.answer || "", state.citations);
  const v = r.verification || {};
  const badges = [];
  if (v.total_citations) {
    const ok = v.valid_citations === v.total_citations;
    badges.push(`<span class="badge ${ok ? "ok" : "warn"}">${v.valid_citations}/${v.total_citations} citations verified</span>`);
  }
  if (r.insufficient?.length) badges.push(`<span class="badge warn">insufficient evidence: ${r.insufficient.length} part(s)</span>`);
  if (r.memories_used?.length) badges.push(`<span class="badge">${r.memories_used.length} memory fact(s) used</span>`);
  const saved = (r.memories_written || []).filter((m) => ["saved", "superseded"].includes(m.verdict));
  if (saved.length) badges.push(`<span class="badge ok">remembered: ${esc(saved.map((m) => m.text).join("; "))}</span>`);
  badges.push(`<span class="badge">${r.mode}</span>`);
  badges.push(`<span class="badge">${(r.latency_ms / 1000).toFixed(1)} s</span>`);
  if (r.usage?.llm_calls) {
    const cost = !r.usage.cost_complete ? " · cost n/a"
      : r.usage.cost_usd === 0 ? " · free (local model)" : ` · $${r.usage.cost_usd.toFixed(4)}`;
    badges.push(`<span class="badge">${r.usage.llm_calls} LLM calls${cost}</span>`);
  }
  if (r.tool_calls?.length) badges.push(`<span class="badge">${r.tool_calls.length} MCP tool calls</span>`);
  let extra = `<div class="meta">${badges.join("")}</div>`;
  for (const w of r.warnings || []) extra += `<div class="notice">${esc(w)}</div>`;
  if (v.invalid_citations?.length) extra += `<div class="notice">Verifier removed ${v.invalid_citations.length} citation(s): ${esc(v.invalid_citations.map((x) => x.problem).join("; "))}</div>`;
  holder.insertAdjacentHTML("beforeend", extra);
  renderCitations();
  renderTrace(r.trace || [], r.tool_calls || []);
  setStatus(r.thread_id ? `Thread ${r.thread_id}` : "");
  loadMemories();
}

// ---------------------------------------------------------------- citations
function renderCitations() {
  const panel = $("#panel-citations");
  $("#cite-count").textContent = state.citations.length ? `(${state.citations.length})` : "";
  if (!state.citations.length) { panel.innerHTML = `<p class="hint">No citations in this answer.</p>`; return; }
  panel.innerHTML = state.citations.map((c) => {
    const url = safeUrl(c.url);
    const label = url ? `<a class="label" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(c.label)}</a>` : `<span class="label">${esc(c.label)}</span>`;
    return `<div class="citation" id="cite-${c.id}">
      <div class="row"><span class="num">[${c.id}]</span><span class="type">${esc(c.type)}</span>${label}</div>
      <div class="title">${esc(c.title)}</div>
      ${c.snippet ? `<details><summary>show source</summary><pre>${esc(c.snippet)}</pre></details>` : ""}
    </div>`;
  }).join("");
}
function focusCitation(n) {
  showTab("citations");
  document.querySelectorAll(".citation").forEach((el) => el.classList.remove("hl"));
  const el = document.getElementById(`cite-${n}`);
  if (el) { el.classList.add("hl"); el.scrollIntoView({ behavior: "smooth", block: "center" }); el.querySelector("details")?.setAttribute("open", ""); }
}

// ---------------------------------------------------------------- trace
function resetTrace() { $("#panel-trace").innerHTML = ""; $("#trace-count").textContent = ""; state.traceEl = null; showTab("trace"); }
function addStep(s) {
  const el = document.createElement("div");
  el.className = "step";
  el.innerHTML = `<div><span class="node">${esc(s.node)}</span><span class="ms">${s.ms ?? ""} ms</span></div><div class="summary">${esc(s.summary)}</div>`;
  $("#panel-trace").appendChild(el);
  $("#trace-count").textContent = `(${$("#panel-trace").querySelectorAll(".step").length})`;
}
function addTool(t) {
  const el = document.createElement("div");
  el.className = "toolcall";
  el.textContent = `→ ${t.sub_question}: ${t.tool}(${JSON.stringify(t.args)})`;
  $("#panel-trace").appendChild(el);
}
function renderTrace(trace, toolCalls) {
  const panel = $("#panel-trace");
  panel.innerHTML = "";
  for (const s of trace) {
    const detail = s.detail && Object.keys(s.detail).length ? JSON.stringify(s.detail, null, 2).slice(0, 6000) : "";
    const el = document.createElement("div");
    el.className = "step";
    el.innerHTML = `<div><span class="node">${esc(s.node)}</span><span class="ms">${s.ms ?? ""} ms</span></div>
      <div class="summary">${esc(s.summary)}</div>
      ${detail ? `<details><summary>details</summary><div class="detail">${esc(detail)}</div></details>` : ""}`;
    panel.appendChild(el);
  }
  if (toolCalls.length) {
    const el = document.createElement("details");
    el.innerHTML = `<summary>${toolCalls.length} MCP tool call(s)</summary>` + toolCalls.map((c) =>
      `<div class="toolcall">${esc(c.server)}.${esc(c.tool)}(${esc(JSON.stringify(c.args))}) · ${c.ok ? `${c.n_items} items` : `error: ${esc(c.error)}`} · ${c.ms} ms</div>`).join("");
    panel.appendChild(el);
  }
  $("#trace-count").textContent = `(${trace.length})`;
}

// ---------------------------------------------------------------- memories
async function loadMemories() {
  const user = $("#user").value.trim(), repo = $("#repo").value;
  const list = $("#memories");
  if (!user || !repo) { list.innerHTML = ""; return; }
  try {
    const mems = await api(`/memories?user_id=${encodeURIComponent(user)}&repo=${encodeURIComponent(repo)}`);
    $("#mem-count").textContent = mems.length ? `(${mems.length})` : "";
    list.innerHTML = mems.length ? mems.map((m) => `<li><div><div class="cat">${esc(m.category)} · importance ${m.importance.toFixed(2)}</div>${esc(m.text)}</div>
      <button type="button" class="link danger" data-id="${m.id}">Forget</button></li>`).join("")
      : `<li class="hint">Nothing yet. Tell BadeBhai about your role or preferences ("I work on the payments module") and it will remember.</li>`;
    list.querySelectorAll("button[data-id]").forEach((b) => b.addEventListener("click", async () => {
      await fetch(`/memories/${b.dataset.id}?user_id=${encodeURIComponent(user)}`, { method: "DELETE" });
      loadMemories();
    }));
  } catch (e) { list.innerHTML = `<li class="hint">Could not load memories: ${esc(e.message)}</li>`; }
}
async function clearMemories() {
  const user = $("#user").value.trim(), repo = $("#repo").value;
  if (!user || !confirm(`Forget everything BadeBhai remembers about ${user} for ${repo}?`)) return;
  await fetch(`/memories?user_id=${encodeURIComponent(user)}&repo=${encodeURIComponent(repo)}`, { method: "DELETE" });
  loadMemories();
}

// ---------------------------------------------------------------- repositories
async function loadRepos(select) {
  const sel = $("#repo");
  try {
    state.repos = await api("/repos");
  } catch (e) { setStatus(`Could not load repositories: ${e.message}`, true); return; }
  const prev = select || sel.value;
  sel.innerHTML = state.repos.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}${r.ref ? " @ " + esc(r.ref) : ""}${r.last_indexed_commit ? "" : " (still being added…)"}</option>`).join("");
  if (prev && state.repos.some((r) => r.name === prev)) sel.value = prev;
  if (sel.value) store.set("badebhai.repo", sel.value);
  updateIntro();
  renderRepoList();
}
function updateIntro() {
  const has = state.repos.length > 0;
  const welcome = $("#welcome"), intro = $("#ask-intro");
  if (welcome) welcome.hidden = has;
  if (intro) intro.hidden = !has;
  const name = $("#repo-name-intro");
  if (name) name.textContent = $("#repo").value || "the repository";
}
function timeAgo(iso) {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 90) return "just now";
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}
function duration(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), r = sec % 60;
  return m < 60 ? `${m}m ${String(r).padStart(2, "0")}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}
function renderRepoList() {
  const wrap = $("#repo-list-wrap"), list = $("#repo-list");
  if (!wrap) return;
  wrap.hidden = !state.repos.length;
  list.innerHTML = state.repos.map((r) => {
    const status = r.last_indexed_commit ? `updated ${timeAgo(r.last_indexed_at)}` : "being added…";
    const canUpdate = r.last_indexed_commit && r.kind !== "local" && r.name.split("/").length === 2;
    const link = r.github ? `<a href="https://github.com/${esc(r.github)}" target="_blank" rel="noopener noreferrer">${esc(r.name)}</a>` : esc(r.name);
    return `<li><div><div class="name">${link}${r.ref ? ` <span class="muted">@ ${esc(r.ref)}</span>` : ""}</div><div class="muted small">${status}</div></div>
      <div class="actions">
        ${r.last_indexed_commit ? `<button type="button" class="link" data-ask="${esc(r.name)}">Ask</button>` : ""}
        ${canUpdate ? `<button type="button" class="link" data-update="${esc(r.name)}" title="Download new commits and re-read changed files">Check for updates</button>` : ""}
      </div></li>`;
  }).join("");
}

// ---------------------------------------------------------------- add a repository
function initAddRepo() {
  const dlg = $("#add-dialog");
  $("#add-repo").addEventListener("click", openAdd);
  document.querySelectorAll("[data-open-add]").forEach((b) => b.addEventListener("click", openAdd));
  $("#add-close").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });  // click outside
  $("#add-form").addEventListener("submit", (e) => { e.preventDefault(); addRepo(); });
  document.querySelectorAll(".example-src").forEach((b) => b.addEventListener("click", () => { $("#add-source").value = b.textContent.trim(); $("#add-source").focus(); }));
  $("#repo-list").addEventListener("click", (e) => {
    const ask = e.target.closest("[data-ask]"), upd = e.target.closest("[data-update]");
    if (ask) useRepo(ask.dataset.ask);
    if (upd) updateRepo(upd.dataset.update);
  });
  $("#add-progress").addEventListener("click", (e) => {
    const ask = e.target.closest("[data-ask]"), upd = e.target.closest("[data-update]");
    if (ask) useRepo(ask.dataset.ask);
    if (upd) updateRepo(upd.dataset.update);
  });
  renderTokenBox(false);
}
function openAdd() {
  const dlg = $("#add-dialog");
  renderRepoList();
  if (!dlg.open) dlg.showModal();
  $("#add-source").focus();
}
function useRepo(name) {
  $("#add-dialog").close();
  $("#repo").value = name;
  store.set("badebhai.repo", name);
  updateIntro();
  loadMemories();
  $("#question").focus();
  setStatus(`Ready: ask anything about ${name}.`);
}

function renderTokenBox(highlight) {
  const box = $("#token-box");
  const st = state.setup || {};
  if (st.github_token) {
    box.innerHTML = `<div class="token ok">✓ A GitHub token is set, so private repositories it can access work too.</div>`;
    return;
  }
  const url = safeUrl(st.token_url) || "https://github.com/settings/personal-access-tokens/new";
  box.innerHTML = `<details class="token${highlight ? " attention" : ""}"${highlight ? " open" : ""}>
    <summary>${highlight ? "This needs a free GitHub token. Here's how (about 2 minutes)" : "Private repository? You'll need a free GitHub token"}</summary>
    <p>Public repositories work without one. A token lets BadeBhai read <strong>private</strong> repositories, and raises
      GitHub's limit for searching issues and pull requests from 60 to 5,000 requests an hour.</p>
    <ol>
      <li>Open <a href="${esc(url)}" target="_blank" rel="noopener noreferrer">GitHub → Settings → Fine-grained tokens → Generate new token</a> (sign in if asked).</li>
      <li>Name it <strong>BadeBhai</strong> and pick an expiry date.</li>
      <li>Under <strong>Repository access</strong> choose <strong>Public repositories</strong>, or <strong>Only select repositories</strong> and pick your private ones.</li>
      <li>If you picked private repositories: under <strong>Permissions</strong>, set <strong>Contents</strong>, <strong>Issues</strong> and <strong>Pull requests</strong> to <strong>Read-only</strong>.</li>
      <li>Click <strong>Generate token</strong> and copy it. It starts with <code>github_pat_</code>.</li>
      <li>In the BadeBhai folder, open the <code>.env</code> file and add the line below. (No <code>.env</code> file? Copy <code>.env.example</code> and name the copy <code>.env</code>.)
        <div class="copyline"><code>GITHUB_TOKEN=github_pat_your_token_here</code></div></li>
      <li>Restart BadeBhai: run the same <code>docker compose … up -d</code> command you started it with
        (or stop <code>badebhai serve</code> and start it again), then reload this page.</li>
    </ol>
    <p class="muted small">The token stays on your computer. BadeBhai only uses it to read from GitHub: it never writes
      anything, never stores the token in its database, and never sends it to this page.</p>
  </details>`;
}

async function addRepo() {
  const source = $("#add-source").value.trim();
  const out = $("#add-progress");
  if (!source) { $("#add-source").focus(); return; }
  renderTokenBox(false);
  out.hidden = false;
  out.innerHTML = `<div class="job"><div class="job-title">Checking <strong>${esc(source)}</strong>…</div></div>`;
  $("#add-submit").disabled = true;
  try {
    const r = await fetch("/repos", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source }) });
    const b = await r.json().catch(() => ({}));
    if (r.status === 409 && b.already_indexed) {
      out.innerHTML = `<div class="job"><div class="job-title">✓ ${esc(b.error)}</div>
        <div class="job-actions"><button type="button" class="primary" data-ask="${esc(b.repo)}">Ask about it</button>
        ${b.can_update ? `<button type="button" data-update="${esc(b.repo)}">Check for updates</button>` : ""}</div></div>`;
      return;
    }
    if (!r.ok) { showAddError(b.error || b.detail || r.statusText, b.needs_token, b.details); return; }
    $("#add-source").value = "";
    await loadRepos($("#repo").value || b.repo);
    watchJob(b.job_id, b.repo, "ingest", true, b.about, b.warnings);
  } catch (e) {
    showAddError(`Couldn't reach BadeBhai: ${e.message}`);
  } finally {
    $("#add-submit").disabled = false;
  }
}
async function updateRepo(name) {
  const out = $("#add-progress");
  out.hidden = false;
  if (!$("#add-dialog").open) $("#add-dialog").showModal();
  try {
    const [owner, repo] = name.split("/");
    const b = await api(`/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/sync`, { method: "POST" });
    watchJob(b.job_id, name, "sync", true);
  } catch (e) { showAddError(e.message); }
}
function showAddError(message, needsToken = false, details = "") {
  const out = $("#add-progress");
  out.hidden = false;
  out.innerHTML = `<div class="job failed"><div class="job-title">${esc(message)}</div>
    ${details ? `<details><summary>technical details</summary><pre>${esc(details)}</pre></details>` : ""}</div>`;
  if (needsToken) renderTokenBox(true);
}

const STAGES = {
  ingest: [["downloading", "Downloading the code"], ["indexing", "Reading files"], ["done", "Ready to answer"]],
  sync: [["downloading", "Fetching new commits"], ["indexing", "Re-reading changed files"], ["done", "Up to date"]],
};
function watchJob(jobId, repo, kind, focus, about = null, warnings = []) {
  const token = Symbol(jobId);
  state.watching = token;
  const samples = [];
  const tick = async () => {
    if (state.watching !== token) return;
    let job;
    try { job = await api(`/jobs/${jobId}`); } catch (e) { setTimeout(tick, 3000); return; }
    about = about || job.about; warnings = warnings?.length ? warnings : job.warnings || [];
    renderJob(job, repo, kind, about, warnings, samples);
    if (job.status === "running") { setTimeout(tick, 1000); return; }
    state.watching = null;
    await loadRepos(job.status === "done" && kind === "ingest" && !$("#repo").value ? repo : null);
    if (job.status === "done") {
      setStatus(kind === "sync" ? `${repo} is up to date.` : `${repo} is ready. Pick it in the Repo menu and ask away.`);
    } else {
      setStatus(`Adding ${repo} failed: ${job.error || "unknown error"}`, true);
      if (job.needs_token) renderTokenBox(true);
    }
  };
  if (focus) { $("#add-progress").hidden = false; }
  tick();
}
function renderJob(job, repo, kind, about, warnings, samples) {
  const out = $("#add-progress");
  const p = job.progress || {};
  const stages = STAGES[kind] || STAGES.ingest;
  const elapsed = (job.finished_at || Date.now() / 1000) - (job.started_at || Date.now() / 1000);
  let pct = 0, line = "";
  if (job.status === "running") {
    if (p.stage === "indexing" && p.total) {
      samples.push([Date.now() / 1000, p.done]);
      while (samples.length > 30) samples.shift();
      pct = Math.round((100 * p.done) / p.total);
      line = `${p.done.toLocaleString()} of ${p.total.toLocaleString()} files · ${duration(elapsed)}`;
      const [t0, d0] = samples[0], [t1, d1] = samples[samples.length - 1];
      if (t1 - t0 > 10 && d1 > d0) line += ` · about ${duration(((p.total - d1) * (t1 - t0)) / (d1 - d0))} left`;
    } else {
      const big = kind === "ingest" && !(about?.size_mb < 50);
      line = `${duration(elapsed)}${big ? " · big repositories take a few minutes to download" : ""}`;
    }
    setStatus(`${kind === "sync" ? "Updating" : "Adding"} ${repo}… ${p.stage === "indexing" && p.total ? pct + "%" : ""}`);
  }
  const current = job.status === "done" ? "done" : p.stage === "indexing" ? "indexing" : "downloading";
  const idx = stages.findIndex(([k]) => k === current);
  const steps = stages.map(([k, label], i) => {
    const cls = job.status === "failed" && i === idx ? "failed" : i < idx || job.status === "done" ? "done" : i === idx ? "active" : "";
    const extra = k === "indexing" && cls === "active" && p.total ? ` (${pct}%)` : "";
    return `<li class="${cls}">${esc(label)}${extra}</li>`;
  }).join("");
  const desc = about?.description ? `<div class="muted small">${esc(about.description)}${about.size_mb ? ` · ~${about.size_mb.toLocaleString()} MB` : ""}</div>` : "";
  const warn = (warnings || []).map((w) => `<div class="notice">${esc(w)}</div>`).join("");
  let body = "";
  if (job.status === "running") {
    body = `<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${current === "indexing" ? Math.max(2, pct) : 0}%"${current === "indexing" ? "" : ' class="indeterminate"'}></span></div>
      <div class="muted small">${line}</div>
      <div class="muted small">You can close this window and keep asking about other repositories meanwhile.</div>`;
  } else if (job.status === "done") {
    const r = job.report || {};
    const what = kind === "sync"
      ? (r.files_indexed || r.files_removed ? `${r.files_indexed} changed file(s) re-read, ${r.files_removed} removed` : "no changes since last time")
      : `${(r.files_indexed || 0).toLocaleString()} files read, ${(r.chunks_written || 0).toLocaleString()} searchable pieces`;
    body = `<div class="ok-line">✓ ${esc(repo)} is ${kind === "sync" ? "up to date" : "ready"}: ${what}, in ${duration(elapsed)}.</div>
      <div class="job-actions"><button type="button" class="primary" data-ask="${esc(repo)}">Start asking</button></div>`;
  } else {
    body = `<div class="err-line">${esc(job.error || "Something went wrong.")}</div>
      ${job.details && job.details !== job.error ? `<details><summary>technical details</summary><pre>${esc(job.details)}</pre></details>` : ""}`;
  }
  out.innerHTML = `<div class="job ${job.status}">
    <div class="job-title">${kind === "sync" ? "Updating" : "Adding"} <strong>${esc(repo)}</strong></div>${desc}
    <ol class="stages">${steps}</ol>${body}${warn}</div>`;
}

init();
