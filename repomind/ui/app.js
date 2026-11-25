// RepoMind web UI: streams /ask/stream (SSE over fetch), renders cited answers, trace and memories.
// All repository-derived text is untrusted: it is HTML-escaped before any formatting is applied.
"use strict";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

const state = { citations: [], busy: false, traceEl: null };

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
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || (await r.text().catch(() => "")) || r.statusText);
  return r.json();
}
function threadKey() { return `repomind.thread.${$("#repo").value}.${$("#user").value}`; }

async function init() {
  const user = store.get("repomind.user", null) || `user-${Math.random().toString(36).slice(2, 7)}`;
  $("#user").value = user;
  store.set("repomind.user", user);
  $("#mode").value = store.get("repomind.mode", "agent");
  try {
    const repos = await api("/repos");
    const sel = $("#repo");
    sel.innerHTML = repos.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}${r.ref ? " @ " + esc(r.ref) : ""}</option>`).join("");
    const saved = store.get("repomind.repo", null);
    if (saved && repos.some((r) => r.name === saved)) sel.value = saved;
    if (!repos.length) setStatus("No repository indexed yet. Ingestion may still be running; refresh in a minute.");
    const pending = repos.filter((r) => !r.last_indexed_commit).map((r) => r.name);
    if (pending.length) setStatus(`Still indexing: ${pending.join(", ")}`);
  } catch (e) { setStatus(`Could not load repositories: ${e.message}`, true); }
  loadMemories();

  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  document.querySelectorAll(".examples button").forEach((b) => b.addEventListener("click", () => { $("#question").value = b.textContent; ask(); }));
  $("#repo").addEventListener("change", () => { store.set("repomind.repo", $("#repo").value); loadMemories(); });
  $("#user").addEventListener("change", () => { store.set("repomind.user", $("#user").value.trim() || user); loadMemories(); });
  $("#mode").addEventListener("change", () => store.set("repomind.mode", $("#mode").value));
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
  if (!repo) { setStatus("Pick a repository first.", true); return; }
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
      : `<li class="hint">Nothing yet. Tell RepoMind about your role or preferences ("I work on the payments module") and it will remember.</li>`;
    list.querySelectorAll("button[data-id]").forEach((b) => b.addEventListener("click", async () => {
      await fetch(`/memories/${b.dataset.id}?user_id=${encodeURIComponent(user)}`, { method: "DELETE" });
      loadMemories();
    }));
  } catch (e) { list.innerHTML = `<li class="hint">Could not load memories: ${esc(e.message)}</li>`; }
}
async function clearMemories() {
  const user = $("#user").value.trim(), repo = $("#repo").value;
  if (!user || !confirm(`Forget everything RepoMind remembers about ${user} for ${repo}?`)) return;
  await fetch(`/memories?user_id=${encodeURIComponent(user)}&repo=${encodeURIComponent(repo)}`, { method: "DELETE" });
  loadMemories();
}

init();
