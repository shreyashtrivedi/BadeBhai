// BadeBhai recorded tour: the real web UI, served as static files, answering from answers recorded
// against a running BadeBhai (scripts/record_tour.py). Loaded before app.js; it replaces fetch() for
// BadeBhai's API paths and replays the recorded trace and answer as a stream, like the live app.
"use strict";

(() => {
  const TOUR = window.BADEBHAI_TOUR;  // recorded.js
  const realFetch = window.fetch.bind(window);
  const norm = (q) => String(q || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const json = (obj, status = 200) => Promise.resolve(new Response(JSON.stringify(obj), {
    status, headers: { "Content-Type": "application/json" } }));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const API = /^\/(auth\/me|setup|repos|memories|jobs|overview|file|feedback|ask|compare)(\/|$)/;

  function findAnswer(question, repo) {
    const q = norm(question);
    return TOUR.answers.find((a) => norm(a.question) === q && (!repo || a.repo === repo))
      || TOUR.answers.find((a) => norm(a.question) === q);
  }

  function sseStream(events) {
    const enc = new TextEncoder();
    return new ReadableStream({
      async start(controller) {
        for (const [delay, ev] of events) {
          if (delay) await sleep(delay);
          controller.enqueue(enc.encode(`event: ${ev.type}\ndata: ${JSON.stringify(ev)}\n\n`));
        }
        controller.close();
      },
    });
  }

  function replay(answer) {
    const events = [];
    for (const step of answer.response.trace || []) events.push([350, { type: "step", node: step.node, summary: step.summary, ms: step.ms }]);
    const words = (answer.response.answer || "").split(/(\s+)/);
    for (let i = 0; i < words.length; i += 6) events.push([25, { type: "token", text: words.slice(i, i + 6).join("") }]);
    events.push([150, { type: "final", response: { ...answer.response, latency_ms: answer.response.latency_ms } }]);
    return events;
  }

  function notRecorded(question) {
    const list = TOUR.answers.map((a) => `• ${a.question}`).join("\n");
    const live = TOUR.live_url ? ` Ask anything on the live demo: ${TOUR.live_url}` : "";
    return `This is a recorded tour, so it only knows the example questions.${live}\n\nRecorded questions:\n${list}`;
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const path = url.pathname;
    if (url.origin !== location.origin || !API.test(path)) return realFetch(input, init);
    const method = (init.method || "GET").toUpperCase();
    const body = init.body ? JSON.parse(init.body) : {};
    if (path === "/auth/me") return json({ auth_enabled: false, user: null });
    if (path === "/setup") return json({ ...TOUR.setup, demo_mode: true, tour: true, live_url: TOUR.live_url, source_url: TOUR.source_url });
    if (path === "/repos" && method === "GET") return json(TOUR.repos);
    if (path.startsWith("/repos")) return json({ error: "This recorded tour can't change repositories. Run your own copy (free): " + TOUR.source_url }, 403);
    if (path.startsWith("/memories")) return json(method === "GET" ? [] : { deleted: 0 });
    if (path.startsWith("/jobs")) return json([]);
    if (path === "/feedback") return json({ id: 0 }, 201);
    if (path === "/overview") {
      const m = TOUR.maps[url.searchParams.get("repo")];
      return m ? json({ repo: url.searchParams.get("repo"), map: m }) : json({ error: "No map recorded for this repo." }, 404);
    }
    if (path === "/file") {
      const f = TOUR.files[`${url.searchParams.get("repo")}::${url.searchParams.get("path")}`];
      return f ? json(f) : json({ error: "This file was not recorded in the tour; open it on GitHub instead." }, 404);
    }
    if (path === "/ask/stream" || path === "/compare/stream") {
      const repo = path === "/compare/stream" ? (body.repos || []).join(" vs ") : body.repo;
      const answer = findAnswer(body.question, repo);
      const events = answer ? replay(answer) : [[0, { type: "error", error: notRecorded(body.question) }]];
      return new Response(sseStream(events), { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    return json({ error: "Not available in the recorded tour." }, 404);
  };

  // The recorded questions become the example buttons (clicking one picks its repo and asks it).
  // This script runs after the page body is parsed and before app.js wires up the buttons.
  const list = document.querySelector(".examples");
  if (list) {
    list.innerHTML = "";
    for (const a of TOUR.answers.filter((x) => !x.repo.includes(" vs "))) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = a.question;
      b.dataset.repo = a.repo;
      li.appendChild(b);
      list.appendChild(li);
    }
    list.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-repo]");
      const sel = document.querySelector("#repo");
      if (b && sel) sel.value = b.dataset.repo;
    }, true);
  }
})();
