/* ai-investing cockpit - shared frontend helpers
 *
 * Loaded on every page (after Tailwind, before page scripts).
 * Exposes a single global: window.Cockpit
 *
 * Responsibilities:
 *   - Toast notifications (replaces alert())
 *   - Persistent topbar with regime + mode + bot state + connection
 *   - Error count badge sync (any page)
 *   - Keyboard shortcuts modal (?)
 *   - Formatting helpers
 *   - Tooltip auto-init from data attributes
 */
(function () {
  "use strict";

  const Cockpit = window.Cockpit = window.Cockpit || {};

  // ----------------------------------------------------------------------
  // Formatting helpers
  // ----------------------------------------------------------------------
  Cockpit.fmt = {
    money(v, opts = {}) {
      if (v == null || Number.isNaN(Number(v))) return "—";
      const n = Number(v);
      return "$" + n.toLocaleString(undefined, {
        maximumFractionDigits: opts.maxFrac ?? 0,
        minimumFractionDigits: opts.minFrac ?? 0,
      });
    },
    pct(v, digits = 2) {
      if (v == null || Number.isNaN(Number(v))) return "—";
      return (Number(v) * 100).toFixed(digits) + "%";
    },
    pctSigned(v, digits = 2) {
      if (v == null || Number.isNaN(Number(v))) return "—";
      const n = Number(v) * 100;
      return (n >= 0 ? "+" : "") + n.toFixed(digits) + "%";
    },
    moneySigned(v) {
      if (v == null || Number.isNaN(Number(v))) return "—";
      const n = Number(v);
      const sign = n >= 0 ? "+" : "-";
      return sign + "$" + Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
    },
    time(iso) {
      if (!iso) return "—";
      try { return new Date(iso).toLocaleString(); } catch { return iso; }
    },
    relTime(iso) {
      if (!iso) return "";
      try {
        const d = new Date(iso);
        const diff = (Date.now() - d.getTime()) / 1000;
        if (diff < 5) return "just now";
        if (diff < 60) return Math.floor(diff) + "s ago";
        if (diff < 3600) return Math.floor(diff / 60) + "m ago";
        if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
        return Math.floor(diff / 86400) + "d ago";
      } catch { return ""; }
    },
    escape(s) {
      if (s == null) return "";
      return String(s)
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    },
  };

  // ----------------------------------------------------------------------
  // Toasts
  // ----------------------------------------------------------------------
  function _toastHost() {
    let host = document.querySelector(".toast-host");
    if (!host) {
      host = document.createElement("div");
      host.className = "toast-host";
      document.body.appendChild(host);
    }
    return host;
  }

  Cockpit.toast = function (msg, opts = {}) {
    const { kind = "info", title = null, ttl = 4200 } = opts;
    const host = _toastHost();
    const el = document.createElement("div");
    el.className = "toast toast-" + kind;
    if (title) {
      el.innerHTML = `<div class="toast-title">${Cockpit.fmt.escape(title)}</div><div class="toast-body">${Cockpit.fmt.escape(msg)}</div>`;
    } else {
      el.textContent = msg;
    }
    host.appendChild(el);
    // Trigger entrance after layout
    requestAnimationFrame(() => el.classList.add("show"));
    const remove = () => {
      el.classList.remove("show");
      setTimeout(() => el.remove(), 300);
    };
    el.addEventListener("click", remove);
    setTimeout(remove, ttl);
    return el;
  };
  Cockpit.toastSuccess = (m, t) => Cockpit.toast(m, { kind: "success", title: t });
  Cockpit.toastError = (m, t) => Cockpit.toast(m, { kind: "error", title: t, ttl: 6500 });
  Cockpit.toastWarn = (m, t) => Cockpit.toast(m, { kind: "warn", title: t });

  // ----------------------------------------------------------------------
  // Wrapped fetch with friendly errors
  // ----------------------------------------------------------------------
  Cockpit.api = {
    async get(path) {
      const r = await fetch(path);
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(detail.detail || detail.error || `HTTP ${r.status}`);
      }
      return r.json();
    },
    async post(path, body) {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body == null ? null : JSON.stringify(body),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(detail.detail || detail.error || `HTTP ${r.status}`);
      }
      return r.json();
    },
  };

  // ----------------------------------------------------------------------
  // Topbar (rendered into element with id="cockpit-topbar")
  // ----------------------------------------------------------------------
  function _regimePill(r) {
    const m = { bull: "pill-green", chop: "pill-blue", bear: "pill-yellow", crisis: "pill-red" };
    return m[r] || "pill-gray";
  }

  Cockpit.renderTopbar = function (state) {
    const slot = document.getElementById("cockpit-topbar");
    if (!slot) return;
    const ctrl = state.control || {};
    const reg = state.regime || {};
    const eff = reg.effective || "?";
    const mode = ctrl.trading_mode || "paper";
    const paused = !!ctrl.paused;
    const modePill = mode === "live"
      ? '<span class="pill pill-red pill-dot pulse-soft">LIVE</span>'
      : '<span class="pill pill-green pill-dot">PAPER</span>';
    const statePill = paused
      ? '<span class="pill pill-yellow">paused</span>'
      : '<span class="pill pill-green">active</span>';
    const regPill = `<span class="pill ${_regimePill(eff)}">${Cockpit.fmt.escape(eff)}</span>`;
    slot.innerHTML = `
      <div class="topbar-stats">
        <span class="text-xs text-gray-500">mode</span> ${modePill}
        <span class="text-xs text-gray-500 ml-2">regime</span> ${regPill}
        <span class="text-xs text-gray-500 ml-2">bot</span> ${statePill}
      </div>
    `;
  };

  Cockpit.markConnection = function (state) {
    const dot = document.getElementById("cockpit-conn-dot");
    if (!dot) return;
    if (state === "live") dot.className = "topbar-brand-dot";
    else if (state === "offline") dot.className = "topbar-brand-dot offline";
    else dot.className = "topbar-brand-dot idle";
  };

  // ----------------------------------------------------------------------
  // Error badge sync (single source of truth across pages)
  // ----------------------------------------------------------------------
  Cockpit.syncErrorBadge = async function () {
    try {
      const j = await Cockpit.api.get("/api/errors");
      const c = j.counts || {};
      const open = (c.error || 0) + (c.warning || 0);
      document.querySelectorAll("#nav-err-badge, [data-err-badge]").forEach((b) => {
        if (open > 0) { b.textContent = String(open); b.style.display = "inline-block"; }
        else { b.style.display = "none"; }
      });
    } catch (e) { /* silent */ }
  };

  // ----------------------------------------------------------------------
  // Tooltips: any element with [data-tt="text"] gets a hover bubble
  // ----------------------------------------------------------------------
  Cockpit.initTooltips = function (root = document) {
    root.querySelectorAll("[data-tt]:not(.tt-ready)").forEach((host) => {
      const text = host.getAttribute("data-tt");
      if (!text) return;
      host.classList.add("tt-ready");
      // Always set native title for accessibility / fallback.
      if (!host.hasAttribute("title")) host.setAttribute("title", text);
      // For explicit .tt elements (typically a `?` span placed next to a label),
      // build the rich hover bubble. For everything else (buttons, pills, etc.)
      // the native title attribute is sufficient and avoids mangling layout.
      if (host.classList.contains("tt")) {
        // Wrap any plain text content into a styled .tt-trigger so the
        // little circular ? badge looks consistent across pages.
        if (!host.querySelector(".tt-trigger")) {
          const existing = host.textContent.trim();
          host.textContent = "";
          const trigger = document.createElement("span");
          trigger.className = "tt-trigger";
          trigger.setAttribute("aria-label", text);
          trigger.tabIndex = 0;
          trigger.textContent = existing || "?";
          host.appendChild(trigger);
        }
        if (!host.querySelector(".tt-content")) {
          const bubble = document.createElement("span");
          bubble.className = "tt-content";
          bubble.textContent = text;
          host.appendChild(bubble);
        }
      }
    });
  };

  // ----------------------------------------------------------------------
  // Keyboard shortcut modal (?)
  // ----------------------------------------------------------------------
  const SHORTCUTS = [
    { key: "P", desc: "Pause/resume bot (dashboard only)" },
    { key: "R", desc: "Refresh data (dashboard only)" },
    { key: "G then D", desc: "Go to Dashboard" },
    { key: "G then T", desc: "Go to Trading" },
    { key: "G then M", desc: "Go to Models" },
    { key: "G then S", desc: "Go to Settings" },
    { key: "G then U", desc: "Go to Updates" },
    { key: "G then E", desc: "Go to Errors" },
    { key: "?", desc: "Show this help" },
    { key: "Esc", desc: "Close dialog" },
  ];

  function _ensureShortcutModal() {
    let bd = document.getElementById("cockpit-shortcut-modal");
    if (bd) return bd;
    bd = document.createElement("div");
    bd.id = "cockpit-shortcut-modal";
    bd.className = "modal-backdrop";
    bd.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <div class="flex items-center justify-between mb-3">
          <div class="text-lg font-semibold">Keyboard shortcuts</div>
          <button class="btn btn-ghost btn-sm" data-close>Close</button>
        </div>
        <div class="space-y-2 text-sm">
          ${SHORTCUTS.map(s => `
            <div class="flex items-center justify-between gap-4">
              <span class="kbd">${s.key}</span>
              <span class="text-gray-400 text-right flex-1">${s.desc}</span>
            </div>
          `).join("")}
        </div>
      </div>
    `;
    document.body.appendChild(bd);
    bd.addEventListener("click", (e) => {
      if (e.target === bd || e.target.matches("[data-close]")) bd.classList.remove("open");
    });
    return bd;
  }
  Cockpit.showShortcuts = function () { _ensureShortcutModal().classList.add("open"); };

  // ----------------------------------------------------------------------
  // Global keyboard handler
  // ----------------------------------------------------------------------
  let _gPrefix = false;
  let _gPrefixTimer = null;
  function _gotoPrefix() {
    _gPrefix = true;
    clearTimeout(_gPrefixTimer);
    _gPrefixTimer = setTimeout(() => { _gPrefix = false; }, 900);
  }
  function _handleKey(e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "?") { e.preventDefault(); Cockpit.showShortcuts(); return; }
    if (e.key === "Escape") {
      document.querySelectorAll(".modal-backdrop.open").forEach(b => b.classList.remove("open"));
      return;
    }
    if (_gPrefix) {
      _gPrefix = false;
      const map = { d: "/", t: "/trading", m: "/models", s: "/settings", u: "/updates", e: "/errors" };
      const target = map[e.key.toLowerCase()];
      if (target) { e.preventDefault(); window.location.href = target; }
      return;
    }
    if (e.key === "g" || e.key === "G") { _gotoPrefix(); }
  }
  document.addEventListener("keydown", _handleKey);

  // ----------------------------------------------------------------------
  // Visibility-aware polling
  // ----------------------------------------------------------------------
  // Background tabs in Chrome throttle setInterval to >= 1s and queue
  // pending fires that all flush when the tab regains focus, causing burst
  // lag on every tab switch. Cockpit.poll() runs `fn` only when the tab is
  // visible and re-fires immediately on visibility change so the user sees
  // fresh data the moment they refocus. Returns a handle with cancel().
  Cockpit.poll = function (fn, ms, opts = {}) {
    const { runImmediately = true, name = "" } = opts;
    let timer = null;
    let cancelled = false;
    let running = false;

    async function tick() {
      if (cancelled || document.hidden) return;
      if (running) return; // skip overlapping calls — avoids backlog on slow endpoints
      running = true;
      try { await fn(); }
      catch (e) { if (name) console.warn(`[poll:${name}]`, e); }
      finally { running = false; }
    }
    function schedule() {
      if (timer != null) clearInterval(timer);
      timer = setInterval(tick, ms);
    }
    function onVisibility() {
      if (cancelled) return;
      if (document.hidden) {
        if (timer != null) { clearInterval(timer); timer = null; }
      } else {
        tick(); // immediate refresh on refocus
        schedule();
      }
    }
    document.addEventListener("visibilitychange", onVisibility);
    if (runImmediately) tick();
    schedule();
    return {
      cancel() {
        cancelled = true;
        if (timer != null) { clearInterval(timer); timer = null; }
        document.removeEventListener("visibilitychange", onVisibility);
      },
      now: tick,
    };
  };

  // ----------------------------------------------------------------------
  // Save-confirmation pulse
  // ----------------------------------------------------------------------
  // After saving a setting (autopilot toggle, schedule edit, etc.) the
  // value re-paints from the next poll — so the user sees no immediate
  // confirmation that their click did anything. Flash a brief green ring
  // around the saved control so the action feels acknowledged.
  Cockpit.flashSaved = function (target, label) {
    const el = typeof target === "string" ? document.getElementById(target) : target;
    if (!el) return;
    el.classList.remove("cockpit-flash-saved");
    // force reflow so the animation restarts if called twice in a row
    void el.offsetWidth;
    el.classList.add("cockpit-flash-saved");
    if (label) Cockpit.toastSuccess(label);
    setTimeout(() => el.classList.remove("cockpit-flash-saved"), 1400);
  };

  // ----------------------------------------------------------------------
  // Elapsed-time ticker
  // ----------------------------------------------------------------------
  // A 60-second LLM call shouldn't look frozen. Wrap an async operation
  // with Cockpit.elapsedTicker(el, { text }) and the element will read
  // "<text>... 12s" with the seconds counter live until you call .stop().
  Cockpit.elapsedTicker = function (target, opts = {}) {
    const el = typeof target === "string" ? document.getElementById(target) : target;
    if (!el) return { stop: () => {} };
    const text = opts.text || "working";
    const t0 = performance.now();
    function paint() {
      const s = ((performance.now() - t0) / 1000).toFixed(0);
      el.textContent = `${text}... ${s}s`;
    }
    paint();
    const id = setInterval(paint, 500);
    return {
      stop(finalText) {
        clearInterval(id);
        const total = ((performance.now() - t0) / 1000).toFixed(2);
        if (finalText) el.textContent = `${finalText} (${total}s)`;
        return parseFloat(total);
      },
      update(newText) { opts.text = newText; paint(); },
    };
  };

  // ----------------------------------------------------------------------
  // Sparkline (tiny inline SVG)
  // ----------------------------------------------------------------------
  Cockpit.sparkline = function (values, opts = {}) {
    const { width = 100, height = 28, color = "#34d399" } = opts;
    if (!values || values.length < 2) return "";
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const pts = values.map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const last = values[values.length - 1];
    const first = values[0];
    const trendColor = last >= first ? "#34d399" : "#f87171";
    const c = opts.color || trendColor;
    return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>`;
  };

  // ----------------------------------------------------------------------
  // Boot
  // ----------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    Cockpit.initTooltips();
    Cockpit.syncErrorBadge();
    setInterval(Cockpit.syncErrorBadge, 15000);
  });
})();
