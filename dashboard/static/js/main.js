/**
 * dashboard.js — TCSP-PC Frontend Logic
 * Handles real-time IoT feed, manual predictions, metrics table, trend chart.
 */

"use strict";

// ── Config ──────────────────────────────────────────────────────────────────
const API = "";          // same origin; empty string = relative URLs
const REFRESH_INTERVAL = 5000;   // ms for auto-refresh
const TREND_MAX_POINTS = 30;

// ── State ───────────────────────────────────────────────────────────────────
let autoRefreshTimer = null;
let trendChart       = null;
const trendData      = { labels: [], eco: [], polluting: [], emission: [] };

// ── DOM helpers ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const setText = (id, val) => { const el = $(id); if (el) el.textContent = val; };

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  setText("clock", new Date().toLocaleTimeString());
}
setInterval(updateClock, 1000);
updateClock();

// ── API badge ────────────────────────────────────────────────────────────────
async function checkApiStatus() {
  const badge = $("api-badge");
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    badge.textContent = `● API OK — ${d.models_available.length} model(s)`;
    badge.className = "badge ok";
  } catch {
    badge.textContent = "● API Offline";
    badge.className = "badge error";
  }
}
checkApiStatus();
setInterval(checkApiStatus, 30000);

// ── Trend chart init ─────────────────────────────────────────────────────────
function initTrendChart() {
  const ctx = document.getElementById("trendChart").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trendData.labels,
      datasets: [
        {
          label: "Emission Rate (g/km)",
          data: trendData.emission,
          borderColor: "#f87171",
          backgroundColor: "rgba(248,113,113,.12)",
          tension: 0.4,
          fill: true,
          pointRadius: 2,
          yAxisID: "yLeft",
        },
        {
          label: "Vehicle Density (/km)",
          data: trendData.polluting,
          borderColor: "#60a5fa",
          backgroundColor: "transparent",
          tension: 0.4,
          borderDash: [4, 3],
          pointRadius: 2,
          yAxisID: "yRight",
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#8892a4", font: { size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8892a4", font: { size: 10 }, maxTicksLimit: 8 },
          grid: { color: "rgba(255,255,255,.05)" },
        },
        yLeft: {
          position: "left",
          ticks: { color: "#f87171", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,.05)" },
          title: { display: true, text: "g/km", color: "#8892a4", font: { size: 10 } },
        },
        yRight: {
          position: "right",
          ticks: { color: "#60a5fa", font: { size: 10 } },
          grid: { display: false },
          title: { display: true, text: "V/km", color: "#8892a4", font: { size: 10 } },
        },
      },
    },
  });
}

function pushTrend(emission, density, label) {
  const t = new Date().toLocaleTimeString();
  trendData.labels.push(t);
  trendData.emission.push(emission.toFixed(1));
  trendData.polluting.push(density);
  if (trendData.labels.length > TREND_MAX_POINTS) {
    trendData.labels.shift();
    trendData.emission.shift();
    trendData.polluting.shift();
  }
  if (trendChart) trendChart.update("none");
}

// ── Live IoT Feed ────────────────────────────────────────────────────────────
async function fetchLive() {
  const city      = $("city-select").value;
  const container = $("sensor-card");
  container.innerHTML = `<span class="spinner"></span> Fetching…`;

  try {
    const r = await fetch(`${API}/api/realtime?city=${encodeURIComponent(city)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    renderSensorCard(d);
    updateKPIs(d);
    renderRecs(d.recommendations || []);
    pushTrend(d.sensor_reading.emission_rate, d.sensor_reading.vehicle_density, d.label);
  } catch (e) {
    container.innerHTML = `<p style="color:var(--danger)">⚠ Error: ${e.message}</p>`;
  }
}

function renderSensorCard(d) {
  const s   = d.sensor_reading;
  const eco = d.prediction === 0;
  const probPct = (d.probability * 100).toFixed(1);
  const barColor = eco ? "var(--accent)" : "var(--danger)";

  $("sensor-card").innerHTML = `
    <table>
      <tr><td>City</td>           <td>${s.city}</td></tr>
      <tr><td>Vehicle Type</td>   <td>${s.vehicle_type}</td></tr>
      <tr><td>Emission Rate</td>  <td>${s.emission_rate} g/km</td></tr>
      <tr><td>Vehicle Density</td><td>${s.vehicle_density} /km</td></tr>
      <tr><td>Distance</td>       <td>${s.distance_km} km</td></tr>
      <tr><td>Travel Time</td>    <td>${s.travel_time_min} min</td></tr>
      <tr><td>Hour</td>           <td>${s.hour}:00</td></tr>
      <tr><td>Classification</td>
          <td class="${eco ? "status-eco" : "status-polluting"}">${d.label}</td>
      </tr>
      <tr><td>Model</td>          <td>${d.model_used || "capsnet"}</td></tr>
    </table>
    <div class="prob-bar-wrap" title="Pollution probability">
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:${probPct}%;background:${barColor}"></div>
      </div>
      <span class="prob-bar-label">${probPct}% polluting</span>
    </div>`;
}

function updateKPIs(d) {
  const eco = d.prediction === 0;
  const prev_eco  = parseInt($("kpi-eco-val").textContent) || 0;
  const prev_poll = parseInt($("kpi-poll-val").textContent) || 0;
  setText("kpi-eco-val",      eco ? prev_eco + 1 : prev_eco);
  setText("kpi-poll-val",     eco ? prev_poll : prev_poll + 1);
  setText("kpi-density-val",  d.sensor_reading.vehicle_density);
  setText("kpi-emission-val", d.sensor_reading.emission_rate.toFixed(1));
}

function renderRecs(recs) {
  const list = $("rec-list");
  if (!recs.length) {
    list.innerHTML = `<li class="placeholder">No recommendations at this time.</li>`;
    return;
  }
  list.innerHTML = recs.map(rec => {
    let cls = "rec-warn", icon = "⚠️";
    if (rec.toLowerCase().includes("no immediate")) { cls = "rec-ok";    icon = "✅"; }
    if (rec.toLowerCase().includes("mandatory"))    { cls = "rec-urgent"; icon = "🚨"; }
    if (rec.toLowerCase().includes("restrict"))     { cls = "rec-urgent"; icon = "🚫"; }
    if (rec.toLowerCase().includes("continue"))     { cls = "rec-ok";    icon = "🔍"; }
    return `<li class="${cls}"><span class="rec-icon">${icon}</span>${rec}</li>`;
  }).join("");
}

// ── Auto-refresh toggle ───────────────────────────────────────────────────────
$("btn-refresh").addEventListener("click", fetchLive);

$("auto-refresh").addEventListener("change", function () {
  if (this.checked) {
    fetchLive();
    autoRefreshTimer = setInterval(fetchLive, REFRESH_INTERVAL);
  } else {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
});

// ── Manual predict form ───────────────────────────────────────────────────────
$("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const btn = this.querySelector(".btn-primary");
  btn.textContent = "Classifying…";
  btn.disabled = true;

  const fd = new FormData(this);
  const body = {};
  fd.forEach((val, key) => { body[key] = isNaN(val) ? val : parseFloat(val); });

  try {
    const r = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    renderFormResult(d);
  } catch (err) {
    $("form-result").innerHTML =
      `<div class="result-card"><p style="color:var(--danger)">Error: ${err.message}</p></div>`;
  } finally {
    btn.textContent = "Classify Vehicle";
    btn.disabled = false;
  }
});

function renderFormResult(d) {
  const eco = d.prediction === 0;
  const probPct = (d.probability * 100).toFixed(1);
  const recs = (d.recommendations || [])
    .map(r => `<li>${r}</li>`).join("");

  $("form-result").innerHTML = `
    <div class="result-card ${eco ? "eco" : "polluting"}">
      <div class="result-label ${eco ? "eco" : "polluting"}">
        ${eco ? "✅ Eco-friendly" : "⚠️ Polluting"}
      </div>
      <div class="result-detail">
        Model: <strong>${d.model_used}</strong> &nbsp;|&nbsp;
        Confidence: <strong>${probPct}%</strong> &nbsp;|&nbsp;
        Inference: <strong>${d.inference_ms} ms</strong>
      </div>
      ${recs ? `<div class="result-recs"><strong>Recommendations:</strong><ul>${recs}</ul></div>` : ""}
    </div>`;
}

// ── Metrics table ─────────────────────────────────────────────────────────────
$("btn-load-metrics").addEventListener("click", loadMetrics);

async function loadMetrics() {
  const wrap = $("metrics-table-wrap");
  wrap.innerHTML = `<span class="spinner"></span> Loading…`;
  try {
    const r = await fetch(`${API}/api/metrics`);
    if (!r.ok) throw new Error(`HTTP ${r.status} — run evaluate.py first`);
    const d = await r.json();
    renderMetricsTable(d.metrics || []);
  } catch (e) {
    wrap.innerHTML = `<p style="color:var(--danger)">⚠ ${e.message}</p>`;
  }
}

function renderMetricsTable(rows) {
  if (!rows.length) {
    $("metrics-table-wrap").innerHTML = `<p class="placeholder">No data.</p>`;
    return;
  }
  const best = rows.reduce((a, b) => a.accuracy > b.accuracy ? a : b, rows[0]);
  const cols = ["name", "accuracy", "f1", "auc", "time_s"];
  const labels = { name: "Model", accuracy: "Accuracy", f1: "F1 Score", auc: "AUC-ROC", time_s: "Time (s)" };

  const header = cols.map(c => `<th>${labels[c]}</th>`).join("");
  const body = rows.map(row => {
    const isBest = row.name === best.name;
    const cells = cols.map(c => {
      if (c === "name") return `<td>${row[c]}</td>`;
      const val = parseFloat(row[c]);
      return `<td>${isNaN(val) ? row[c] : val.toFixed(4)}</td>`;
    }).join("");
    return `<tr class="${isBest ? "best-row" : ""}">${cells}</tr>`;
  }).join("");

  $("metrics-table-wrap").innerHTML = `
    <table class="metrics-table">
      <thead><tr>${header}</tr></thead>
      <tbody>${body}</tbody>
    </table>
    <p style="font-size:.75rem;color:var(--muted);margin-top:.5rem">
      ★ Best model highlighted in green.
    </p>`;
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTrendChart();
  // Auto-fetch one live reading on load
  fetchLive();
});