"""Static HTML for the monitoring dashboard.

Kept as a Python string so it ships inside the app image with no static-file
mount or template engine. It is fully self-contained (no external CDN/assets)
and polls the operational JSON endpoints from the browser.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SmartPark — Monitoring</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0f1720; color: #e6edf3;
  }
  header {
    padding: 18px 24px; background: #111c2b; border-bottom: 1px solid #223;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  header h1 { font-size: 18px; margin: 0; font-weight: 650; }
  .pill { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: #1d2c40; }
  .pill.ok { background: #16371f; color: #6ee787; }
  .pill.bad { background: #3a1a1a; color: #ff9a9a; }
  main { padding: 24px; display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section { background: #111c2b; border: 1px solid #223; border-radius: 10px; padding: 16px; }
  h2 { font-size: 14px; margin: 0 0 12px; text-transform: uppercase; letter-spacing: .06em; color: #9db2c8; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1c2a3a; }
  th { color: #8aa0b6; font-weight: 600; position: sticky; top: 0; background: #111c2b; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 6px; }
  .badge.ok { background: #16371f; color: #6ee787; }
  .badge.error { background: #3a1a1a; color: #ff9a9a; }
  .bar { height: 6px; border-radius: 3px; background: #22364a; overflow: hidden; margin-top: 4px; }
  .bar > span { display: block; height: 100%; background: #3fb950; }
  .uuid { font-family: ui-monospace, Menlo, monospace; font-size: 12px; padding: 6px 8px;
          border-bottom: 1px solid #1c2a3a; word-break: break-all; }
  .muted { color: #6f8296; font-size: 12px; }
  .tablewrap { max-height: 60vh; overflow: auto; }
  footer { padding: 12px 24px; color: #6f8296; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>🅿️ SmartPark Monitoring</h1>
  <span class="pill" id="health">health: …</span>
  <span class="pill" id="count">0 car parks</span>
  <span class="pill" id="refresh">next refresh …</span>
</header>
<main>
  <section>
    <h2>Car-park statuses</h2>
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Car park</th><th>Status</th><th class="num">Empty</th>
            <th class="num">Occupied</th><th class="num">Total</th>
            <th>Confidence</th><th class="num">ms</th><th>Last seen</th>
          </tr>
        </thead>
        <tbody id="rows"><tr><td colspan="8" class="muted">No data yet.</td></tr></tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Unique UUIDs (last <span id="win">30</span>s)</h2>
    <div id="uuids"><div class="muted">None seen recently.</div></div>
  </section>
  <section style="grid-column: 1 / -1;">
    <h2>Availability plot (matplotlib, on demand)</h2>
    <img id="plot" alt="Availability plot"
         style="width: 100%; border-radius: 8px; background: #0f1720;" />
    <div class="muted">Rendered server-side by /api/operations/plot.png</div>
  </section>
</main>
<footer>Auto-refreshes every 3s · reads /api/operations/statuses and /api/operations/recent-uuids</footer>
<script>
const REFRESH_MS = 3000;
let countdown = REFRESH_MS / 1000;

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function loadStatuses() {
  const res = await fetch('/api/operations/statuses');
  const data = await res.json();
  document.getElementById('count').textContent = data.count + ' car parks';
  const rows = document.getElementById('rows');
  if (!data.statuses.length) {
    rows.innerHTML = '<tr><td colspan="8" class="muted">No data yet — call /api/find-carparks.</td></tr>';
    return;
  }
  rows.innerHTML = data.statuses.map(s => {
    const conf = (s.confidence_score * 100).toFixed(0);
    const badge = s.status === 'ok' ? 'ok' : 'error';
    const when = s.last_seen ? esc(s.last_seen.replace('T', ' ').slice(0, 19)) : '—';
    const confCell = s.status === 'ok'
      ? conf + '%<div class="bar"><span style="width:' + conf + '%"></span></div>'
      : esc(s.detail || 'error');
    return '<tr><td>' + esc(s.carpark_id) + '</td>' +
      '<td><span class="badge ' + badge + '">' + s.status + '</span></td>' +
      '<td class="num">' + s.empty_count + '</td>' +
      '<td class="num">' + s.occupied_count + '</td>' +
      '<td class="num">' + s.total_spaces + '</td>' +
      '<td>' + confCell + '</td>' +
      '<td class="num">' + (s.speed_inference ?? 0) + '</td>' +
      '<td class="muted">' + when + '</td></tr>';
  }).join('');
}

async function loadUuids() {
  const res = await fetch('/api/operations/recent-uuids');
  const data = await res.json();
  document.getElementById('win').textContent = data.window_seconds;
  const box = document.getElementById('uuids');
  box.innerHTML = data.uuids.length
    ? data.uuids.map(u => '<div class="uuid">' + esc(u) + '</div>').join('')
    : '<div class="muted">None seen in the last ' + data.window_seconds + 's.</div>';
}

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    const el = document.getElementById('health');
    el.textContent = 'model: ' + (h.model_loaded ? 'loaded' : 'not loaded');
    el.className = 'pill ' + (h.model_loaded ? 'ok' : 'bad');
  } catch (e) {
    const el = document.getElementById('health');
    el.textContent = 'health: unreachable'; el.className = 'pill bad';
  }
}

function loadPlot() {
  // Cache-bust so the browser re-fetches a freshly rendered PNG each cycle.
  document.getElementById('plot').src = '/api/operations/plot.png?t=' + Date.now();
}

async function refresh() {
  await Promise.allSettled([loadStatuses(), loadUuids(), loadHealth()]);
  loadPlot();
  countdown = REFRESH_MS / 1000;
}

setInterval(() => {
  countdown -= 1;
  document.getElementById('refresh').textContent =
    'next refresh ' + Math.max(countdown, 0) + 's';
  if (countdown <= 0) refresh();
}, 1000);

refresh();
</script>
</body>
</html>
"""
