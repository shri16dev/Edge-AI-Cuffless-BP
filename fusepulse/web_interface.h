/*
 * web_interface.h
 * Full HTML dashboard stored in ESP32 flash (PROGMEM).
 * Served at http://192.168.4.1
 * Real-time data via WebSocket ws://192.168.4.1:81
 */

#pragma once

const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FusePulse BP Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:16px}
h1{font-size:20px;font-weight:600;color:#f8fafc;letter-spacing:.5px}
.sub{font-size:12px;color:#64748b;margin-top:2px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.ws-dot{width:8px;height:8px;border-radius:50%;background:#ef4444;transition:background .3s}
.ws-dot.on{background:#22c55e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
.card{background:#1e2130;border:1px solid #2d3348;border-radius:12px;padding:14px 16px}
.card-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px}
.card-value{font-size:28px;font-weight:700;line-height:1;color:#f1f5f9}
.card-unit{font-size:12px;color:#64748b;margin-top:3px}
.bp-card{background:linear-gradient(135deg,#1e2d40,#1e2130);border-color:#2d4a6e}
.bp-val{font-size:36px;color:#60a5fa;font-weight:700}
.hr-val{color:#f87171}
.pwv-val{color:#34d399}
.ptt-val{color:#a78bfa}
.status-row{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.badge{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500}
.badge-green{background:#14532d;color:#86efac}
.badge-red{background:#450a0a;color:#fca5a5}
.badge-amber{background:#451a03;color:#fcd34d}
.badge-gray{background:#1e293b;color:#94a3b8}
.chart-wrap{background:#1e2130;border:1px solid #2d3348;border-radius:12px;padding:14px;margin-bottom:16px}
.chart-label{font-size:11px;color:#64748b;margin-bottom:8px;letter-spacing:.5px}
canvas{width:100%!important;display:block}
.cal-card{background:#1e2130;border:1px solid #2d3348;border-radius:12px;padding:16px;margin-bottom:16px}
.cal-card h3{font-size:14px;color:#f1f5f9;margin-bottom:4px}
.cal-card p{font-size:12px;color:#64748b;margin-bottom:12px;line-height:1.5}
.form-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.form-group{flex:1;min-width:100px}
label{display:block;font-size:11px;color:#94a3b8;margin-bottom:4px}
input[type=number]{width:100%;background:#0f1117;border:1px solid #374151;border-radius:8px;color:#f1f5f9;padding:8px 10px;font-size:15px;outline:none}
input[type=number]:focus{border-color:#3b82f6}
button{background:#2563eb;color:#fff;border:none;border-radius:8px;padding:9px 20px;font-size:13px;cursor:pointer;font-weight:500;transition:background .2s}
button:hover{background:#1d4ed8}
button.danger{background:#dc2626}
button.danger:hover{background:#b91c1c}
.cal-status{font-size:12px;color:#86efac;margin-top:8px;min-height:18px}
.history-table{width:100%;border-collapse:collapse;font-size:13px}
.history-table th{text-align:left;padding:6px 8px;color:#64748b;font-size:11px;border-bottom:1px solid #2d3348}
.history-table td{padding:6px 8px;border-bottom:1px solid #1e2130;color:#e2e8f0}
.history-table tr:last-child td{border-bottom:none}
.trend-up{color:#f87171} .trend-ok{color:#86efac} .trend-warn{color:#fcd34d}
footer{text-align:center;font-size:11px;color:#334155;margin-top:20px}
@media(max-width:420px){.bp-val{font-size:28px}}
</style>
</head>
<body>

<header>
  <div>
    <h1>FusePulse</h1>
    <div class="sub">Cuffless BP Monitor — PTT Method</div>
  </div>
  <div style="display:flex;align-items:center;gap:6px">
    <div class="ws-dot" id="dot"></div>
    <span style="font-size:12px;color:#64748b" id="ws-label">Connecting...</span>
  </div>
</header>

<!-- Status badges -->
<div class="status-row">
  <span class="badge badge-gray"  id="badge-cal">Not calibrated</span>
  <span class="badge badge-gray"  id="badge-fin">No finger</span>
  <span class="badge badge-gray"  id="badge-mot">Motion OK</span>
  <span class="badge badge-gray"  id="badge-bp">--</span>
</div>

<!-- Vitals cards -->
<div class="grid">
  <div class="card bp-card" style="grid-column:span 2">
    <div class="card-label">Blood Pressure</div>
    <div class="bp-val" id="v-bp">-- / --</div>
    <div class="card-unit">mmHg  systolic / diastolic</div>
  </div>
  <div class="card">
    <div class="card-label">Heart Rate</div>
    <div class="card-value hr-val" id="v-hr">--</div>
    <div class="card-unit">bpm</div>
  </div>
  <div class="card">
    <div class="card-label">PTT</div>
    <div class="card-value ptt-val" id="v-ptt">--</div>
    <div class="card-unit">milliseconds</div>
  </div>
  <div class="card">
    <div class="card-label">PWV</div>
    <div class="card-value pwv-val" id="v-pwv">--</div>
    <div class="card-unit">m/s  (pulse wave velocity)</div>
  </div>
  <div class="card">
    <div class="card-label">ASI</div>
    <div class="card-value pwv-val" id="v-asi">--</div>
    <div class="card-unit">m/s  (arterial stiffness)</div>
  </div>
</div>

<!-- Live waveforms -->
<div class="chart-wrap">
  <div class="chart-label">LIVE PPG WAVEFORMS</div>
  <canvas id="ppgChart" height="100"></canvas>
</div>

<!-- BP trend -->
<div class="chart-wrap">
  <div class="chart-label">BP TREND (last 30 readings)</div>
  <canvas id="bpChart" height="80"></canvas>
</div>

<!-- Calibration -->
<div class="cal-card">
  <h3>One-Shot Calibration</h3>
  <p>Take a single reading with a traditional cuff right now, enter the values below, then click Calibrate. The device will personalise its formula to your arterial characteristics.</p>
  <div class="form-row">
    <div class="form-group">
      <label>Systolic (SBP)</label>
      <input type="number" id="inp-sbp" min="70" max="200" placeholder="e.g. 120">
    </div>
    <div class="form-group">
      <label>Diastolic (DBP)</label>
      <input type="number" id="inp-dbp" min="40" max="130" placeholder="e.g. 80">
    </div>
  </div>
  <button onclick="doCalibrate()">Calibrate now</button>
  <button class="danger" onclick="resetCal()" style="margin-left:8px">Reset</button>
  <div class="cal-status" id="cal-msg"></div>
</div>

<!-- Reading history -->
<div class="cal-card">
  <h3>Reading History</h3>
  <table class="history-table">
    <thead><tr><th>#</th><th>Time</th><th>SBP</th><th>DBP</th><th>HR</th><th>PTT</th><th>Status</th></tr></thead>
    <tbody id="hist-body"><tr><td colspan="7" style="color:#64748b;text-align:center">No readings yet</td></tr></tbody>
  </table>
</div>

<footer>FusePulse v1.1 — Formula-based PTT estimation. Not a medical device.</footer>

<script>
// ============================================================
// WebSocket
// ============================================================
var ws, reconnectTimer;
var waveW = new Array(80).fill(0);
var waveF = new Array(80).fill(0);
var bpHistory = [];          // {t, sbp, dbp}
var readingCount = 0;
var lastSBP = 0, lastDBP = 0;

function connect() {
  ws = new WebSocket('ws://' + location.hostname + ':81');
  ws.onopen = function() {
    document.getElementById('dot').className = 'ws-dot on';
    document.getElementById('ws-label').textContent = 'Live';
    clearTimeout(reconnectTimer);
  };
  ws.onclose = function() {
    document.getElementById('dot').className = 'ws-dot';
    document.getElementById('ws-label').textContent = 'Reconnecting...';
    reconnectTimer = setTimeout(connect, 2000);
  };
  ws.onmessage = function(e) {
    try { handleData(JSON.parse(e.data)); } catch(err) {}
  };
}

var warmingUp = false;

function handleData(d) {
  // Waveform buffers
  waveW.push(parseFloat(d.w)); waveW.shift();
  waveF.push(parseFloat(d.f)); waveF.shift();

  var sbp = parseInt(d.sbp), dbp = parseInt(d.dbp);
  var hr  = parseInt(d.hr),  ptt = parseInt(d.ptt);
  var pwv = parseFloat(d.pwv), asi = parseFloat(d.asi);

  // FIX 1 — clear display when finger removed
  if (!d.fin) {
    document.getElementById('v-bp').textContent  = '-- / --';
    document.getElementById('v-hr').textContent  = '--';
    document.getElementById('v-ptt').textContent = '--';
    document.getElementById('v-pwv').textContent = '--';
    document.getElementById('v-asi').textContent = '--';
    document.getElementById('badge-bp').className    = 'badge badge-gray';
    document.getElementById('badge-bp').textContent  = '--';
    lastSBP = 0; lastDBP = 0;
  }

  // FIX 2 — warmup indicator: finger on but sbp still 0
  if (d.fin && sbp === 0) {
    if (!warmingUp) {
      warmingUp = true;
      document.getElementById('v-bp').textContent = 'Warming up...';
      document.getElementById('badge-bp').className   = 'badge badge-amber';
      document.getElementById('badge-bp').textContent = 'Stabilising';
    }
  } else {
    warmingUp = false;
  }

  if (d.fin && sbp > 0) {
    document.getElementById('v-bp').textContent  = sbp + ' / ' + dbp;
    document.getElementById('v-hr').textContent  = hr  || '--';
    document.getElementById('v-ptt').textContent = ptt > 0 ? ptt  : '--';
    document.getElementById('v-pwv').textContent = pwv > 0 ? pwv.toFixed(2) : '--';
    document.getElementById('v-asi').textContent = asi > 0 ? asi.toFixed(2) : '--';

    if (sbp !== lastSBP || dbp !== lastDBP) {
      lastSBP = sbp; lastDBP = dbp;
      var now = new Date();
      bpHistory.push({t: now.toLocaleTimeString(), sbp, dbp, hr, ptt});
      if (bpHistory.length > 30) bpHistory.shift();
      addHistoryRow(bpHistory[bpHistory.length-1], ++readingCount);
      drawBPChart();
    }

    var bpBadge = document.getElementById('badge-bp');
    if      (sbp >= 180 || dbp >= 120) { bpBadge.className='badge badge-red';   bpBadge.textContent='Crisis'; }
    else if (sbp >= 140 || dbp >= 90)  { bpBadge.className='badge badge-red';   bpBadge.textContent='Hypertension'; }
    else if (sbp >= 130 || dbp >= 80)  { bpBadge.className='badge badge-amber'; bpBadge.textContent='Elevated'; }
    else                                { bpBadge.className='badge badge-green'; bpBadge.textContent='Normal'; }
  }

  var calB = document.getElementById('badge-cal');
  calB.className   = d.cal ? 'badge badge-green' : 'badge badge-amber';
  calB.textContent = d.cal ? 'Calibrated' : 'Pre-cal estimate';

  var finB = document.getElementById('badge-fin');
  finB.className   = d.fin ? 'badge badge-green' : 'badge badge-red';
  finB.textContent = d.fin ? 'Finger detected'   : 'No finger';

  var motB = document.getElementById('badge-mot');
  motB.className   = d.mot ? 'badge badge-amber' : 'badge badge-green';
  motB.textContent = d.mot ? 'Motion detected'   : 'Stable';

  drawPPGChart();
}

// ============================================================
// PPG waveform canvas
// ============================================================
var ppgCanvas = document.getElementById('ppgChart');
var ppgCtx = ppgCanvas.getContext('2d');

function drawPPGChart() {
  var W = ppgCanvas.offsetWidth, H = ppgCanvas.height;
  ppgCanvas.width = W;
  ppgCtx.clearRect(0, 0, W, H);

  // Grid
  ppgCtx.strokeStyle = '#1e293b'; ppgCtx.lineWidth = 1;
  for (var gy = H*0.25; gy < H; gy += H*0.25) {
    ppgCtx.beginPath(); ppgCtx.moveTo(0,gy); ppgCtx.lineTo(W,gy); ppgCtx.stroke();
  }

  drawWave(ppgCtx, waveW, W, H, '#60a5fa', 'Wrist', 4);
  drawWave(ppgCtx, waveF, W, H, '#f87171', 'Finger', 18);
}

function drawWave(ctx, data, W, H, color, label, textY) {
  var max = Math.max(...data), min = Math.min(...data);
  var range = max - min || 1;
  ctx.beginPath();
  ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  for (var i = 0; i < data.length; i++) {
    var x = (i / (data.length-1)) * W;
    var y = H - ((data[i] - min) / range) * H * 0.85 - H*0.05;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = color; ctx.font = '10px sans-serif';
  ctx.fillText(label, 4, textY);
}

// ============================================================
// BP trend chart
// ============================================================
var bpCanvas = document.getElementById('bpChart');
var bpCtx = bpCanvas.getContext('2d');

function drawBPChart() {
  if (bpHistory.length < 2) return;
  var W = bpCanvas.offsetWidth, H = bpCanvas.height;
  bpCanvas.width = W;
  bpCtx.clearRect(0, 0, W, H);

  var sbpVals = bpHistory.map(r => r.sbp);
  var dbpVals = bpHistory.map(r => r.dbp);
  var allVals = sbpVals.concat(dbpVals);
  var lo = Math.min(...allVals) - 10, hi = Math.max(...allVals) + 10;
  var range = hi - lo || 1;

  // Reference line 120/80
  [120, 80].forEach(function(ref) {
    bpCtx.strokeStyle = '#334155'; bpCtx.lineWidth = 1; bpCtx.setLineDash([4,3]);
    var ry = H - ((ref - lo)/range)*H*0.85 - H*0.05;
    bpCtx.beginPath(); bpCtx.moveTo(0,ry); bpCtx.lineTo(W,ry); bpCtx.stroke();
  });
  bpCtx.setLineDash([]);

  function plotLine(vals, color) {
    bpCtx.beginPath(); bpCtx.strokeStyle = color; bpCtx.lineWidth = 2;
    vals.forEach(function(v, i) {
      var x = (i / (vals.length-1)) * W;
      var y = H - ((v - lo) / range) * H * 0.85 - H*0.05;
      if (i===0) bpCtx.moveTo(x,y); else bpCtx.lineTo(x,y);
    });
    bpCtx.stroke();
    vals.forEach(function(v, i) {
      var x = (i / (vals.length-1)) * W;
      var y = H - ((v - lo) / range) * H * 0.85 - H*0.05;
      bpCtx.fillStyle = color; bpCtx.beginPath(); bpCtx.arc(x,y,3,0,Math.PI*2); bpCtx.fill();
    });
  }
  plotLine(sbpVals, '#60a5fa');   // SBP blue
  plotLine(dbpVals, '#f87171');   // DBP red

  bpCtx.fillStyle='#60a5fa'; bpCtx.font='10px sans-serif'; bpCtx.fillText('SBP',4,10);
  bpCtx.fillStyle='#f87171'; bpCtx.fillText('DBP',34,10);
}

// ============================================================
// History table
// ============================================================
function addHistoryRow(r, n) {
  var tbody = document.getElementById('hist-body');
  if (n === 1) tbody.innerHTML = '';
  var cat = r.sbp >= 140 ? '<span class="trend-up">High</span>'
          : r.sbp >= 130 ? '<span class="trend-warn">Elevated</span>'
          : '<span class="trend-ok">Normal</span>';
  var tr = tbody.insertRow(0);
  tr.innerHTML = '<td>' + n + '</td><td>' + r.t + '</td>'
    + '<td>' + r.sbp + '</td><td>' + r.dbp + '</td>'
    + '<td>' + (r.hr||'--') + '</td><td>' + (r.ptt||'--') + ' ms</td>'
    + '<td>' + cat + '</td>';
}

// ============================================================
// Calibration
// ============================================================
function doCalibrate() {
  var sbp = document.getElementById('inp-sbp').value;
  var dbp = document.getElementById('inp-dbp').value;
  if (!sbp || !dbp) { document.getElementById('cal-msg').textContent = 'Enter both values.'; return; }
  document.getElementById('cal-msg').textContent = 'Sending...';
  fetch('/calibrate', {
    method: 'POST',
    headers: {'Content-Type':'application/x-www-form-urlencoded'},
    body: 'sbp=' + sbp + '&dbp=' + dbp
  }).then(r => r.json()).then(function(d) {
    if (d.ok) {
      document.getElementById('cal-msg').textContent =
        'Calibrated at PTT ' + parseFloat(d.ptt).toFixed(1) + ' ms  |  SBP ' + d.sbp + ' mmHg';
    } else {
      document.getElementById('cal-msg').textContent = 'Error: ' + (d.error||'unknown');
    }
  }).catch(function() {
    document.getElementById('cal-msg').textContent = 'Network error — retry.';
  });
}

function resetCal() {
  document.getElementById('inp-sbp').value = '';
  document.getElementById('inp-dbp').value = '';
  document.getElementById('cal-msg').textContent = 'Reset. Re-calibrate when ready.';
}

// ============================================================
// Start
// ============================================================
connect();
setInterval(function() { drawPPGChart(); }, 100);
</script>
</body>
</html>
)rawliteral";
