/* ── State ─────────────────────────────────────────────────────────────────── */
let currentSymbol       = 'JPY=X';
let trajData            = [];
let currentTrajWindow   = '5d';

/* ── Init ──────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('symbolSelect');
  if (sel) {
    currentSymbol = sel.value;
    sel.addEventListener('change', () => {
      currentSymbol = sel.value;
      loadAll();
    });
  }
  document.getElementById('refreshBtn')?.addEventListener('click', loadAll);

  document.querySelectorAll('.traj-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.traj-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTrajWindow = btn.dataset.window;
      renderTrajectoryCanvas(trajData, currentTrajWindow);
    });
  });

  loadOverview();
  loadAll();
});

/* ── Helpers ───────────────────────────────────────────────────────────────── */
function setLoading(on) {
  document.getElementById('loadingOverlay')?.classList.toggle('hidden', !on);
}
function showError(msg) {
  const el = document.getElementById('errorBanner');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 5000);
}
function fmt(v, d = 2, suf = '') {
  if (v == null) return 'N/A';
  return Number(v).toFixed(d) + suf;
}
function fmtPct(v, plus = true) {
  if (v == null) return 'N/A';
  const n = Number(v) * 100;
  return (plus && n >= 0 ? '+' : '') + n.toFixed(2) + '%';
}
function fmtPrice(v) {
  if (v == null) return 'N/A';
  const n = Number(v);
  if (n >= 10000) return n.toLocaleString('ja-JP', { maximumFractionDigits: 2 });
  if (n >= 100)   return n.toFixed(2);
  return n.toFixed(4);
}

/* ── Overview ──────────────────────────────────────────────────────────────── */
async function loadOverview() {
  try {
    const res  = await fetch('/api/overview');
    const data = await res.json();
    renderOverview(data.overview);
  } catch (e) { console.warn('overview error', e); }
}

function renderOverview(overview) {
  const grid = document.getElementById('fractalGrid');
  if (!grid) return;
  grid.innerHTML = overview.map(item => {
    const cls = item.bullish === true  ? 'bullish'
              : item.bullish === false ? 'bearish'
              : item.sentiment === '警戒' ? 'alert' : 'neutral';
    return `<div class="fractal-item">
      <span class="fractal-flag">${item.flag}</span>
      <span class="fractal-label">${item.key}</span>
      <span class="fractal-colon">:</span>
      <span class="fractal-icon">${item.icon}</span>
      <span class="fractal-sentiment ${cls}">${item.sentiment}</span>
    </div>`;
  }).join('');
}

/* ── Main load ─────────────────────────────────────────────────────────────── */
async function loadAll() {
  setLoading(true);
  try {
    const res  = await fetch(`/api/analysis/${encodeURIComponent(currentSymbol)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    document.getElementById('headerBadge').textContent =
      data.name + '  ' + fmtPrice(data.technical?.price);

    renderFractalFooter(data.total_days, data.matched);
    renderWinRates(data.win_rates);
    renderAction(data.action);
    trajData = data.trajectory || [];
    renderTrajectoryCanvas(trajData, currentTrajWindow);
    renderMetrics(data.technical);
  } catch (e) {
    showError('データ取得失敗: ' + e.message);
  } finally {
    setLoading(false);
  }
}

/* ── フラクタルフッター ────────────────────────────────────────────────────── */
function renderFractalFooter(totalDays, matched) {
  const el = document.getElementById('fractalFooter');
  if (!el) return;
  el.innerHTML = `過去 <span class="count-highlight">${totalDays.toLocaleString()}</span> 営業日から完全合致する <span class="count-highlight">${matched}</span> 日を抽出`;
}

/* ── 統計勝率予測 ──────────────────────────────────────────────────────────── */
function renderWinRates(winRates) {
  const el = document.getElementById('winRateRows');
  if (!el) return;
  let html = '';
  winRates.forEach((row, i) => {
    if (i === 2) html += `<hr class="winrate-divider">`;
    html += `<div class="winrate-row">
      <div class="winrate-meta">${row.label} (${row.method})</div>
      <div class="winrate-labels">
        <span class="winrate-buy">買: ${row.buy_pct}%</span>
        <span class="winrate-sell">売: ${row.sell_pct}%</span>
      </div>
      <div class="winrate-bar-track">
        <div class="winrate-bar-fill" style="width:${row.buy_pct}%"></div>
      </div>
    </div>`;
  });
  el.innerHTML = html;
}

/* ── 推奨アクション ────────────────────────────────────────────────────────── */
function renderAction(action) {
  if (!action?.label) return;
  const isLong = action.is_long;
  const cls    = isLong ? 'long' : 'short';

  const at = document.getElementById('actionText');
  if (at) { at.className = `action-main-text ${cls}`; at.innerHTML = `${action.label}<br>${action.direction}`; }

  const wb = document.getElementById('actionWinBadge');
  if (wb) wb.innerHTML = `歴史的勝率: <span class="pct">${action.win_pct}%</span><br>発生エッジ: <span class="edge-val">${action.edge >= 0 ? '+' : ''}${action.edge}%</span>`;

  const tp = document.getElementById('targetPrice');
  if (tp) {
    const moveSign = isLong ? '-' : '+';
    tp.innerHTML = `
      <div class="action-price">${fmtPrice(action.target_price)}<span class="action-price-unit">円</span></div>
      <div class="action-sub">歴史的平均: <span class="action-sub-val">${moveSign}${Math.abs(action.hist_avg_move).toFixed(2)}円</span></div>
      <div class="action-sub">最大到達値: <span class="action-sub-val">${fmtPrice(action.max_reached)}</span></div>`;
  }

  const sp = document.getElementById('stopPrice');
  if (sp) {
    sp.innerHTML = `
      <div class="action-price">${fmtPrice(action.stop_price)}<span class="action-price-unit">円</span></div>
      <div class="action-sub">平均逆行幅: <span class="action-sub-val">+${action.avg_adverse.toFixed(2)}円</span></div>
      <div class="action-sub">最大逆行値: <span class="action-sub-val">${fmtPrice(action.max_adverse)}</span></div>`;
  }
}

/* ── 未来軌道チャート (純粋 Canvas 2D) ─────────────────────────────────────── */
function renderTrajectoryCanvas(traj, window) {
  const canvas = document.getElementById('trajectoryChart');
  if (!canvas || !traj?.length) return;

  // Determine how many points to show
  const limits  = { '全体': traj.length, '24h': 2, '5d': 5, '1m': 21 };
  const count   = Math.min(limits[window] ?? traj.length, traj.length);
  const data    = traj.slice(0, count);

  const W  = canvas.offsetWidth  || 800;
  const H  = canvas.offsetHeight || 280;
  canvas.width  = W;
  canvas.height = H;

  const ctx   = canvas.getContext('2d');
  const PAD   = { top: 20, right: 20, bottom: 36, left: 72 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  const prices  = data.map(p => p.price);
  const minP    = Math.min(...prices);
  const maxP    = Math.max(...prices);
  const rangeP  = maxP - minP || 1;

  const xOf = i  => PAD.left + (i / (data.length - 1)) * plotW;
  const yOf = p  => PAD.top  + plotH - ((p - minP) / rangeP) * plotH;

  // ── Grid ─────────────────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(36,53,82,0.9)';
  ctx.lineWidth   = 1;
  const gridLines = 5;
  for (let i = 0; i <= gridLines; i++) {
    const y = PAD.top + (i / gridLines) * plotH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();

    const val = maxP - (i / gridLines) * rangeP;
    ctx.fillStyle  = '#7a9cc8';
    ctx.font       = '11px monospace';
    ctx.textAlign  = 'right';
    ctx.fillText(fmtPrice(val), PAD.left - 6, y + 4);
  }

  // ── Gradient fill ────────────────────────────────────────────────────────
  if (data.length > 1) {
    const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + plotH);
    grad.addColorStop(0,   'rgba(0,212,255,0.30)');
    grad.addColorStop(1,   'rgba(0,212,255,0.00)');

    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(prices[0]));
    for (let i = 1; i < data.length; i++) {
      const x0 = xOf(i - 1), y0 = yOf(prices[i - 1]);
      const x1 = xOf(i),     y1 = yOf(prices[i]);
      const mx = (x0 + x1) / 2;
      ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1);
    }
    ctx.lineTo(xOf(data.length - 1), PAD.top + plotH);
    ctx.lineTo(xOf(0), PAD.top + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // ── Line ─────────────────────────────────────────────────────────────────
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = 'round';
  ctx.shadowColor  = 'rgba(0,212,255,0.4)';
  ctx.shadowBlur   = 6;
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(prices[0]));
  for (let i = 1; i < data.length; i++) {
    const x0 = xOf(i - 1), y0 = yOf(prices[i - 1]);
    const x1 = xOf(i),     y1 = yOf(prices[i]);
    const mx = (x0 + x1) / 2;
    ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── Points ───────────────────────────────────────────────────────────────
  const currentP = traj[0]?.price;
  data.forEach((pt, i) => {
    const x = xOf(i), y = yOf(pt.price);
    const color = pt.above_current ? '#00e676' : '#ff3d3d';

    ctx.beginPath();
    ctx.arc(x, y, i === 0 ? 5 : 4, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1.5;
    ctx.fill();
    ctx.stroke();
  });

  // ── X-axis labels ─────────────────────────────────────────────────────────
  ctx.fillStyle  = '#7a9cc8';
  ctx.font       = '11px monospace';
  ctx.textAlign  = 'center';
  data.forEach((pt, i) => {
    const label = pt.day === 0 ? '現在' : `${pt.day}日後`;
    ctx.fillText(label, xOf(i), H - 8);
  });
}

/* ── テクニカル指標チップ ──────────────────────────────────────────────────── */
function renderMetrics(tech) {
  if (!tech) return;
  const strip = document.getElementById('metricsStrip');
  if (!strip) return;

  const rsiCls    = tech.rsi < 30 ? 'green' : tech.rsi > 70 ? 'red' : '';
  const signalCls = tech.signal === 'BUY' ? 'green' : tech.signal === 'SELL' ? 'red' : 'orange';
  const mddCls    = tech.max_drawdown > 0.08 ? 'red' : 'green';

  const chips = [
    { label: 'シグナル', val: tech.signal || 'N/A',                       cls: signalCls },
    { label: 'スコア',   val: tech.signal_score ?? 'N/A',                  cls: '' },
    { label: 'RSI',      val: fmt(tech.rsi, 1),                            cls: rsiCls },
    { label: 'Sharpe',   val: fmt(tech.sharpe, 2),                         cls: '' },
    { label: 'PF',       val: fmt(tech.profit_factor, 2),                  cls: '' },
    { label: 'MDD',      val: fmtPct(tech.max_drawdown, false),            cls: mddCls },
    { label: 'VaR 95%',  val: fmtPct(tech.var_95_daily, false),            cls: '' },
    { label: 'Kelly',    val: fmtPct(tech.kelly_frac, false),              cls: 'cyan' },
    { label: 'Edge',     val: fmt(tech.edge, 4),
      cls: (tech.edge || 0) >= 0.04 ? 'green' : '' },
    { label: 'δ',        val: fmt(tech.mispricing_score, 3),               cls: '' },
    { label: 'MA',       val: tech.golden_cross || '―',
      cls: tech.golden_cross === 'GOLDEN' ? 'green' : tech.golden_cross === 'DEATH' ? 'red' : '' },
    { label: 'MACD',
      val: (tech.macd_hist ?? 0) >= 0 ? '▲ 上昇' : '▼ 下降',
      cls: (tech.macd_hist ?? 0) >= 0 ? 'green' : 'red' },
  ];

  strip.innerHTML = chips.map(c => `<div class="metric-chip">
    <span class="mc-label">${c.label}</span>
    <span class="mc-val ${c.cls}">${c.val}</span>
  </div>`).join('');
}
