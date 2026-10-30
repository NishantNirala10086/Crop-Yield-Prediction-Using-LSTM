const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let opts = null;

async function api(url, body) {
  const r = await fetch(url, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}
const cap = s => s[0].toUpperCase() + s.slice(1);
const fmt = (v, d = 2) => v == null ? "–" : Number(v).toFixed(d);

async function init() {
  try {
    opts = await api("/api/options");
    $("district").innerHTML = opts.districts.map(d => `<option>${d}</option>`).join("");
    $("crop").innerHTML = opts.crops.map(c => `<option value="${c}">${cap(c)}</option>`).join("");
    $("year").innerHTML = [...opts.years].reverse().map(y => `<option>${y}</option>`).join("");
  } catch (e) { setMsg(e.message, true); }
}
function setMsg(t, err = false) { $("msg").textContent = t; $("msg").className = "msg" + (err ? " err" : ""); }

$("rain").oninput = e => $("rainOut").textContent = (e.target.value > 0 ? "+" : "") + e.target.value + "%";
$("temp").oninput = e => $("tempOut").textContent = (e.target.value > 0 ? "+" : "") + Number(e.target.value).toFixed(1) + " °C";

$("form").onsubmit = async ev => {
  ev.preventDefault();
  const btn = ev.target.querySelector("button"); btn.disabled = true; setMsg("Running the model…");
  try {
    const r = await api("/api/predict", {district:$("district").value, crop:$("crop").value, year:$("year").value,
      rain_change_pct:$("rain").value, temp_change_c:$("temp").value});
    render(r); setMsg("");
  } catch (e) { setMsg(e.message, true); }
  btn.disabled = false;
};

function render(r) {
  $("result").hidden = false;
  $("pred").textContent = fmt(r.predicted);
  $("range").textContent = r.low != null ? `likely ${fmt(r.low)} – ${fmt(r.high)} t/ha` : "";
  $("avg5").textContent = fmt(r.avg5) + " t/ha";
  const d = r.vs_avg5_pct;
  $("delta").textContent = d == null ? "–" : (d > 0 ? "+" : "") + d + "%";
  $("delta").className = d == null ? "" : d >= 0 ? "up" : "down";
  const changed = r.scenario.rain_change_pct != 0 || r.scenario.temp_change_c != 0;
  $("actual").textContent = fmt(r.actual) + " t/ha" + (changed ? " (unadjusted)" : "");
  $("prev").textContent = fmt(r.prev_yield) + " t/ha";
  $("histChart").innerHTML = historySVG(r);
  $("wxChart").innerHTML = weatherSVG(r.weather);
}

function historySVG(r) {
  const W = 760, H = 280, m = {l:44, r:16, t:14, b:30};
  const pts = r.history.filter(p => p.year < r.year);
  const all = pts.map(p => p.yield).concat([r.predicted, r.low ?? r.predicted, r.high ?? r.predicted, r.actual ?? r.predicted]);
  const y0 = Math.floor(Math.min(...all) * 2) / 2 - .25, y1 = Math.ceil(Math.max(...all) * 2) / 2 + .25;
  const xs = r.history.map(p => p.year), x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = v => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Yield history with forecast">`;
  for (let v = Math.ceil(y0); v <= y1; v += 1) s += `<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css('--line')}"/><text x="${m.l-8}" y="${Y(v)+4}" text-anchor="end">${v}</text>`;
  for (let v = x0; v <= x1; v += Math.max(1, Math.round((x1 - x0) / 8))) s += `<text x="${X(v)}" y="${H-10}" text-anchor="middle">${v}</text>`;
  if (r.low != null) s += `<rect x="${X(r.year)-9}" y="${Y(r.high)}" width="18" height="${Y(r.low)-Y(r.high)}" fill="${css('--gold')}" opacity=".28"/>`;
  s += `<polyline fill="none" stroke="${css('--blue')}" stroke-width="2" points="${pts.map(p => X(p.year)+","+Y(p.yield)).join(" ")}"/>`;
  if (pts.length) { const l = pts[pts.length-1]; s += `<line x1="${X(l.year)}" y1="${Y(l.yield)}" x2="${X(r.year)}" y2="${Y(r.predicted)}" stroke="${css('--gold')}" stroke-dasharray="4 3" stroke-width="2"/>`; }
  s += pts.map(p => `<circle cx="${X(p.year)}" cy="${Y(p.yield)}" r="2.6" fill="${css('--blue')}"/>`).join("");
  s += `<circle cx="${X(r.year)}" cy="${Y(r.predicted)}" r="6" fill="${css('--gold')}"/>`;
  if (r.actual != null) s += `<circle cx="${X(r.year)}" cy="${Y(r.actual)}" r="5" fill="none" stroke="${css('--ink')}" stroke-width="1.6"/>`;
  s += `<text x="${m.l}" y="12">t/ha</text></svg>`;
  s = s.replace("</svg>", `<g transform="translate(${W-250},${m.t+6})"><circle cx="0" cy="0" r="4" fill="${css('--blue')}"/><text x="8" y="4">recorded</text><circle cx="72" cy="0" r="5" fill="${css('--gold')}"/><text x="82" y="4">forecast</text><circle cx="142" cy="0" r="4.5" fill="none" stroke="${css('--ink')}" stroke-width="1.5"/><text x="151" y="4">recorded that year</text></g></svg>`);
  return s;
}

function weatherSVG(w) {
  const W = 760, H = 220, m = {l:44, r:44, t:14, b:28}, n = w.length, bw = (W - m.l - m.r) / n;
  const rmax = Math.max(50, ...w.map(a => a.rainfall)) * 1.1, tmin = Math.min(...w.map(a => a.tmax)) - 3, tmax = Math.max(...w.map(a => a.tmax)) + 3;
  const R = v => H - m.b - v / rmax * (H - m.t - m.b), T = v => H - m.b - (v - tmin) / (tmax - tmin) * (H - m.t - m.b);
  const mon = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly rainfall and maximum temperature">`;
  s += w.map((a, i) => `<rect x="${m.l+i*bw+5}" y="${R(a.rainfall)}" width="${bw-10}" height="${H-m.b-R(a.rainfall)}" fill="${css('--blue')}" opacity=".75"/><text x="${m.l+i*bw+bw/2}" y="${H-10}" text-anchor="middle">${mon[i]}</text>`).join("");
  s += `<polyline fill="none" stroke="${css('--gold')}" stroke-width="2.2" points="${w.map((a,i)=>(m.l+i*bw+bw/2)+","+T(a.tmax)).join(" ")}"/>`;
  s += `<text x="${m.l-8}" y="${m.t+4}" text-anchor="end">${Math.round(rmax)}</text><text x="${m.l-8}" y="${H-m.b}" text-anchor="end">0</text>`;
  s += `<text x="${W-m.r+8}" y="${m.t+4}">${Math.round(tmax)}°</text><text x="${W-m.r+8}" y="${H-m.b}">${Math.round(tmin)}°</text>`;
  s += `<text x="${m.l}" y="12">rainfall mm (bars) · max temp °C (line)</text></svg>`;
  return s;
}
init();
