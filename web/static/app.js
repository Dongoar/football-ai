let lastRows = [];

function val(id){
  const el = document.getElementById(id);
  const v = el.value.trim();
  if(v === "") return null;
  const n = Number(v);
  // Si es 0 o negativo, lo tratamos como inválido (no existe cuota 0)
  if(!Number.isFinite(n) || n <= 1.000) return n; // permito 1.01+ realista, pero si pones 1 exacto también lo deja
  return n;
}

function pct(x){ return (x*100).toFixed(2) + "%"; }
function num(x){ return (x === null || x === undefined) ? "-" : x.toFixed(2); }

async function run(){
  const home = document.getElementById("home").value;
  const away = document.getElementById("away").value;
  const max_goals = Number(document.getElementById("max_goals").value || 8);

  const payload = {
    home, away, max_goals,
    odds: {
      home: val("odds_home"),
      draw: val("odds_draw"),
      away: val("odds_away"),
      over25: val("odds_over25"),
      under25: val("odds_under25"),
      btts_yes: val("odds_btts_yes"),
      btts_no: val("odds_btts_no"),
    }
  };

  const err = document.getElementById("error");
  err.classList.add("hidden");
  err.textContent = "";

  const res = await fetch("/api/predict", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(payload)
  });

  if(!res.ok){
    const data = await res.json().catch(()=>({error:"Error"}));
    err.textContent = data.error || "Error";
    err.classList.remove("hidden");
    return;
  }

  const data = await res.json();

  // DEBUG: mira en consola qué llega
  console.log("API /predict response:", data);

  document.getElementById("meta").textContent =
    `⚽ ${data.home} vs ${data.away} | λ local=${data.lam_home.toFixed(3)} | λ visita=${data.lam_away.toFixed(3)}`;

  // tabla
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  (data.rows || []).forEach(r=>{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.market}</td>
      <td>${pct(r.p_model)}</td>
      <td>${num(r.fair_odds)}</td>
      <td>${r.odds_book == null ? "-" : num(r.odds_book)}</td>
      <td>${r.edge == null ? "-" : pct(r.edge)}</td>
      <td>${r.ev == null ? "-" : r.ev.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });

  // top picks
  const picks = document.getElementById("picks");
  picks.innerHTML = "";
  (data.top_picks || []).forEach(p=>{
    const div = document.createElement("div");
    div.className = "pick";
    div.innerHTML = `
      <div class="t">${p.market}</div>
      <div class="s">P=${pct(p.p_model)} · Cuota justa=${p.fair_odds.toFixed(2)} · Casa=${p.odds_book.toFixed(2)}</div>
      <div class="s">Edge=${pct(p.edge)} · EV=${p.ev.toFixed(3)}</div>
    `;
    picks.appendChild(div);
  });

  // ✅ top scores robusto
  const scores = document.getElementById("scores");
  scores.innerHTML = "";

  const list = (data.top_scores && Array.isArray(data.top_scores)) ? data.top_scores : [];
  if(list.length === 0){
    const div = document.createElement("div");
    div.className = "pill";
    div.textContent = "No llegaron top_scores desde el backend. Revisa consola (F12) para ver el JSON.";
    scores.appendChild(div);
  } else {
    list.forEach(s=>{
      const div = document.createElement("div");
      div.className = "pill";
      div.textContent = `${data.home} ${s.hg}-${s.ag} ${data.away} · ${pct(s.p)}`;
      scores.appendChild(div);
    });
  }

  // guardar para export
  lastRows = data.rows || [];
}

async function exportCSV(){
  if(!lastRows.length){
    alert("Primero calcula un partido.");
    return;
  }

  const res = await fetch("/api/export_csv", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ rows: lastRows })
  });

  const data = await res.json().catch(()=>({}));
  if(!res.ok){
    alert(data.error || "Error exportando");
    return;
  }

  alert("Exportado en: " + data.file);
}

document.getElementById("btn").addEventListener("click", run);
document.getElementById("exportBtn").addEventListener("click", exportCSV);
