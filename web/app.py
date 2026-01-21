from flask import Flask, render_template, request, jsonify
import numpy as np
import math
import os
import time
import csv
from joblib import load

app = Flask(__name__)

# Carga de modelos mejorados (B)
HOME_PACK = load("outputs/models/home_model.joblib")
AWAY_PACK = load("outputs/models/away_model.joblib")
TEAM_TO_IDX = HOME_PACK["team_to_idx"]

def poisson_pmf(k: int, lam: float) -> float:
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def goal_dist(lam: float, max_goals: int = 8) -> np.ndarray:
    probs = np.array([poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    return probs / probs.sum()

def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p

def implied_prob(odds: float) -> float:
    return 0.0 if odds is None or odds <= 0 else 1.0 / odds

def edge(p_model: float, odds_book: float) -> float:
    return p_model - implied_prob(odds_book)

def ev(p_model: float, odds_book: float) -> float:
    return p_model * (odds_book - 1.0) - (1.0 - p_model)

def predict(home: str, away: str, max_goals: int = 8):
    if home not in TEAM_TO_IDX or away not in TEAM_TO_IDX:
        return None

    h = TEAM_TO_IDX[home]
    a = TEAM_TO_IDX[away]

    ha = HOME_PACK["ha"]
    ad = HOME_PACK["ad"]
    aa = AWAY_PACK["aa"]
    hd = AWAY_PACK["hd"]

    X_home = np.array([[ha[h], ad[a], 1.0]], dtype=float)
    X_away = np.array([[aa[a], hd[h], 0.0]], dtype=float)

    lam_home = float(HOME_PACK["model"].predict(X_home)[0])
    lam_away = float(AWAY_PACK["model"].predict(X_away)[0])

    lam_home = max(lam_home, 1e-6)
    lam_away = max(lam_away, 1e-6)

    p_home_g = goal_dist(lam_home, max_goals)
    p_away_g = goal_dist(lam_away, max_goals)
    score_mat = np.outer(p_home_g, p_away_g)

    # 1X2
    p_home = float(np.tril(score_mat, -1).sum())
    p_draw = float(np.trace(score_mat))
    p_away = float(np.triu(score_mat, 1).sum())

    # Over/Under 2.5
    totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    p_over25 = float(score_mat[totals >= 3].sum())
    p_under25 = 1.0 - p_over25

    # BTTS
    p_btts_yes = float(score_mat[1:, 1:].sum())
    p_btts_no = 1.0 - p_btts_yes

    # Top 10 scores
    flat = score_mat.flatten()
    top_idx = flat.argsort()[::-1][:10]
    top_scores = []
    for idx in top_idx:
        hg = idx // (max_goals + 1)
        ag = idx % (max_goals + 1)
        top_scores.append({"hg": int(hg), "ag": int(ag), "p": float(score_mat[hg, ag])})

    return {
        "lam_home": lam_home,
        "lam_away": lam_away,
        "markets": {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
            "over25": p_over25,
            "under25": p_under25,
            "btts_yes": p_btts_yes,
            "btts_no": p_btts_no,
        },
        "top_scores": top_scores
    }

@app.get("/")
def index():
    teams = sorted(TEAM_TO_IDX.keys())
    return render_template("index.html", teams=teams)

@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    home = data.get("home")
    away = data.get("away")
    max_goals = int(data.get("max_goals", 8))
    odds = data.get("odds", {})  # cuotas opcionales

    res = predict(home, away, max_goals)
    if res is None:
        return jsonify({"error": "Equipo no encontrado. Revisa nombres."}), 400

    out = {
        "home": home,
        "away": away,
        "lam_home": res["lam_home"],
        "lam_away": res["lam_away"],
        "rows": [],
        "top_scores": res["top_scores"],
        "top_picks": []
    }

    mapping = [
        ("1 (Local)", "home", odds.get("home")),
        ("X (Empate)", "draw", odds.get("draw")),
        ("2 (Visita)", "away", odds.get("away")),
        ("Over 2.5", "over25", odds.get("over25")),
        ("Under 2.5", "under25", odds.get("under25")),
        ("BTTS Sí", "btts_yes", odds.get("btts_yes")),
        ("BTTS No", "btts_no", odds.get("btts_no")),
    ]

    for label, key, o in mapping:
        p = float(res["markets"][key])
        fo = fair_odds(p)

        if o is None or str(o).strip() == "":
            o_val = None
        else:
            try:
                o_val = float(o)
            except ValueError:
                o_val = None

        row = {
            "market": label,
            "p_model": p,
            "fair_odds": fo,
            "odds_book": o_val,
            "edge": None,
            "ev": None
        }

        if o_val is not None and o_val > 0:
            row["edge"] = edge(p, o_val)
            row["ev"] = ev(p, o_val)

        out["rows"].append(row)

    # ✅ Top picks: EV > 0 y Edge > 2%
    top_picks = []
    for r in out["rows"]:
        if r["odds_book"] is None:
            continue
        if r["ev"] is not None and r["edge"] is not None:
            if r["ev"] > 0 and r["edge"] > 0.02:
                top_picks.append(r)

    top_picks = sorted(top_picks, key=lambda x: x["ev"], reverse=True)
    out["top_picks"] = top_picks[:5]

    return jsonify(out)

@app.post("/api/export_csv")
def api_export_csv():
    data = request.get_json(force=True)
    rows = data.get("rows", [])
    if not rows:
        return jsonify({"error": "No hay datos para exportar"}), 400

    os.makedirs("outputs/reports", exist_ok=True)
    filename = f"outputs/reports/value_bets_{int(time.time())}.csv"

    keys = ["market", "p_model", "fair_odds", "odds_book", "edge", "ev"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    return jsonify({"ok": True, "file": filename})

if __name__ == "__main__":
    app.run(debug=True)
