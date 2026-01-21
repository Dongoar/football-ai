import argparse
import math
import numpy as np
import pandas as pd
from joblib import load

def poisson_pmf(k: int, lam: float) -> float:
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def goal_dist(lam: float, max_goals: int = 8) -> np.ndarray:
    probs = np.array([poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    return probs / probs.sum()

def implied_prob(odds: float) -> float:
    return 0.0 if odds is None or odds <= 0 else 1.0 / odds

def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p

def expected_value(p: float, odds: float) -> float:
    if odds is None:
        return float("nan")
    return p * (odds - 1.0) - (1.0 - p)

def value_edge(p: float, odds: float) -> float:
    if odds is None:
        return float("nan")
    return p - implied_prob(odds)

def predict_markets(home_pack, away_pack, home: str, away: str, max_goals: int = 8):
    team_to_idx = home_pack["team_to_idx"]
    if home not in team_to_idx or away not in team_to_idx:
        return None

    h = team_to_idx[home]
    a = team_to_idx[away]

    ha = home_pack["ha"]
    ad = home_pack["ad"]
    aa = away_pack["aa"]
    hd = away_pack["hd"]

    X_home = np.array([[ha[h], ad[a], 1.0]], dtype=float)
    X_away = np.array([[aa[a], hd[h], 0.0]], dtype=float)

    lam_home = float(home_pack["model"].predict(X_home)[0])
    lam_away = float(away_pack["model"].predict(X_away)[0])
    lam_home = max(lam_home, 1e-6)
    lam_away = max(lam_away, 1e-6)

    p_home_g = goal_dist(lam_home, max_goals)
    p_away_g = goal_dist(lam_away, max_goals)
    score_mat = np.outer(p_home_g, p_away_g)

    p_home_win = float(np.tril(score_mat, -1).sum())
    p_draw = float(np.trace(score_mat))
    p_away_win = float(np.triu(score_mat, 1).sum())

    totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    p_over25 = float(score_mat[totals >= 3].sum())
    p_under25 = 1.0 - p_over25

    p_btts_yes = float(score_mat[1:, 1:].sum())
    p_btts_no = 1.0 - p_btts_yes

    return {
        "home": home,
        "away": away,
        "lam_home": lam_home,
        "lam_away": lam_away,
        "p_home": p_home_win,
        "p_draw": p_draw,
        "p_away": p_away_win,
        "p_over25": p_over25,
        "p_under25": p_under25,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/matches.csv", help="Dataset procesado")
    parser.add_argument("--models_dir", default="outputs/models", help="Carpeta modelos mejorados")
    parser.add_argument("--max_goals", type=int, default=8)
    parser.add_argument("--last_n", type=int, default=30, help="Tomar últimos N partidos del CSV como demo")
    parser.add_argument("--odds_csv", default=None, help="CSV opcional con cuotas para comparar (ver formato abajo)")
    parser.add_argument("--out", default="outputs/reports/value_bets.csv", help="Salida CSV")
    args = parser.parse_args()

    home_pack = load(f"{args.models_dir}/home_model.joblib")
    away_pack = load(f"{args.models_dir}/away_model.joblib")

    df = pd.read_csv(args.csv)

    # Demo: últimos N partidos (sirve para probar la automatización)
    df_demo = df.tail(args.last_n).copy()

    # Cuotas opcionales
    odds = None
    if args.odds_csv:
        odds = pd.read_csv(args.odds_csv)
        # formato esperado:
        # home,away,odds_home,odds_draw,odds_away,odds_over25,odds_under25,odds_btts_yes,odds_btts_no
        odds = odds.set_index(["home", "away"])

    rows = []
    for _, r in df_demo.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]
        pred = predict_markets(home_pack, away_pack, home, away, args.max_goals)
        if pred is None:
            continue

        # cuotas casa (si hay)
        if odds is not None and (home, away) in odds.index:
            o = odds.loc[(home, away)]
            odds_home = float(o.get("odds_home", np.nan))
            odds_draw = float(o.get("odds_draw", np.nan))
            odds_away = float(o.get("odds_away", np.nan))
            odds_over25 = float(o.get("odds_over25", np.nan))
            odds_under25 = float(o.get("odds_under25", np.nan))
            odds_btts_yes = float(o.get("odds_btts_yes", np.nan))
            odds_btts_no = float(o.get("odds_btts_no", np.nan))
        else:
            odds_home = odds_draw = odds_away = None
            odds_over25 = odds_under25 = None
            odds_btts_yes = odds_btts_no = None

        # arma filas por mercado
        markets = [
            ("1", pred["p_home"], odds_home),
            ("X", pred["p_draw"], odds_draw),
            ("2", pred["p_away"], odds_away),
            ("Over2.5", pred["p_over25"], odds_over25),
            ("Under2.5", pred["p_under25"], odds_under25),
            ("BTTS_Yes", pred["p_btts_yes"], odds_btts_yes),
            ("BTTS_No", pred["p_btts_no"], odds_btts_no),
        ]

        for m, p, ob in markets:
            rows.append({
                "home": home,
                "away": away,
                "market": m,
                "p_model": p,
                "fair_odds": fair_odds(p),
                "odds_book": ob,
                "edge": value_edge(p, ob),
                "ev": expected_value(p, ob),
                "lam_home": pred["lam_home"],
                "lam_away": pred["lam_away"],
            })

    out_df = pd.DataFrame(rows)

    # Ranking: si no hay cuotas, ordena por probabilidad; si hay, por EV
    if out_df["odds_book"].notna().any():
        out_df = out_df.sort_values(["ev"], ascending=False)
    else:
        out_df = out_df.sort_values(["p_model"], ascending=False)

    os.makedirs("outputs/reports", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"✅ Listo. Exportado: {args.out}")
    print(out_df.head(20).to_string(index=False))

if __name__ == "__main__":
    import os
    main()
