import argparse
import math
import numpy as np
from joblib import load

def poisson_pmf(k: int, lam: float) -> float:
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def goal_dist(lam: float, max_goals: int = 8) -> np.ndarray:
    probs = np.array([poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    probs = probs / probs.sum()  # normaliza por truncamiento
    return probs

def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p

def implied_prob(odds: float) -> float:
    return 0.0 if odds <= 0 else 1.0 / odds

def value_edge(p_model: float, odds_book: float) -> float:
    """Edge = p_model - implied_prob(odds). Positivo => value."""
    return p_model - implied_prob(odds_book)

def expected_value(p_model: float, odds_book: float) -> float:
    """EV por 1 unidad apostada: EV = p*(odds-1) - (1-p)*1"""
    return p_model * (odds_book - 1.0) - (1.0 - p_model)

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_odds(x: float) -> str:
    if math.isinf(x):
        return "∞"
    return f"{x:.2f}"

def safe_get_team_idx(team_to_idx: dict, team: str):
    if team not in team_to_idx:
        # mensaje útil para el usuario
        sample = sorted(team_to_idx.keys())[:30]
        raise KeyError(
            f"Equipo '{team}' no existe en el modelo/dataset.\n"
            f"Ejemplos de nombres válidos: {sample}"
        )
    return team_to_idx[team]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, help="Equipo local (igual que en el CSV)")
    parser.add_argument("--away", required=True, help="Equipo visitante (igual que en el CSV)")
    parser.add_argument("--max_goals", type=int, default=8, help="Máximo de goles (0..N)")
    parser.add_argument("--models_dir", default="outputs/models", help="Carpeta de modelos")

    # Cuotas opcionales de la casa para comparar (decimal)
    parser.add_argument("--odds_home", type=float, default=None, help="Cuota casa: Local (1)")
    parser.add_argument("--odds_draw", type=float, default=None, help="Cuota casa: Empate (X)")
    parser.add_argument("--odds_away", type=float, default=None, help="Cuota casa: Visita (2)")
    parser.add_argument("--odds_over25", type=float, default=None, help="Cuota casa: Over 2.5")
    parser.add_argument("--odds_under25", type=float, default=None, help="Cuota casa: Under 2.5")
    parser.add_argument("--odds_btts_yes", type=float, default=None, help="Cuota casa: BTTS Sí")
    parser.add_argument("--odds_btts_no", type=float, default=None, help="Cuota casa: BTTS No")
    args = parser.parse_args()

    # 🔥 MODELO MEJORADO (B)
    home_pack = load(f"{args.models_dir}/home_model.joblib")
    away_pack = load(f"{args.models_dir}/away_model.joblib")

    # Se asume que ambos packs comparten el mismo team_to_idx
    team_to_idx = home_pack["team_to_idx"]

    h = safe_get_team_idx(team_to_idx, args.home)
    a = safe_get_team_idx(team_to_idx, args.away)

    # Features
    # Home model: [home_attack(home), away_defense(away), is_home=1]
    ha = home_pack["ha"]
    ad = home_pack["ad"]
    X_home = np.array([[ha[h], ad[a], 1.0]], dtype=float)

    # Away model: [away_attack(away), home_defense(home), is_home=0]
    aa = away_pack["aa"]
    hd = away_pack["hd"]
    X_away = np.array([[aa[a], hd[h], 0.0]], dtype=float)

    lam_home = float(home_pack["model"].predict(X_home)[0])
    lam_away = float(away_pack["model"].predict(X_away)[0])
    lam_home = max(lam_home, 1e-6)
    lam_away = max(lam_away, 1e-6)

    max_g = args.max_goals
    p_home_g = goal_dist(lam_home, max_g)
    p_away_g = goal_dist(lam_away, max_g)
    score_mat = np.outer(p_home_g, p_away_g)

    # 1X2
    p_home_win = float(np.tril(score_mat, -1).sum())
    p_draw = float(np.trace(score_mat))
    p_away_win = float(np.triu(score_mat, 1).sum())

    # Over/Under 2.5
    totals = np.add.outer(np.arange(max_g + 1), np.arange(max_g + 1))
    p_over25 = float(score_mat[totals >= 3].sum())
    p_under25 = 1.0 - p_over25

    # BTTS
    p_btts_yes = float(score_mat[1:, 1:].sum())
    p_btts_no = 1.0 - p_btts_yes

    # Top marcadores
    flat = score_mat.flatten()
    top_idx = flat.argsort()[::-1][:10]
    top_scores = []
    for idx in top_idx:
        hg = idx // (max_g + 1)
        ag = idx % (max_g + 1)
        top_scores.append((hg, ag, float(score_mat[hg, ag])))

    print(f"\n⚽ {args.home} vs {args.away}")
    print(f"λ local  = {lam_home:.3f} | λ visita = {lam_away:.3f}\n")

    markets = [
        ("1 (Local)", p_home_win, args.odds_home),
        ("X (Empate)", p_draw, args.odds_draw),
        ("2 (Visita)", p_away_win, args.odds_away),
        ("Over 2.5", p_over25, args.odds_over25),
        ("Under 2.5", p_under25, args.odds_under25),
        ("BTTS Sí", p_btts_yes, args.odds_btts_yes),
        ("BTTS No", p_btts_no, args.odds_btts_no),
    ]

    print("📊 Mercados (Probabilidad → Cuota justa) + Value si das cuota de casa")
    print("-" * 78)
    header = f"{'Mercado':<14} {'P(model)':>10} {'Cuota justa':>12} {'Cuota casa':>11} {'Edge':>10} {'EV':>10}"
    print(header)
    print("-" * 78)

    for name, p, odds_book in markets:
        fo = fair_odds(p)
        if odds_book is None:
            print(f"{name:<14} {fmt_pct(p):>10} {fmt_odds(fo):>12} {'-':>11} {'-':>10} {'-':>10}")
        else:
            edge = value_edge(p, odds_book)
            ev = expected_value(p, odds_book)
            print(f"{name:<14} {fmt_pct(p):>10} {fmt_odds(fo):>12} {odds_book:>11.2f} {fmt_pct(edge):>10} {ev:>10.3f}")

    print("\n🏁 Top 10 marcadores exactos más probables:")
    for hg, ag, p in top_scores:
        print(f" - {args.home} {hg}-{ag} {args.away}: {fmt_pct(p)}")

    given = [(n, p, o) for (n, p, o) in markets if o is not None]
    if given:
        best = sorted(given, key=lambda x: expected_value(x[1], x[2]), reverse=True)
        print("\n💡 Mejores oportunidades (según EV, mayor es mejor):")
        for n, p, o in best[:3]:
            ev = expected_value(p, o)
            edge = value_edge(p, o)
            print(f" - {n}: EV={ev:.3f} | Edge={fmt_pct(edge)} | P={fmt_pct(p)} | Cuota={o:.2f}")

if __name__ == "__main__":
    main()
