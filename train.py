import os
import argparse
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.linear_model import PoissonRegressor

def ensure_dirs():
    os.makedirs("outputs/models", exist_ok=True)

def prepare_features(df: pd.DataFrame):
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}

    n = len(teams)
    home_attack = np.zeros(n)
    home_defense = np.zeros(n)
    away_attack = np.zeros(n)
    away_defense = np.zeros(n)

    home_games = np.zeros(n)
    away_games = np.zeros(n)

    for _, r in df.iterrows():
        h = team_to_idx[r["HomeTeam"]]
        a = team_to_idx[r["AwayTeam"]]

        home_attack[h] += r["FTHG"]
        home_defense[h] += r["FTAG"]
        away_attack[a] += r["FTAG"]
        away_defense[a] += r["FTHG"]

        home_games[h] += 1
        away_games[a] += 1

    # promedios
    home_attack /= np.maximum(home_games, 1)
    home_defense /= np.maximum(home_games, 1)
    away_attack /= np.maximum(away_games, 1)
    away_defense /= np.maximum(away_games, 1)

    return team_to_idx, home_attack, home_defense, away_attack, away_defense

def build_matrix(df, team_to_idx, ha, hd, aa, ad, target="home"):
    X = []
    y = []

    for _, r in df.iterrows():
        h = team_to_idx[r["HomeTeam"]]
        a = team_to_idx[r["AwayTeam"]]

        if target == "home":
            X.append([ha[h], ad[a], 1.0])
            y.append(r["FTHG"])
        else:
            X.append([aa[a], hd[h], 0.0])
            y.append(r["FTAG"])

    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/matches.csv")
    args = parser.parse_args()

    ensure_dirs()

    df = pd.read_csv(args.csv)
    print(f"📊 Partidos: {len(df)}")

    team_to_idx, ha, hd, aa, ad = prepare_features(df)

    Xh, yh = build_matrix(df, team_to_idx, ha, hd, aa, ad, "home")
    Xa, ya = build_matrix(df, team_to_idx, ha, hd, aa, ad, "away")

    home_model = PoissonRegressor(alpha=0.001, max_iter=3000)
    away_model = PoissonRegressor(alpha=0.001, max_iter=3000)

    home_model.fit(Xh, yh)
    away_model.fit(Xa, ya)

    dump({
        "model": home_model,
        "team_to_idx": team_to_idx,
        "ha": ha,
        "ad": ad
    }, "outputs/models/home_model.joblib")

    dump({
        "model": away_model,
        "team_to_idx": team_to_idx,
        "aa": aa,
        "hd": hd
    }, "outputs/models/away_model.joblib")

    print("✅ Modelos mejorados entrenados y guardados")

if __name__ == "__main__":
    main()
