import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/matches.csv"

# Columnas que necesitamos para el primer modelo
NEEDED = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]

def main():
    os.makedirs("data/processed", exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No encontré CSV en {RAW_DIR}. Coloca tus temporadas ahí.")

    print("📥 Archivos encontrados:")
    for f in csv_files:
        print(" -", f)

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)

        # Validación de columnas
        missing = [c for c in NEEDED if c not in df.columns]
        if missing:
            print(f"⚠️ Saltando {path} (faltan columnas: {missing})")
            continue

        df = df[NEEDED].copy()

        # Limpieza básica
        df = df.dropna()
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna()

        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        # filtros razonables
        df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)]
        df = df[(df["FTHG"] <= 15) & (df["FTAG"] <= 15)]

        dfs.append(df)

        print(f"✅ {os.path.basename(path)} -> {len(df):,} filas")

    if not dfs:
        raise ValueError("No se pudo procesar ningún CSV. Revisa columnas HomeTeam/AwayTeam/FTHG/FTAG.")

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\n📦 Listo. Total filas: {len(out):,}")
    print("🧾 Guardado en:", OUT_PATH)

if __name__ == "__main__":
    main()
