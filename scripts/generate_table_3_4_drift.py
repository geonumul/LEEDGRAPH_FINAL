"""Regenerate Table_3_4_drift.csv.

Computes per-version drift between the original achievement ratio and the v5
standardized achievement ratio using the precomputed parquet feature table.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "project_features.parquet"
FALLBACK_PATH = ROOT / "data" / "project_features_option_a.parquet"
OUT_PATH = ROOT / "results" / "tables" / "Table_3_4_drift.csv"

VERSION_ORDER = ["v2.0", "v2.2", "v2009", "v3", "v4", "v4.1", "v5", "unknown"]


def load_features() -> pd.DataFrame:
    path = INPUT_PATH if INPUT_PATH.exists() else FALLBACK_PATH
    df = pd.read_parquet(path)
    print(f"Loaded: {path} (N={len(df)})")
    return df


def compute_drift_pp(df: pd.DataFrame) -> pd.Series:
    if "drift" in df.columns and df["drift"].notna().all():
        # `drift` in parquet is stored as a ratio (0–1); convert to %p.
        return df["drift"].astype(float).abs() * 100.0
    return (
        df["achievement_ratio_original"].astype(float)
        - df["achievement_ratio_v5"].astype(float)
    ).abs() * 100.0


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def row_for(label: str, sub: pd.DataFrame) -> dict:
        return {
            "인증 버전": label,
            "표본 수": len(sub),
            "원본 달성률 평균": f"{sub['achievement_ratio_original'].mean() * 100:.2f}%",
            "v5 달성률 평균": f"{sub['achievement_ratio_v5'].mean() * 100:.2f}%",
            "평균 drift": f"{sub['drift_pp'].mean():.2f}%p",
            "최대 drift": f"{sub['drift_pp'].max():.2f}%p",
            "20%p 초과 건수": int((sub["drift_pp"] > 20.0).sum()),
        }

    versions_present = [v for v in VERSION_ORDER if v in df["original_version"].unique()]
    extras = sorted(set(df["original_version"].unique()) - set(versions_present))
    for ver in versions_present + extras:
        sub = df[df["original_version"] == ver]
        if len(sub) == 0:
            continue
        rows.append(row_for(ver, sub))

    rows.append(row_for("전체", df))
    return pd.DataFrame(rows)


def main() -> None:
    df = load_features()
    df = df.copy()
    df["drift_pp"] = compute_drift_pp(df)

    summary = aggregate(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
