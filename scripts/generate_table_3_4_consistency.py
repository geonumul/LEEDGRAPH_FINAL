"""Regenerate Table_3_4_consistency.csv (paper Table 2).

Cross-checks the reported "TOTAL X / Y" parsed from each scorecard PDF against
the sum of per-category awarded values parsed from the same PDF. Both numbers
come from independent regex extractions in loader.py, so a mismatch indicates a
parsing failure or an irregularity in the source PDF.

The category sum is split into:
  - main_sum  : SS + WE + EA + MR + IEQ + LT + IP   (main rating categories)
  - bonus_sum : IN + RP                              (bonus credits)
  - total_sum : main_sum + bonus_sum
Consistency is judged on total_sum vs reported_total, since the LEED scorecard
"TOTAL awarded" line includes Innovation (IN) and Regional Priority (RP) bonus
credits on top of the main categories.
"""

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from src.data.loader import LEEDDataLoader  # noqa: E402


MAIN_COLS = ["SS", "WE", "EA", "MR", "IEQ", "LT", "IP"]
BONUS_COLS = ["IN", "RP"]
CATEGORY_COLS = MAIN_COLS + BONUS_COLS
TOLERANCE = 0.5
VERSION_ORDER = ["v2.0", "v2.2", "v2009", "v3", "v4", "v4.1", "v5", "unknown"]

OUT_PATH = ROOT / "results" / "tables" / "Table_3_4_consistency.csv"
MISMATCH_PATH = ROOT / "results" / "tables" / "Table_3_4_consistency_mismatches.csv"


def parse_all() -> pd.DataFrame:
    loader = LEEDDataLoader(data_dir=str(ROOT / "data"))
    return loader.load_scorecard_batch(str(ROOT / "data" / "scorecards"))


def _col_sum(df: pd.DataFrame, cats: list) -> pd.Series:
    cols = [f"{c}_awarded" for c in cats if f"{c}_awarded" in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].fillna(0).sum(axis=1).astype(float)


def compute_category_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Add main_sum / bonus_sum / total_sum columns to df (in place) and return it."""
    df["main_sum"] = _col_sum(df, MAIN_COLS)
    df["bonus_sum"] = _col_sum(df, BONUS_COLS)
    df["total_sum"] = df["main_sum"] + df["bonus_sum"]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def row_for(label: str, sub: pd.DataFrame) -> dict:
        diff_abs = sub["abs_diff"]
        return {
            "인증 버전": label,
            "표본 수": len(sub),
            "일치 건수": int(sub["match"].sum()),
            "일치율": f"{(sub['match'].mean() * 100):.1f}%",
            "평균 절대 오차(점)": f"{diff_abs.mean():.2f}",
            "최대 절대 오차(점)": f"{diff_abs.max():.2f}",
        }

    versions_present = [v for v in VERSION_ORDER if v in df["version"].unique()]
    extras = sorted(set(df["version"].unique()) - set(versions_present))
    for ver in versions_present + extras:
        sub = df[df["version"] == ver]
        if len(sub) == 0:
            continue
        rows.append(row_for(ver, sub))

    rows.append(row_for("전체", df))
    return pd.DataFrame(rows)


def main() -> None:
    df = parse_all()

    df["reported_total"] = df["total_awarded"].astype(float)
    df = compute_category_sum(df)
    df["diff"] = df["reported_total"] - df["total_sum"]
    df["abs_diff"] = df["diff"].abs()
    df["match"] = df["abs_diff"] <= TOLERANCE

    summary = aggregate(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved summary: {OUT_PATH}")
    print(summary.to_string(index=False))

    mismatches = (
        df.loc[~df["match"]]
        .sort_values("abs_diff", ascending=False)
        [[
            "project_id", "project_name", "version", "reported_total",
            "main_sum", "bonus_sum", "total_sum", "diff", "abs_diff", "file_name",
        ]]
    )

    if len(mismatches) > 0:
        mismatches.to_csv(MISMATCH_PATH, index=False, encoding="utf-8-sig")
        print(f"\nSaved mismatches: {MISMATCH_PATH}  ({len(mismatches)} rows)")
        by_ver = mismatches["version"].value_counts()
        print("Mismatches by version:", by_ver.to_dict())
        print("\nTop 5 mismatches (by abs error):")
        for _, r in mismatches.head(5).iterrows():
            print(
                f"  [{r['version']}] {r['project_id']} | {r['project_name']} "
                f"| reported={r['reported_total']:.0f} "
                f"main={r['main_sum']:.0f} bonus={r['bonus_sum']:.0f} "
                f"total={r['total_sum']:.0f} diff={r['diff']:+.1f}"
            )
    else:
        if MISMATCH_PATH.exists():
            MISMATCH_PATH.unlink()
        print("\nNo mismatches found.")


if __name__ == "__main__":
    main()
