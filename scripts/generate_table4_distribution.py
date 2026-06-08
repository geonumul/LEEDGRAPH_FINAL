"""Regenerate Table4_distribution.csv (paper Table 4).

Combines the certification-grade distribution and the LEED-version distribution
into a single long-form table, replacing the legacy key-value Table1_dataset.csv.
"""

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "project_features.parquet"
TABLES = ROOT / "results" / "tables"
OUT_PATH = TABLES / "Table4_distribution.csv"
LEGACY_DIR = TABLES / "legacy"
LEGACY_SRC = TABLES / "Table1_dataset.csv"

GRADE_ORDER = ["Gold", "Silver", "Platinum", "Certified"]
VERSION_ORDER = ["v4", "v2009", "v4.1", "v2.2", "v2.0"]


def ordered_counts(series: pd.Series, preferred: list) -> list:
    """(label, count) pairs: preferred order first, then any leftovers by count desc."""
    counts = series.value_counts()
    pairs = [(v, int(counts[v])) for v in preferred if v in counts.index]
    leftovers = [(v, int(c)) for v, c in counts.items() if v not in preferred]
    leftovers.sort(key=lambda x: x[1], reverse=True)
    return pairs + leftovers


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    n = len(df)
    print(f"Loaded {FEATURES_PATH.name}: {n} rows")

    rows = []
    for label, count in ordered_counts(df["certification_level"], GRADE_ORDER):
        rows.append({"구분": "등급", "항목": label, "건수": count,
                     "비율": f"{count / n * 100:.1f}%"})
    for label, count in ordered_counts(df["original_version"], VERSION_ORDER):
        rows.append({"구분": "인증 버전", "항목": label, "건수": count,
                     "비율": f"{count / n * 100:.1f}%"})
    rows.append({"구분": "합계", "항목": "-", "건수": n, "비율": "100.0%"})

    table = pd.DataFrame(rows, columns=["구분", "항목", "건수", "비율"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_PATH}\n")
    print(table.to_string(index=False))

    grade_sum = table.loc[table["구분"] == "등급", "건수"].sum()
    ver_sum = table.loc[table["구분"] == "인증 버전", "건수"].sum()
    print(f"\n등급 합계 = {grade_sum}  ({'OK' if grade_sum == n else 'MISMATCH'})")
    print(f"인증 버전 합계 = {ver_sum}  ({'OK' if ver_sum == n else 'MISMATCH'})")

    # 박사님 검토본 본문 수치 대조
    expected = {"Gold": 235, "Silver": 118, "Platinum": 56, "Certified": 51}
    actual = {r["항목"]: r["건수"] for r in rows if r["구분"] == "등급"}
    print("\n[paper cross-check] grade counts:")
    for g, exp in expected.items():
        got = actual.get(g, 0)
        print(f"  {g}: expected {exp}, got {got}  ({'OK' if got == exp else 'DIFF'})")

    # 레거시 Table1_dataset.csv 이동 (삭제하지 않음)
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = LEGACY_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    if LEGACY_SRC.exists():
        shutil.move(str(LEGACY_SRC), str(LEGACY_DIR / LEGACY_SRC.name))
        print(f"\nMoved legacy file -> {LEGACY_DIR / LEGACY_SRC.name}")
    else:
        print(f"\nLegacy file already moved/absent: {LEGACY_SRC.name}")


if __name__ == "__main__":
    main()
