"""Regenerate project_features.parquet and standardized_credits.parquet.

Re-runs the deterministic (rule-only) standardization pipeline over all 460
scorecard PDFs. Use this after changing the PDF parser or the rule mapper.

Rule-only by design: OPENAI_API_KEY is removed from the environment so the
LangGraph workflow takes the `route_after_hallucination_check -> finalize`
branch for every project, exactly as the committed project_features.parquet
(standardization_track == "rule" for all 460 rows) was produced.

Mirrors notebook 01 (`01_전처리.ipynb`) cells 6 and 8.
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "notebooks"))

# Force the deterministic rule-only path (see module docstring).
os.environ.pop("OPENAI_API_KEY", None)

from src.langgraph_workflow.graph import run_standardization  # noqa: E402
from src.data.loader import LEEDDataLoader  # noqa: E402

# load_dotenv() runs at graph.py import time and may re-populate the key.
os.environ.pop("OPENAI_API_KEY", None)

DATA = ROOT / "data"
FEATURES_PATH = DATA / "project_features.parquet"
CREDITS_PATH = DATA / "standardized_credits.parquet"


def build_credit_rows(project_id, version, leed_system, credit_mappings):
    rows = []
    for cm in credit_mappings:
        rows.append({
            "project_id": project_id,
            "source_version": version,
            "leed_system": leed_system,
            "source_credit_name": cm.get("credit_name", ""),
            "v5_credit_code": cm.get("v5_code", "UNKNOWN"),
            "v5_category": cm.get("v5_category"),
            "points_awarded": cm.get("awarded", 0),
            "points_possible": cm.get("possible", 0),
            "mapping_method": "rule" if cm.get("matched") else "unmatched",
        })
    return rows


def main() -> None:
    csv_df = LEEDDataLoader().load_project_directory()
    pdf_files = sorted((DATA / "scorecards").glob("*.pdf"))
    print(f"CSV projects: {len(csv_df)}, PDFs: {len(pdf_files)}")

    old = pd.read_parquet(FEATURES_PATH) if FEATURES_PATH.exists() else None

    feature_rows, credit_rows = [], []
    counters = {"success": 0, "failed": 0}

    for i, pdf in enumerate(pdf_files, 1):
        if i % 50 == 0 or i == 1 or i == len(pdf_files):
            print(f"[{i:3d}/{len(pdf_files)}] {pdf.name[:55]}")
        try:
            state = run_standardization(pdf_path=str(pdf), directory_df=csv_df)
            if state.get("status") != "completed":
                raise RuntimeError(f"status={state.get('status')}")

            final = state["final_v5_data"]
            project = state.get("project", {})
            rule_result = state.get("rule_mapping_result", {})
            math_result = state.get("math_validation_result", {})

            version = final.get("original_version", "unknown")
            project_id = final.get("project_id", pdf.name)
            leed_system = final.get("leed_system", "")

            feature_rows.append({
                "project_id": project_id,
                "project_name": final.get("project_name", ""),
                "leed_system": leed_system,
                "building_type": final.get("building_type", ""),
                "gross_area_sqm": final.get("gross_area_sqm", 0),
                "original_version": version,
                "certification_level": final.get("certification_level", ""),
                "total_score_original": final.get("total_score_original", 0),
                "total_score_v5": final.get("total_score_v5", 0),
                "achievement_ratio_original": final.get("achievement_ratio_original", 0),
                "achievement_ratio_v5": final.get("achievement_ratio_v5", 0),
                "standardization_track": final.get("standardization_track", "rule"),
                "drift": math_result.get("ratio_drift", 0),
                "credit_rule_hit_rate": rule_result.get("credit_rule_hit_rate", None),
                **{k: v for k, v in final.items() if k.startswith("ratio_")},
                **{k: v for k, v in final.items() if k.startswith("score_v5_")},
            })

            cm = rule_result.get("credit_mappings", [])
            if cm:
                credit_rows.extend(build_credit_rows(project_id, version, leed_system, cm))
            else:
                for cat in rule_result.get("mapped_categories", {}):
                    credit_rows.append({
                        "project_id": project_id, "source_version": version,
                        "leed_system": leed_system,
                        "source_credit_name": f"[category] {cat}",
                        "v5_credit_code": f"CAT_{cat}", "v5_category": cat,
                        "points_awarded": project.get("categories", {}).get(cat, 0),
                        "points_possible": project.get("categories_possible", {}).get(cat, 0),
                        "mapping_method": "category_proportional",
                    })

            counters["success"] += 1
        except Exception as e:
            counters["failed"] += 1
            print(f"  ERROR {pdf.name}: {e}")

    print(f"\nDone: {counters}")

    feat_df = pd.DataFrame(feature_rows)
    cred_df = pd.DataFrame(credit_rows)
    feat_df.to_parquet(FEATURES_PATH, index=False)
    cred_df.to_parquet(CREDITS_PATH, index=False)
    print(f"Saved {FEATURES_PATH.name}: {len(feat_df)} rows")
    print(f"Saved {CREDITS_PATH.name}: {len(cred_df)} rows")

    print("\ntrack:", feat_df["standardization_track"].value_counts().to_dict())
    print("drift>0.20:", int((feat_df['drift'] > 0.20).sum()))

    # ── IP-fix impact: old vs new ───────────────────────────────────────────
    if old is not None and "score_v5_IP" in feat_df.columns:
        merged = old[["project_id", "score_v5_IP", "total_score_v5", "drift"]].merge(
            feat_df[["project_id", "score_v5_IP", "total_score_v5", "drift"]],
            on="project_id", how="inner", suffixes=("_old", "_new"),
        )
        ip_changed = (merged["score_v5_IP_old"].fillna(0)
                      != merged["score_v5_IP_new"].fillna(0))
        print(f"\n[IP-fix impact]  rows with score_v5_IP changed: {ip_changed.sum()}/{len(merged)}")
        print(f"  score_v5_IP nonzero: old={int((old['score_v5_IP'].fillna(0) > 0).sum())}"
              f" -> new={int((feat_df['score_v5_IP'].fillna(0) > 0).sum())}")
        d = (merged["total_score_v5_new"] - merged["total_score_v5_old"]).abs()
        print(f"  total_score_v5 mean abs delta: {d.mean():.4f}, max: {d.max():.4f}")
        dd = (merged["drift_new"] - merged["drift_old"]).abs()
        print(f"  drift mean abs delta: {dd.mean():.4f}, max: {dd.max():.4f}")


if __name__ == "__main__":
    main()
