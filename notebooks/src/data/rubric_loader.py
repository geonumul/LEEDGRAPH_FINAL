"""
LEED 루브릭 xlsx 자동 로더

data/rubrics/ 폴더 구조에 파일을 넣으면 자동으로 읽힘.

폴더 구조 (버전별 서브폴더):
    data/rubrics/
        v4/
            LEED_v4_BDC_New_Construction.xlsx
            LEED_v4_BDC_Core_Shell.xlsx
            LEED_v4_OM_Existing_Buildings.xlsx
            ...
        v4.1/
            LEED_v4.1_BDC_New_Construction.xlsx
            ...
        v2009/
            LEED_v2009_BDC_New_Construction.xlsx
            ...
        v2.2/
            LEED_v2.2_BDC_New_Construction.xlsx
            ...
        v2.0/
            ...
        v1.0_pilot/
            ...
        v5/
            LEED_v5_Scorecard_BDC_New_Construction.xlsx  ← 이미 있음

파일명 규칙 (아무 이름이나 가능하나, 권장):
    LEED_{버전}_{시스템유형}.xlsx
    예) LEED_v4_BDC_New_Construction.xlsx
        LEED_v4_OM_Existing_Buildings.xlsx
        LEED_v2009_BDC_Schools.xlsx

사용 방법:
    from src.data.rubric_loader import load_all_rubrics, get_rubric_max

    cache = load_all_rubrics()
    max_ea = get_rubric_max(cache, version="v4", leed_system="LEED O+M: EB&O (v4)", cat="EA")
    # → 56.0 (O+M v4의 EA 만점) 또는 None (파일 없으면)
"""

import re
from pathlib import Path

import pandas as pd


# =============================================================================
# 카테고리 키워드 매핑
# 각 카테고리를 식별하는 핵심 키워드 (소문자)
# =============================================================================
_CAT_KEYWORDS: dict = {
    "LT":  ["location", "transportation"],
    "SS":  ["sustainable", "site"],
    "WE":  ["water", "effici"],
    "EA":  ["energy", "atmospher"],
    "MR":  ["material", "resource"],
    "IEQ": ["indoor", "environmental"],
    "IN":  ["innovation"],
    "RP":  ["regional", "priority"],
    "IP":  ["integrative", "process"],
}


def _detect_category(text: str) -> str | None:
    """텍스트 한 줄에서 LEED 카테고리 코드 반환. 매칭 없으면 None."""
    t = text.lower()
    for cat, keywords in _CAT_KEYWORDS.items():
        if all(kw in t for kw in keywords):
            return cat
    return None


def _extract_version_from_folder(path: Path) -> str:
    """
    파일 경로의 폴더명에서 버전 추출.

    예) rubrics/v4/BDC_NC.xlsx → "v4"
        rubrics/v4.1/BDC_NC.xlsx → "v4.1"
        rubrics/v1.0_pilot/BDC_NC.xlsx → "v1.0 pilot"
    """
    for part in path.parts:
        part_lower = part.lower()
        if re.match(r"v\d", part_lower):
            # v1.0_pilot → "v1.0 pilot" (공백으로 정규화)
            return part_lower.replace("_", " ")
    return "unknown"


def _parse_v5_rubric_xlsx(filepath: Path) -> dict:
    """
    LEED v5 스코어카드 xlsx 파서 (v4와 포맷이 다름).

    v5 xlsx 구조:
        시트 1 (Cover): 설명만 있음
        시트 2 (credit category view): 실제 데이터
            - 카테고리 헤더 행: ['Category Name (CODE)', 'max_pts', '0']
            - 크레딧 행: ['False', 'False', 'False', 'CREDITcode', 'Name', 'pts', '0']

    Returns:
        dict: {cat_code: max_points}  예) {"IP": 1, "LT": 15, "EQ": 13, ...}
    """
    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        if len(xl.sheet_names) < 2:
            return {}
        df = xl.parse(xl.sheet_names[1], header=None)
    except Exception as e:
        print(f"[RubricLoader] v5 읽기 실패: {filepath.name} ({e})")
        return {}

    cat_maxes: dict = {}
    skip_words = {"total", "leed", "project", "yes", "maybe", "no", "how", "false", "true", "key"}

    for _, row in df.iterrows():
        vals = [str(v) for v in row.values if str(v) not in ("nan", "None", "")]
        if not vals:
            continue
        # 카테고리 헤더 행: vals[0]에 '(CODE)' 형식, vals[1]이 만점
        first = vals[0].lower()
        if any(first.startswith(w) for w in skip_words):
            continue
        if "(" in vals[0] and len(vals) >= 2:
            try:
                pts = float(vals[1])
                if pts > 0:
                    code_m = re.search(r"\((\w+)\)", vals[0])
                    if code_m:
                        code = code_m.group(1).upper()
                        cat_maxes[code] = int(pts)
            except (ValueError, TypeError):
                pass

    return cat_maxes


def _parse_rubric_xlsx(filepath: Path) -> dict:
    """
    LEED 스코어카드 xlsx에서 카테고리별 만점(Possible Points) 추출.

    USGBC 스코어카드 xlsx는 보통 첫 번째 시트에:
        - 카테고리 헤더 행 (예: "SUSTAINABLE SITES", "Possible Points  10")
        - 또는 "Category | Possible | ..." 컬럼 구조

    두 가지 파싱 방식을 순서대로 시도:
        1. "Possible" 컬럼 기반 파싱
        2. 행 텍스트에서 카테고리 + 숫자 직접 추출

    Returns:
        dict: {cat: max_points} 또는 {} (파싱 실패 시)
    """
    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        df = xl.parse(xl.sheet_names[0], header=None)
    except Exception as e:
        print(f"[RubricLoader] 읽기 실패: {filepath.name} ({e})")
        return {}

    cat_maxes: dict = {}

    # ── 방식 1: 컬럼 구조 파싱 ──────────────────────────────────────────
    # "Possible" 또는 "Max" 단어가 포함된 컬럼 찾기
    header_row = None
    for i, row in df.iterrows():
        row_str = " ".join(str(v).lower() for v in row.values if pd.notna(v))
        if "possible" in row_str or "maximum" in row_str or "max point" in row_str:
            header_row = i
            break

    if header_row is not None:
        possible_col = None
        for j, val in enumerate(df.iloc[header_row]):
            if pd.notna(val) and "possible" in str(val).lower():
                possible_col = j
                break

        if possible_col is not None:
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i]
                row_text = " ".join(str(v) for v in row.values if pd.notna(v))
                cat = _detect_category(row_text)
                if cat and cat not in cat_maxes:
                    val = row.iloc[possible_col]
                    try:
                        cat_maxes[cat] = float(val)
                    except (ValueError, TypeError):
                        pass

    # ── 방식 2: 행 텍스트 직접 파싱 ─────────────────────────────────────
    # 방식 1로 못 찾은 카테고리를 텍스트 패턴으로 보완
    if len(cat_maxes) < 5:
        for _, row in df.iterrows():
            row_text = " ".join(str(v) for v in row.values if pd.notna(v))
            cat = _detect_category(row_text)
            if not cat or cat in cat_maxes:
                continue

            # "Possible Points: 33" 또는 숫자 직접 추출
            m = re.search(r"possible[^0-9]*?(\d+)", row_text, re.IGNORECASE)
            if m:
                cat_maxes[cat] = float(m.group(1))
                continue

            # 줄에서 등장하는 가장 큰 정수를 만점으로 추정
            nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", row_text)]
            if nums:
                cat_maxes[cat] = float(max(nums))

    return cat_maxes


# =============================================================================
# 공개 API
# =============================================================================

def load_all_rubrics(rubrics_dir: str = "data/rubrics") -> dict:
    """
    rubrics_dir 아래 모든 xlsx 파일을 스캔하여 캐시 구성.

    Returns:
        dict 구조:
        {
            "v4": {
                "leed_v4_bdc_new_construction": {"LT": 16, "SS": 10, ...},
                "leed_v4_om_existing_buildings": {"LT": 0, "EA": 56, ...},
                ...
            },
            "v4.1": { ... },
            ...
        }

    파일이 하나도 없으면 빈 dict 반환 (에러 없음).
    """
    cache: dict = {}
    rubrics_path = Path(rubrics_dir)

    if not rubrics_path.exists():
        return cache

    xlsx_files = list(rubrics_path.rglob("*.xlsx"))
    if not xlsx_files:
        return cache

    for xlsx_file in xlsx_files:
        version = _extract_version_from_folder(xlsx_file)
        # 폴더명에서 rating system 추출 (예: v4/BD+C_NewConstruction → "bd+c_newconstruction")
        parts = xlsx_file.parts
        rubrics_idx = next((i for i, p in enumerate(parts) if "rubrics" in p.lower()), -1)
        if rubrics_idx >= 0 and rubrics_idx + 2 < len(parts):
            fname_key = parts[rubrics_idx + 2].lower()   # rating system 폴더명 사용
        else:
            fname_key = xlsx_file.stem.lower()

        # v5는 별도 파서 사용
        if version == "v5":
            cat_maxes = _parse_v5_rubric_xlsx(xlsx_file)
        else:
            cat_maxes = _parse_rubric_xlsx(xlsx_file)
        if not cat_maxes:
            continue

        if version not in cache:
            cache[version] = {}
        cache[version][fname_key] = cat_maxes

        cats_found = list(cat_maxes.keys())
        print(f"[RubricLoader] {xlsx_file.name} ({version}) → 카테고리 {len(cats_found)}개: {cat_maxes}")

    loaded = sum(len(v) for v in cache.values())
    print(f"[RubricLoader] 총 {loaded}개 루브릭 파일 로딩 완료")
    return cache


def get_rubric_max(
    cache: dict,
    version: str,
    leed_system: str,
    cat: str,
) -> float | None:
    """
    캐시에서 특정 버전/시스템의 카테고리 만점 조회.

    매칭 전략 (우선순위 순):
        1. leed_system 키워드가 파일명에 일부 포함
           예) "LEED O+M: EB&O (v4)" → "om" 키워드 → "leed_v4_om_existing_buildings"
        2. 해당 버전의 첫 번째 파일 (fallback)

    Returns:
        float: 만점 또는 None (캐시에 해당 버전/카테고리 없음)
    """
    ver_cache = cache.get(version, {})
    if not ver_cache:
        return None

    # leed_system에서 핵심 키워드 추출
    # "LEED BD+C: New Construction (v4)" → ["bdc", "new", "construction"]
    sys_keywords = re.findall(r"[a-z]+", leed_system.lower())
    sys_keywords = [k for k in sys_keywords if len(k) > 2 and k not in ("leed", "the", "and")]

    best_match = None
    best_score = 0
    for fname_key, cat_maxes in ver_cache.items():
        score = sum(1 for kw in sys_keywords if kw in fname_key)
        if score > best_score:
            best_score = score
            best_match = cat_maxes

    # 매칭 파일 없으면 첫 번째 파일 사용
    if best_match is None:
        best_match = next(iter(ver_cache.values()))

    val = best_match.get(cat)
    return float(val) if val else None
