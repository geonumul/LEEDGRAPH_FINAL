"""
LangGraph 노드 정의

아키텍처 (2-track 설계):
    Track 1 - 결정론적 (LLM 없음, 토큰 소모 없음):
        pdf_ingest → csv_match → rule_mapper → hallucination_checker → finalize

    Track 2 - LLM 폴백 (Track 1 실패 시에만 진입):
        hallucination_checker FAIL → llm_mapper ⇄ llm_validator (최대 3회) → finalize

설계 원칙:
    - rule_mapper: 버전별 수식 기반 매핑. 비율 공식, 교통 크레딧 분리 등을 하드코딩.
    - hallucination_checker: LLM 없이 수학적 제약 조건만 검사.
    - llm_mapper / llm_validator: 규칙으로 풀 수 없는 엣지케이스(unknown 버전 등)만 처리.
"""

import json
import time
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        RetryError,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False

from .state import LEEDStandardizationState
from src.data.loader import LEEDDataLoader, LEED_VERSION_MAX_SCORES
from src.data.rubric_loader import load_all_rubrics, get_rubric_max

# 루브릭 캐시: 모듈 로딩 시 1회 스캔 (data/rubrics/ 아래 xlsx 자동 감지)
# 파일이 없으면 빈 dict → 기존 hardcoded fallback으로 동작 (에러 없음)
_RUBRIC_CACHE: dict = load_all_rubrics()

# 매핑 규칙 캐시: data/rubrics/mapping_rules.yaml
# {source_credit_lower: [rule_dict, ...]} 인덱스로 빠른 부분 문자열 조회
_MAPPING_RULES: list = []
_MAPPING_RULES_INDEX: dict = {}
try:
    _rules_path = Path("data/rubrics/mapping_rules.yaml")
    if _rules_path.exists():
        with open(_rules_path, "r", encoding="utf-8") as _f:
            _MAPPING_RULES = yaml.safe_load(_f) or []
        for _rule in _MAPPING_RULES:
            _key = _rule.get("source_credit", "").lower()
            if _key:
                _MAPPING_RULES_INDEX.setdefault(_key, []).append(_rule)
        print(f"[MappingRules] {len(_MAPPING_RULES)}개 규칙 로딩 완료")
except Exception as _e:
    print(f"[MappingRules] 로딩 실패: {_e}")


# =============================================================================
# 상수: v5 카테고리 최대 점수 (rating system별, 실제 v5 xlsx 기반)
# v5부터 총점이 100점으로 변경되었고, IN/RP 카테고리가 폐지됨.
# IEQ → EQ 코드 변경.
# =============================================================================

# BD+C (New Construction, Core and Shell, Data Centers 등)
V5_MAX_BDC: dict = {
    "IP":  1,   # Integrative Process
    "LT": 15,   # Location & Transportation
    "SS": 11,   # Sustainable Sites
    "WE":  9,   # Water Efficiency
    "EA": 33,   # Energy and Atmosphere
    "MR": 18,   # Materials & Resources
    "EQ": 13,   # Indoor Environmental Quality (v5에서 IEQ → EQ 코드 변경)
}  # Total = 100

# ID+C (Commercial Interiors, Retail, Hospitality)
V5_MAX_IDC: dict = {
    "IP":  1,
    "LT": 14,
    "WE": 10,
    "EA": 31,
    "MR": 26,
    "EQ": 18,
}  # Total = 100 (SS 카테고리 없음)

# O+M (Existing Buildings, Warehouses 등)
V5_MAX_OM: dict = {
    "IP":  2,
    "LT":  8,
    "SS":  2,
    "WE": 15,
    "EA": 34,
    "MR": 13,
    "EQ": 26,
}  # Total = 100

V5_TOTAL = 100  # v5 만점 (모든 rating system 공통)

# 하위 호환: rule_mapper/hallucination_checker에서 _get_v5_max()로 동적 선택
def _get_v5_max(leed_system: str) -> dict:
    """leed_system 문자열에서 v5 카테고리 만점 테이블 선택."""
    s = leed_system.lower()
    if "o+m" in s or "o&m" in s or "operation" in s or "maintenance" in s:
        return V5_MAX_OM
    elif "id+c" in s or "interior" in s:
        return V5_MAX_IDC
    else:
        return V5_MAX_BDC  # BD+C 또는 unknown

# =============================================================================
# 상수: 버전별 BD+C 기준 카테고리 최대 점수
# 각 버전의 "원래" 만점 구조. 이 값으로 비율을 계산한 뒤 V5_MAX에 적용.
#
# [중요] LEED O+M / ID+C 등 시스템별 실제 만점은 다를 수 있음.
#         → PDF에서 "possible" 값을 파싱했다면, 이 테이블 대신 그 값을 우선 사용.
# =============================================================================
VERSION_BD_C_MAX: dict = {
    # ── 초기 버전 (총점 69점 체계) ────────────────────────────────────────
    # 교통(Transportation) 크레딧이 SS 안에 포함되어 있음.
    # v2.2 SS 총 14점 중 교통 관련 Credits 4.1~4.4 = 최대 7점 (아래 상세 참조)
    "v1.0 pilot": {"SS": 14, "WE": 5,  "EA": 17, "MR": 13, "IEQ": 15, "IN": 5},
    "v2.0":       {"SS": 14, "WE": 5,  "EA": 17, "MR": 13, "IEQ": 15, "IN": 5},
    "v2.2":       {"SS": 14, "WE": 5,  "EA": 17, "MR": 13, "IEQ": 15, "IN": 5},

    # ── LEED 2009 / v3 (총점 110점 체계) ─────────────────────────────────
    # SS 26점 중 교통 관련 Credits 4.1~4.4 = 최대 6점
    # RP(Regional Priority) 신설 (4점)
    "v2009": {"SS": 26, "WE": 10, "EA": 35, "MR": 14, "IEQ": 15, "IN": 6, "RP": 4},
    "v3":    {"SS": 26, "WE": 10, "EA": 35, "MR": 14, "IEQ": 15, "IN": 6, "RP": 4},

    # ── v4 (총점 110점 체계) ──────────────────────────────────────────────
    # LT(Location & Transportation) 카테고리 신설 → SS에서 분리됨
    # IP(Integrative Process) 신설 (2점)
    # WE가 11점 (v4.1/v5는 12점으로 증가)
    "v4":   {"LT": 16, "SS": 10, "WE": 11, "EA": 33, "MR": 13, "IEQ": 16, "IN": 6, "RP": 4, "IP": 2},

    # ── v4.1 (총점 110점 체계) ────────────────────────────────────────────
    # WE가 12점으로 증가 (v4의 11점 → v4.1의 12점)
    # 나머지는 v5와 동일 → 직접 매핑 가능
    "v4.1": {"LT": 16, "SS": 10, "WE": 12, "EA": 33, "MR": 13, "IEQ": 16, "IN": 6, "RP": 4, "IP": 2},

    # ── v5 (총점 100점 체계, BD+C 기준) ──────────────────────────────────
    # IN/RP 폐지, IEQ→EQ, 총점 110→100
    "v5":   {"IP": 1, "LT": 15, "SS": 11, "WE": 9, "EA": 33, "MR": 18, "EQ": 13},
}

# =============================================================================
# 상수: 구버전 SS 내 교통 크레딧 최대 점수
# SS에서 LT로 분리할 때 사용. 실제 크레딧명 패턴은 아래 TRANSPORT_KEYWORDS 참조.
#
# [근거]
#   v2.2 SS Credits 4.1(1) + 4.2(1) + 4.3(3) + 4.4(2) = 7점
#   v2009/v3 SS Credits 4.1(6점 total) = 6점
#   (참고: USGBC LEED Reference Guide 각 버전 SS 챕터)
# =============================================================================
SS_TRANSPORT_MAX: dict = {
    "v1.0 pilot": 7,   # v2.2와 구조 동일로 추정
    "v2.0":       7,
    "v2.2":       7,   # Credits 4.1(1)+4.2(1)+4.3(3)+4.4(2) = 7
    "v2009":      6,   # Credit 4.1~4.4 합계 = 6
    "v3":         6,
}

# 교통 관련 크레딧을 판별할 키워드
# PDF의 credits 딕셔너리 키(크레딧명)를 소문자 비교
TRANSPORT_KEYWORDS: tuple = (
    "alternative transportation",
    "public transportation",
    "transit",
    "bicycle",
    "low-emitting",
    "fuel-efficient",
    "parking capacity",
    "green vehicles",
    "electric vehicle",
)

# 검증 허용 오차: 달성률(achieved/max) 변화가 이 값 초과 시 hallucination으로 판정
RATIO_DRIFT_THRESHOLD: float = 0.20   # 20%

# hallucination_checker 실패 시 LLM 폴백 허용 최대 반복 횟수
LLM_MAX_ITERATIONS: int = 3


# =============================================================================
# 헬퍼 함수
# =============================================================================

def _extract_transport_from_credits(credits: dict, version: str) -> float:
    """
    PDF에서 파싱된 개별 크레딧(credits)에서 교통 관련 점수 합산.

    PDF 스코어카드에 크레딧 상세 데이터가 있으면 정확하게 분리 가능.
    없으면 역사적 비율(SS_TRANSPORT_MAX 기반)로 추정.

    Args:
        credits: {"Credit: Alternative Transportation - ...": {"awarded": 1, "possible": 1}, ...}
        version: "v2.2" 등 원본 LEED 버전 문자열

    Returns:
        float: 교통 관련 크레딧 획득 점수 합계
    """
    if not credits:
        return 0.0

    total_transport = 0.0
    for credit_name, scores in credits.items():
        name_lower = credit_name.lower()
        if any(kw in name_lower for kw in TRANSPORT_KEYWORDS):
            total_transport += scores.get("awarded", 0)

    return total_transport


def _proportional(awarded: float, old_max: float, new_max: float) -> float:
    """
    비율 환산 공식: (획득점수 / 구버전최대) × 신버전최대

    단, 결과를 new_max로 클램핑하여 초과 방지.

    예시) v4 WE: awarded=8, old_max=11, new_max=12
         → 8/11 × 12 = 8.73 → round(8.73, 2) = 8.73
    """
    if old_max <= 0:
        return 0.0
    raw = (awarded / old_max) * new_max
    return round(min(raw, new_max), 2)


def _clamp(value: float, max_val: float) -> float:
    """값을 [0, max_val] 범위로 클램핑"""
    return round(max(0.0, min(value, max_val)), 2)


def _lookup_credit_rule(credit_name: str, version: str) -> dict | None:
    """
    개별 크레딧명으로 mapping_rules.yaml에서 v5 매핑 규칙 조회.

    매칭 방식: source_credit을 소문자로 변환 후 부분 문자열 매칭.
    여러 규칙 매칭 시 가장 긴 source_credit 우선 (specificity).
    버전 필터: rule의 source_versions에 현재 version이 포함된 것만.

    Args:
        credit_name: PDF 크레딧명 (예: "Credit: Optimize Energy Performance 18pt")
        version: 원본 LEED 버전 (예: "v4")

    Returns:
        dict: 매칭된 규칙 또는 None
    """
    if not credit_name or not _MAPPING_RULES_INDEX:
        return None

    # 전처리: 소문자, 접두사 제거, 점수 표기 제거
    name_lower = credit_name.lower()
    for prefix in ("credit:", "prerequisite:", "prereq:", "requirement:", "leed ap:"):
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):].strip()

    best_rule: dict | None = None
    best_len = 0

    for rule_key, rules in _MAPPING_RULES_INDEX.items():
        if rule_key in name_lower:
            for rule in rules:
                rule_versions = [str(v).lower() for v in rule.get("source_versions", [])]
                if rule_versions and version.lower() not in rule_versions:
                    continue
                if len(rule_key) > best_len:
                    best_len = len(rule_key)
                    best_rule = rule

    return best_rule


def get_llm(model: str = "gpt-4.1", temperature: float = 0.1) -> ChatOpenAI:
    """LLM 인스턴스 생성. LLM 노드에서만 호출됨.

    Phase 6 재실행: gpt-4.1로 복귀 (mini는 시간 단축 효과 미미, 정확도 ↓).
    """
    return ChatOpenAI(model=model, temperature=temperature)


# TPM 30k 기준 최소 대기 (호출 사이 간격)
_LLM_MIN_SLEEP = 2.0
_LLM_MAX_RETRIES = 5


def _invoke_llm_with_retry(llm: ChatOpenAI, messages: list) -> object:
    """
    OpenAI 429 Rate Limit 대응 LLM 호출 래퍼.

    - 호출 전 최소 2초 sleep (TPM 30k 기준)
    - 429/503 등 일시적 오류 시 exponential backoff (2→4→8→16→32초)
    - 최대 5회 재시도 후 실패하면 RateLimitError를 raise
    """
    import openai

    time.sleep(_LLM_MIN_SLEEP)

    if _TENACITY_AVAILABLE:
        @retry(
            retry=retry_if_exception_type((
                openai.RateLimitError,
                openai.APIStatusError,
                openai.APIConnectionError,
            )),
            wait=wait_exponential(multiplier=2, min=2, max=32),
            stop=stop_after_attempt(_LLM_MAX_RETRIES),
            reraise=True,
        )
        def _call():
            return llm.invoke(messages)
        return _call()
    else:
        # tenacity 없으면 단순 재시도
        last_exc = None
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                return llm.invoke(messages)
            except Exception as e:
                last_exc = e
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
        raise last_exc


# =============================================================================
# Node 1: PDF Ingest
# =============================================================================

def pdf_ingest_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [PDF Ingest Node]
    역할: Scorecard PDF를 파싱하여 프로젝트 기초 정보와 카테고리/크레딧 점수 추출.
          pdf_path가 없으면 state의 project를 그대로 사용(수동 입력 모드).
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        return {**state, "logs": ["[PDF Ingest] pdf_path 없음 - project 직접 사용 모드"]}

    loader = LEEDDataLoader()
    try:
        parsed = loader.parse_scorecard_pdf(pdf_path)
        log = (
            f"[PDF Ingest] 완료 - {parsed.get('project_name', '?')} "
            f"(ID: {parsed.get('project_id', '?')}, 버전: {parsed.get('version', '?')}, "
            f"총점: {parsed.get('total_awarded', '?')}/{parsed.get('total_possible', '?')})"
        )
        return {**state, "parsed_pdf": parsed, "logs": [log]}
    except Exception as e:
        return {**state, "status": "failed", "logs": [f"[PDF Ingest] 오류: {e}"]}


# =============================================================================
# Node 2: CSV Match
# =============================================================================

def csv_match_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [CSV Match Node]
    역할: PDF에서 추출한 project_id를 PublicLEEDProjectDirectory CSV와 매칭.
          매칭 결과로 건물 메타데이터(면적, 건물유형 등)를 보완하고 project 필드 구성.

    매칭 우선순위:
        1. project_id 정확 매칭 (신뢰도 최고)
        2. 건물명 소문자 비교 매칭 (fallback)
        3. 매칭 실패 시 PDF 데이터만으로 project 구성

    [면적 변환]
        USGBC CSV의 GrossFloorArea는 sq ft 단위.
        → sq ft × 0.0929 = sqm 변환.
    """
    parsed = state.get("parsed_pdf")
    existing_project = state.get("project")

    # PDF 없이 project 직접 주어진 경우 건너뜀
    if not parsed and existing_project:
        return {**state, "logs": ["[CSV Match] 직접 project 사용 - 매칭 건너뜀"]}

    directory_df = state.get("directory_df")
    if directory_df is None:
        loader = LEEDDataLoader()
        try:
            directory_df = loader.load_project_directory()
        except Exception as e:
            return {**state, "logs": [f"[CSV Match] CSV 로딩 실패: {e}"]}

    loader = LEEDDataLoader()
    matched = loader.match_scorecard_to_directory(parsed, directory_df)

    # PDF 카테고리 → mapper용 형태로 변환 (awarded 값만 추출)
    raw_cats = parsed.get("categories", {})
    cats_for_mapper = {cat: scores.get("awarded", 0) for cat, scores in raw_cats.items()}

    # 기본값: PDF에서
    version = parsed.get("version", "unknown")
    cert_level = parsed.get("certification_level", "")
    total_score = float(parsed.get("total_score", 0))

    # CSV 매칭으로 보완
    if matched:
        cert_level = cert_level or str(matched.get("CertLevel", ""))
        total_score = total_score or float(matched.get("PointsAchieved", 0) or 0)
        if version == "unknown":
            version = str(matched.get("LEEDSystemVersion", "v4"))

    # 면적: sq ft → sqm 변환
    gross_area_sqm = 0.0
    if matched:
        try:
            area_val = float(matched.get("GrossFloorArea", 0) or 0)
            unit = str(matched.get("UnitOfMeasurement", "sq ft"))
            gross_area_sqm = area_val * 0.0929 if "ft" in unit else area_val
        except (ValueError, TypeError):
            gross_area_sqm = 0.0

    project = {
        "project_id":          parsed.get("project_id", ""),
        "project_name":        parsed.get("project_name", ""),
        "version":             version,
        "leed_system":         parsed.get("leed_system", ""),   # BD+C, O+M 등 시스템명 보존
        "building_type":       str(matched.get("ProjectTypes", "")) if matched else "",
        "gross_area_sqm":      gross_area_sqm,
        "certification_level": cert_level,
        "categories":          cats_for_mapper,
        "credits":             parsed.get("credits", {}),       # 크레딧 상세 (SS→LT 분리에 사용)
        "total_score_raw":     total_score,
        # PDF 카테고리의 possible(만점) 값도 보존 → rule_mapper에서 O+M 등 시스템별 만점에 활용
        "categories_possible": {cat: s.get("possible", 0) for cat, s in raw_cats.items()},
    }

    match_method = matched.get("_match_method", "none") if matched else "none"
    if matched:
        log = (
            f"[CSV Match] 매칭 성공 (방법: {match_method}) - "
            f"{project['project_name']} | 버전: {version} | 등급: {cert_level} "
            f"| 면적: {gross_area_sqm:.0f}sqm"
        )
    else:
        log = (
            f"[CSV Match] CSV 매칭 실패 - PDF 데이터만 사용 "
            f"({project['project_name']}, ID: {project['project_id']})"
        )

    return {
        **state,
        "matched_building": matched,
        "project":          project,
        "logs":             [log],
    }


# =============================================================================
# Node 3: Rule Mapper (결정론적 매핑 - LLM 없음)
# =============================================================================

def rule_mapper_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [Rule Mapper Node] - LLM 없음, 토큰 소모 없음

    역할: LEED 버전별 수식 기반 카테고리 매핑.
          모든 한국 LEED 인증 버전(v1.0 pilot ~ v4.1)을 v5 BD+C 기준으로 변환.

    매핑 전략 (버전별):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ v4.1, v5 → 직접 매핑 (구조 동일)                                        │
    │                                                                         │
    │ v4 → WE만 비율 환산 (11pt→12pt), 나머지 직접                            │
    │                                                                         │
    │ v2009/v3 → SS에서 교통 분리(SS_transport → LT),                         │
    │            WE/EA/MR/IEQ 비율 환산, RP 직접, IP=0                        │
    │                                                                         │
    │ v2.2/v2.0/v1.0 pilot → SS에서 교통 분리,                               │
    │                         WE/EA/MR/IEQ/IN 전부 비율 환산, RP=IP=0        │
    └─────────────────────────────────────────────────────────────────────────┘

    SS→LT 교통 분리 우선순위:
        1. PDF 크레딧 상세 데이터 있음 → 키워드 매칭으로 정확히 추출
        2. 크레딧 데이터 없음 → 버전별 역사적 평균 비율로 추정
           (v2.2: 교통max=7, v2009: 교통max=6 기준 비율 적용)

    [O+M 등 비-BD+C 시스템 처리]
        PDF의 categories_possible (스코어카드에 명시된 만점)이 BD+C 만점과
        다를 경우(예: O+M EA=56), 해당 카테고리에만 PDF possible 값을 사용.
        이렇게 하면 시스템별 만점 차이가 자동으로 반영됨.
    """
    project = state.get("project", {})
    version = project.get("version", "unknown")
    cats = project.get("categories", {})          # {카테고리: 획득점수}
    credits = project.get("credits", {})           # {크레딧명: {awarded, possible}}
    cats_possible = project.get("categories_possible", {})  # PDF에서 파싱한 만점
    leed_system = project.get("leed_system", "")

    # ── v5 만점 테이블 (rating system별) ──────────────────────────────────
    v5_max = _get_v5_max(leed_system)

    # ── 버전별 BD+C 기준 만점 테이블 ──────────────────────────────────────
    # unknown 버전은 v4로 fallback (한국 건물 중 가장 많은 비중)
    bd_c_max = VERSION_BD_C_MAX.get(version, VERSION_BD_C_MAX["v4"])

    # ── 카테고리별 실제 만점 결정 ──────────────────────────────────────────
    # 우선순위:
    #   1. PDF possible (스코어카드에 명시된 값 - O+M/ID+C 등 시스템별 차이 자동 반영)
    #   2. 루브릭 xlsx 조회 (data/rubrics/{version}/*.xlsx - 파일 있을 때만)
    #   3. 하드코딩 BD+C 기준 (VERSION_BD_C_MAX + V5_MAX 최후 fallback)
    def get_old_max(cat: str) -> float:
        pdf_possible = cats_possible.get(cat, 0)
        if pdf_possible > 0:
            return float(pdf_possible)
        rubric_max = get_rubric_max(_RUBRIC_CACHE, version, leed_system, cat)
        if rubric_max is not None and rubric_max > 0:
            return float(rubric_max)
        return float(bd_c_max.get(cat, v5_max.get(cat, 1)))

    # ── 1. 교통 크레딧 분리 (SS → SS + LT) ───────────────────────────────
    # v4 이상은 LT가 이미 독립 카테고리이므로 분리 불필요
    needs_lt_split = version in ("v1.0 pilot", "v2.0", "v2.2", "v2009", "v3")

    transport_awarded = 0.0
    transport_max = 0.0
    ss_pure_awarded = float(cats.get("SS", 0))
    ss_pure_max = get_old_max("SS")

    if needs_lt_split:
        # 우선 1: PDF 크레딧 상세 데이터로 정확히 추출
        transport_awarded = _extract_transport_from_credits(credits, version)

        if transport_awarded > 0:
            # 크레딧 데이터 기반 → 교통 만점도 같은 방식으로 추출
            transport_max = sum(
                s.get("possible", 0)
                for name, s in credits.items()
                if any(kw in name.lower() for kw in TRANSPORT_KEYWORDS)
            )
            transport_max = max(transport_max, 1.0)
            ss_pure_awarded = max(0.0, ss_pure_awarded - transport_awarded)
            ss_pure_max = max(0.0, ss_pure_max - transport_max)
            lt_source = "credit-exact"

        else:
            # 우선 2: 크레딧 데이터 없음 → 역사적 비율로 추정
            # v2.2/v2.0/v1.0: 교통max=7, v2009/v3: 교통max=6
            transport_max = float(SS_TRANSPORT_MAX.get(version, 6))
            ss_non_transport_max = max(0.0, ss_pure_max - transport_max)

            # SS 내 교통 점수 = 전체 SS 중 교통 비율만큼 가중
            if ss_pure_max > 0:
                transport_ratio = transport_max / ss_pure_max
                transport_awarded = round(ss_pure_awarded * transport_ratio, 2)
            else:
                transport_awarded = 0.0

            ss_pure_awarded = max(0.0, round(ss_pure_awarded - transport_awarded, 2))
            ss_pure_max = ss_non_transport_max
            lt_source = "ratio-estimated"

    # ── 2. v5 각 카테고리 점수 계산 ──────────────────────────────────────
    mapped: dict = {}

    if needs_lt_split:
        # LT: 분리된 교통 크레딧을 v5 LT 만점으로 비율 환산
        mapped["LT"] = _proportional(transport_awarded, transport_max, v5_max.get("LT", 15))

        # SS: 순수 SS(교통 제외)를 v5 SS 만점으로 비율 환산
        ss_non_transport_v5_base = {
            "v1.0 pilot": 7,   # 14 - 7 = 7 (교통 제외)
            "v2.0":       7,
            "v2.2":       7,   # 14 - 7 = 7
            "v2009":      20,  # 26 - 6 = 20
            "v3":         20,
        }.get(version, ss_pure_max)
        actual_ss_max = ss_pure_max if ss_pure_max > 0 else ss_non_transport_v5_base
        if "SS" in v5_max:
            mapped["SS"] = _proportional(ss_pure_awarded, actual_ss_max, v5_max["SS"])

    elif version in ("v4", "v4.1", "v5"):
        # v4 이상: LT/SS 이미 분리됨 → 직접 또는 비율 환산
        lt_awarded = float(cats.get("LT", 0))
        lt_max = get_old_max("LT")
        mapped["LT"] = _proportional(lt_awarded, lt_max, v5_max.get("LT", 15))

        ss_awarded = float(cats.get("SS", 0))
        ss_max = get_old_max("SS")
        if "SS" in v5_max:
            mapped["SS"] = _proportional(ss_awarded, ss_max, v5_max["SS"])

    else:
        # unknown 버전 fallback
        mapped["LT"] = 0.0
        if "SS" in v5_max:
            mapped["SS"] = _proportional(float(cats.get("SS", 0)), get_old_max("SS"), v5_max["SS"])

    # ── WE ────────────────────────────────────────────────────────────────
    mapped["WE"] = _proportional(float(cats.get("WE", 0)), get_old_max("WE"), v5_max.get("WE", 9))

    # ── EA ────────────────────────────────────────────────────────────────
    mapped["EA"] = _proportional(float(cats.get("EA", 0)), get_old_max("EA"), v5_max.get("EA", 33))

    # ── MR ────────────────────────────────────────────────────────────────
    mapped["MR"] = _proportional(float(cats.get("MR", 0)), get_old_max("MR"), v5_max.get("MR", 18))

    # ── EQ (Indoor Environmental Quality, 구버전 IEQ에서 코드 변경) ────────
    # PDF에서는 여전히 "IEQ"로 파싱되므로 cats.get("IEQ", 0) 사용
    ieq_awarded = float(cats.get("IEQ", cats.get("EQ", 0)))
    ieq_old_max = cats_possible.get("IEQ", cats_possible.get("EQ", 0))
    if ieq_old_max <= 0:
        ieq_old_max = get_old_max("IEQ") or get_old_max("EQ")
    mapped["EQ"] = _proportional(ieq_awarded, float(ieq_old_max), v5_max.get("EQ", 13))

    # ── IP (Integrative Process) ──────────────────────────────────────────
    # v4 이전: IP 없음 → 0
    # v4 이상: 비율 환산
    if version in ("v1.0 pilot", "v2.0", "v2.2", "v2009", "v3"):
        mapped["IP"] = 0.0
    else:
        mapped["IP"] = _proportional(float(cats.get("IP", 0)), get_old_max("IP"), v5_max.get("IP", 1))

    # ── IN, RP: v5에서 폐지 → dropped_categories에 기록, mapped에는 미포함 ──
    dropped: dict = {}
    if cats.get("IN", 0) > 0:
        dropped["IN"] = float(cats["IN"])
    if cats.get("RP", 0) > 0:
        dropped["RP"] = float(cats["RP"])

    total_v5 = round(sum(mapped.values()), 2)
    v5_total_max = sum(v5_max.values())

    # ── 크레딧 레벨 규칙 매핑 (mapping_rules.yaml 조회) ──────────────────
    credit_mappings: list = []
    rule_hits = 0
    rule_misses = 0

    for credit_name, scores in credits.items():
        rule = _lookup_credit_rule(credit_name, version)
        if rule:
            rule_hits += 1
            credit_mappings.append({
                "credit_name":  credit_name,
                "awarded":      scores.get("awarded", 0),
                "possible":     scores.get("possible", 0),
                "v5_code":      rule.get("target_v5_code"),
                "v5_name":      rule.get("target_v5_name"),
                "v5_category":  rule.get("target_v5_category"),
                "confidence":   rule.get("confidence", "medium"),
                "matched":      True,
            })
        else:
            rule_misses += 1
            credit_mappings.append({
                "credit_name":  credit_name,
                "awarded":      scores.get("awarded", 0),
                "possible":     scores.get("possible", 0),
                "v5_code":      "UNKNOWN",
                "v5_name":      None,
                "v5_category":  None,
                "confidence":   None,
                "matched":      False,
            })

    total_credits = rule_hits + rule_misses
    hit_rate = (rule_hits / total_credits) if total_credits > 0 else 0.0

    # ── 매핑 근거 문자열 구성 ─────────────────────────────────────────────
    if needs_lt_split:
        lt_note = (
            f"SS→LT 교통 분리({lt_source}): "
            f"교통점수={transport_awarded:.1f}/{transport_max:.0f}"
        )
    else:
        lt_note = "LT/SS 이미 분리됨 (v4 이상)"

    dropped_note = f" | 폐지카테고리(IN/RP) 제외: {dropped}" if dropped else ""

    rationale = (
        f"[Rule Mapper] 버전={version} | {lt_note} | "
        f"v5 총점={total_v5}/{v5_total_max}{dropped_note} | "
        f"원본 총점={project.get('total_score_raw', '?')}"
    )

    rule_mapping_result = {
        "mapped_categories":   mapped,
        "mapping_rationale":   rationale,
        "proportional_scores": {
            cat: f"{cats.get(cat, cats.get('IEQ', 0) if cat == 'EQ' else 0):.1f}/{get_old_max('IEQ' if cat == 'EQ' else cat):.0f} → {score:.2f}/{v5_max.get(cat, 0)}"
            for cat, score in mapped.items()
        },
        "total_score_v5":      total_v5,
        "dropped_categories":  dropped,       # IN/RP 원본 점수 기록 (분석용)
        "credit_mappings":     credit_mappings,   # 크레딧 레벨 규칙 매핑 결과
        "credit_rule_hits":    rule_hits,
        "credit_rule_misses":  rule_misses,
        "credit_rule_hit_rate": round(hit_rate, 4),
    }

    log = (
        f"[Rule Mapper] {version} → v5 매핑 완료: {total_v5:.1f}/{v5_total_max} | "
        f"크레딧 규칙 히트: {rule_hits}/{total_credits} ({hit_rate:.0%})"
    )
    return {
        **state,
        "rule_mapping_result": rule_mapping_result,
        "validation_mode":     "rule",
        "logs":                [log],
    }


# =============================================================================
# Node 4: Hallucination Checker (수학적 검증 - LLM 없음)
# =============================================================================

def hallucination_checker_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [Hallucination Checker Node] - LLM 없음, 토큰 소모 없음

    역할: rule_mapper 결과의 수학적 타당성을 검증.
          다음 5가지 조건을 모두 통과해야 PASS.

    검증 항목:
        1. 카테고리 점수 범위: 0 ≤ score ≤ V5_MAX[cat]
        2. 총점 일관성: sum(categories) ≈ total_score_v5 (오차 0.5 이내)
        3. 달성률 드리프트: |원본달성률 - v5달성률| ≤ RATIO_DRIFT_THRESHOLD(20%)
        4. 음수 점수 없음
        5. v5에 존재하지 않는 카테고리 없음

    [달성률 계산]
        원본 달성률 = total_score_raw / version_total_max
        v5 달성률   = total_score_v5  / 110
        드리프트가 크면 SS→LT 분리 비율 추정이 잘못됐거나 버전 식별 오류일 가능성.

    [PASS 기준]
        모든 항목 통과 시 → finalize로 이동 (LLM 호출 없음)
    [FAIL 기준]
        하나라도 실패 시 → llm_mapper로 이동 (LLM 폴백)
    """
    project = state.get("project", {})
    mapping = state.get("rule_mapping_result", {})

    if not mapping:
        return {
            **state,
            "math_validation_result": {"passed": False, "issues": ["rule_mapping_result 없음"],
                                       "achievement_ratio_original": 0, "achievement_ratio_v5": 0,
                                       "ratio_drift": 1.0},
            "logs": ["[Hallucination Check] rule_mapping_result 없음 - LLM 폴백"],
        }

    mapped = mapping.get("mapped_categories", {})
    total_v5 = mapping.get("total_score_v5", 0.0)
    version = project.get("version", "v4")
    leed_system = project.get("leed_system", "")
    v5_max = _get_v5_max(leed_system)
    v5_total_max = float(sum(v5_max.values()))
    issues = []

    # ── 검증 1: 카테고리 점수 범위 ────────────────────────────────────────
    for cat, score in mapped.items():
        cat_max = v5_max.get(cat)
        if cat_max is None:
            issues.append(f"v5에 없는 카테고리: {cat}")
            continue
        if score < 0:
            issues.append(f"{cat} 음수: {score}")
        if score > cat_max + 0.01:   # 부동소수점 오차 0.01 허용
            issues.append(f"{cat} 초과: {score:.2f} > max {cat_max}")

    # ── 검증 2: 총점 일관성 ───────────────────────────────────────────────
    computed_total = sum(mapped.values())
    if abs(computed_total - total_v5) > 0.5:
        issues.append(
            f"총점 불일치: sum={computed_total:.2f}, reported={total_v5:.2f}"
        )

    # ── 검증 3: 달성률 드리프트 ───────────────────────────────────────────
    # 원본 달성률
    ver_max_total = sum(VERSION_BD_C_MAX.get(version, VERSION_BD_C_MAX["v4"]).values())
    ver_max_total = max(ver_max_total, 1)
    raw_total = float(project.get("total_score_raw", 0))
    ratio_orig = raw_total / ver_max_total

    # v5 달성률
    ratio_v5 = total_v5 / v5_total_max

    drift = abs(ratio_orig - ratio_v5)
    if drift > RATIO_DRIFT_THRESHOLD:
        issues.append(
            f"달성률 드리프트 {drift:.1%} > 허용({RATIO_DRIFT_THRESHOLD:.0%}): "
            f"원본={ratio_orig:.1%}, v5={ratio_v5:.1%}"
        )

    passed = len(issues) == 0
    result = {
        "passed":                    passed,
        "issues":                    issues,
        "achievement_ratio_original": round(ratio_orig, 4),
        "achievement_ratio_v5":       round(ratio_v5, 4),
        "ratio_drift":                round(drift, 4),
    }

    if passed:
        log = (
            f"[Hallucination Check] PASS - "
            f"달성률: 원본={ratio_orig:.1%} / v5={ratio_v5:.1%} "
            f"(drift={drift:.1%})"
        )
    else:
        log = f"[Hallucination Check] FAIL - 문제 {len(issues)}건: {issues}"

    return {
        **state,
        "math_validation_result": result,
        "logs":                   [log],
    }


# =============================================================================
# Node 5: LLM Mapper (폴백 - LLM 사용, 토큰 소모 있음)
# =============================================================================

def llm_mapper_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [LLM Mapper Node] - 토큰 소모 있음 (V2: rule 거부 시에도 진입)

    V2 진입 경로:
        (1) hallucination_checker FAIL → rule 계산 자체가 이상 → LLM 재매핑
        (2) llm_validator(target=rule) FAIL → LLM이 rule 결과 거부 → LLM 재매핑
        (3) llm_validator(target=llm)  FAIL → 이전 LLM 재매핑 결과 거부 → 재매핑 loop

    역할:
        - rule_mapping_result와 이전 LLM feedback을 모두 context로 활용
        - LLM이 독립적으로 재매핑한 결과를 mapping_result에 저장
        - validation_target을 "llm"으로 전환 (다음 llm_validator는 LLM 결과 검증)
        - validation_mode를 "llm"으로 마킹 (최종 채택 결과 = mapping_result)

    LLM 프롬프트 전략:
        - rule_mapper의 출력 + math_validation 이슈 + validator feedback 모두 전달
        - 버전별 매핑 가이드 텍스트 포함
        - JSON only 응답 강제
    """
    # OPENAI_API_KEY 없으면 LLM 호출 불가 → rule_mapper 결과로 graceful fallback
    import os as _os
    if not _os.environ.get("OPENAI_API_KEY"):
        rule_result = state.get("rule_mapping_result", {})
        log = (
            "[LLM Mapper] OPENAI_API_KEY 없음 - rule_mapper 결과로 fallback (LLM 호출 생략). "
            f"이슈: {state.get('math_validation_result', {}).get('issues', [])}"
        )
        return {
            **state,
            "mapping_result":    rule_result,   # rule 결과를 그대로 사용
            "validation_mode":   "rule",        # rule 경로로 마킹
            "validation_target": "rule",        # 검증 대상 그대로 유지
            "logs":              [log],
        }

    llm = get_llm()
    project = state.get("project", {})
    version = project.get("version", "unknown")
    current_iter = state.get("current_iteration", 0)
    prev_target = state.get("validation_target", "rule")

    # ── 실패 사유 수집 ────────────────────────────────────────────────────
    math_result = state.get("math_validation_result", {})
    math_issues = math_result.get("issues", [])

    # llm_validator 피드백 (재시도 시)
    prev_feedback = ""
    prev_validation = state.get("validation_result")
    if prev_validation and not prev_validation.get("is_valid", True):
        prev_feedback = f"\n\n[이전 검증 실패 피드백]\n{prev_validation.get('feedback', '')}"

    # V2: rule 결과가 LLM 검증에서 거부됐을 때 rule 매핑을 context로 포함
    rule_context = ""
    if prev_target == "rule":
        rule_result = state.get("rule_mapping_result", {})
        if rule_result:
            rule_context = (
                f"\n\n[참고] 결정론적 Rule 매핑 결과 (LLM이 거부함):\n"
                f"  카테고리: {json.dumps(rule_result.get('mapped_categories', {}), ensure_ascii=False)}\n"
                f"  v5 총점: {rule_result.get('total_score_v5', '?')}\n"
                f"  근거: {rule_result.get('mapping_rationale', '')}\n"
                f"이 Rule 결과를 참고하되, 검증 피드백을 반영하여 독립적으로 재매핑하세요."
            )

    # ── 버전 매핑 가이드 ──────────────────────────────────────────────────
    version_guides = {
        "v1.0 pilot / v2.0 / v2.2": (
            "총점 69점 체계. SS(14)에 교통크레딧 포함(약 7pt). "
            "LT 없음→SS 교통분 비율로 LT 추정. RP/IP 없음→0."
        ),
        "v2009 / v3": (
            "총점 110점. SS(26)에 교통크레딧 포함(약 6pt). "
            "LT 없음→SS 교통분 비율로 LT 추정. IP 없음→0."
        ),
        "v4": "총점 110점. LT/SS 이미 분리. WE만 11→12 비율 환산 필요.",
        "v4.1 / v5": "v5와 구조 동일. 직접 매핑.",
    }
    guide_text = "\n".join(f"  {k}: {v}" for k, v in version_guides.items())

    # LLM 프롬프트용 v5_max 계산 (leed_system 기반)
    _llm_v5_max = _get_v5_max(project.get("leed_system", ""))
    _llm_v5_cats_str = ", ".join(f"{k}={v}" for k, v in _llm_v5_max.items())
    _llm_v5_total = sum(_llm_v5_max.values())
    _llm_cats_json = "{" + ", ".join(f'"{k}": <숫자>' for k in _llm_v5_max) + "}"

    system_prompt = f"""당신은 LEED(Leadership in Energy and Environmental Design) 버전 표준화 전문가입니다.
구버전 LEED 카테고리 점수를 최신 v5 기준으로 정확하게 매핑합니다.

v5 카테고리 최대점수: {_llm_v5_cats_str} (합계={_llm_v5_total})
※ v5에서 IN(Innovation), RP(Regional Priority) 카테고리는 폐지됨. IEQ → EQ로 코드 변경.

반드시 다음 JSON 형식으로만 응답하세요:
{{
  "mapped_categories": {_llm_cats_json},
  "mapping_rationale": "<매핑 근거>",
  "proportional_scores": {{}}
}}"""

    user_prompt = f"""다음 LEED 프로젝트를 v5 기준으로 매핑해주세요.

[프로젝트]
버전: {version}
인증등급: {project.get('certification_level', '?')}
원본 총점: {project.get('total_score_raw', '?')}
원본 카테고리 점수(획득): {json.dumps(project.get('categories', {}), ensure_ascii=False)}
원본 카테고리 만점(possible): {json.dumps(project.get('categories_possible', {}), ensure_ascii=False)}

[규칙 기반 매핑 실패 사유]
{chr(10).join(f"  - {i}" for i in math_issues) if math_issues else "  (사유 없음 - 정밀도 향상 목적)"}

[버전별 매핑 가이드]
{guide_text}

주의사항:
1. 각 카테고리 점수가 v5 최대값을 초과하지 않도록 할 것
2. 원본 버전에 없는 카테고리(예: v2.2의 LT, RP, IP)는 0으로 설정
3. 달성률(획득/최대)을 최대한 보존할 것
{rule_context}{prev_feedback}
JSON으로만 응답하세요."""

    try:
        response = _invoke_llm_with_retry(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        resp_text = response.content.strip()
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(resp_text)
        mapped_cats = parsed.get("mapped_categories", {})

        # LLM 출력도 클램핑 (환각 방지 보정)
        _llm_v5_max_clamp = _get_v5_max(project.get("leed_system", ""))
        for cat in list(mapped_cats.keys()):
            mapped_cats[cat] = _clamp(float(mapped_cats[cat]), _llm_v5_max_clamp.get(cat, 999))

        total_v5 = round(sum(mapped_cats.values()), 2)
        _v5_total = sum(_llm_v5_max_clamp.values())
        mapping_result = {
            "mapped_categories":  mapped_cats,
            "mapping_rationale":  parsed.get("mapping_rationale", ""),
            "proportional_scores": parsed.get("proportional_scores", {}),
            "total_score_v5":      total_v5,
        }
        log = f"[LLM Mapper Iter {current_iter+1}] 완료 - v5 총점: {total_v5:.1f}/{_v5_total}"

    except Exception as e:
        # Rate Limit 5회 초과 or 파싱 실패 → rule_mapper 결과 fallback
        fallback = state.get("rule_mapping_result") or {}
        mapping_result = {
            "mapped_categories":  fallback.get("mapped_categories", {}),
            "mapping_rationale":  f"LLM 실패({type(e).__name__}: {e}) - rule fallback",
            "proportional_scores": {},
            "total_score_v5":      fallback.get("total_score_v5", 0),
        }
        log = f"[LLM Mapper Iter {current_iter+1}] 실패 ({type(e).__name__}) - rule fallback 적용"

    return {
        **state,
        "mapping_result":    mapping_result,
        "validation_mode":   "llm",
        "validation_target": "llm",     # V2: 다음 validator는 LLM 결과 검증
        "current_iteration": current_iter + 1,
        "logs":              [log],
    }


# =============================================================================
# Node 6: LLM Validator (V2: rule / llm 두 대상 모두 검증)
# =============================================================================

def llm_validator_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [LLM Validator Node] - 토큰 소모 있음 (V2: rule / llm 두 대상 검증)

    V2 변경:
        - validation_target == "rule" → rule_mapping_result 검증 (의미적 타당성)
        - validation_target == "llm"  → mapping_result 검증 (할루시네이션·수치)
        - 두 경우 프롬프트가 분기됨

    [validation_score >= 0.8] → is_valid=True → finalize
    [validation_score <  0.8 & iter < max] → llm_mapper 재매핑
    [iter >= max] → 강제 통과 (무한 루프 방지)
    """
    llm = get_llm()
    project = state.get("project", {})
    current_iter = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", LLM_MAX_ITERATIONS)
    target = state.get("validation_target", "rule")  # V2

    # V2: 검증 대상 선택
    if target == "rule":
        mapping = state.get("rule_mapping_result", {})
        target_label = "Rule 기반 결정론적 매핑 결과"
    else:
        mapping = state.get("mapping_result", {})
        target_label = "LLM 재매핑 결과"

    # 최대 반복 초과 시 강제 통과
    if current_iter >= max_iter:
        result = {
            "is_valid": True,
            "validation_score": 0.6,
            "issues": ["최대 반복 도달 - 강제 승인"],
            "feedback": "",
            "iteration": current_iter,
            "target": target,
        }
        return {
            **state,
            "validation_result": result,
            "logs": [f"[LLM Validator] 최대 반복({max_iter}) 도달 - 강제 승인"],
        }

    # ══════════════════════════════════════════════════════════════════════
    # Rule 검증 프롬프트 (V2 신규) — 의미적 타당성에 초점
    # ══════════════════════════════════════════════════════════════════════
    if target == "rule":
        system_prompt = """당신은 LEED 인증 심사 전문가입니다.
아래는 결정론적 규칙으로 계산된 v5 매핑 결과입니다. 수치는 수학적 제약을 이미 통과했습니다.
당신의 역할은 이 매핑이 **의미적으로** 타당한지 점검하는 것입니다.

[중점 검증 관점]
1. LEED 버전 특성이 올바르게 반영되었는지
   - v2.2 / v2009: SS 카테고리에 교통 크레딧이 포함됨 → LT 분리가 합리적인지
   - v4 / v4.1: LT/SS 이미 분리되어 있음 → 그대로 유지됐는지
2. 크레딧이 체계적으로 누락되지 않았는지 (e.g., EA 상세 크레딧 대신 카테고리 합계로 처리됐는지)
3. v5 신규 카테고리(IP, Integrative Process)의 배분이 건물 특성에 적절한지
4. 원본 건물 맥락(인증등급, 건물 유형, 연면적)과 v5 환산 점수가 상식적으로 일치하는지
5. 원본 버전에 없는 카테고리(v2.2→LT/RP/IP, v2009→IP)에 점수가 잘못 부여되지 않았는지

[합격 기준]
- validation_score >= 0.8 → is_valid=true (Rule 결과 채택)
- validation_score <  0.8 → is_valid=false (LLM 재매핑 요청)
- Rule이 수학적으로 맞아도 의미가 틀리면 is_valid=false로 판정하세요.

반드시 JSON 형식으로만 응답하세요:
{
  "is_valid": true/false,
  "validation_score": 0.0~1.0,
  "issues": ["의미적 문제점1", "의미적 문제점2"],
  "feedback": "LLM Mapper에게 전달할 재매핑 지시 (버전 특성·누락 크레딧 명시)"
}"""

        user_prompt = f"""다음 Rule 매핑 결과를 의미적으로 검증하세요.

[원본]
버전: {project.get('version', '?')}
인증등급: {project.get('certification_level', '?')}
건물유형: {project.get('building_type', '?')}
연면적: {project.get('gross_area_sqm', '?')} sqm
원본 총점: {project.get('total_score_raw', '?')}
원본 카테고리 점수: {json.dumps(project.get('categories', {}), ensure_ascii=False)}
원본 카테고리 만점: {json.dumps(project.get('categories_possible', {}), ensure_ascii=False)}

[{target_label}]
v5 총점: {mapping.get('total_score_v5', '?')}
카테고리별: {json.dumps(mapping.get('mapped_categories', {}), ensure_ascii=False)}
근거: {mapping.get('mapping_rationale', '')}
크레딧 매핑 성공률: {mapping.get('credit_rule_hit_rate', 'N/A')}

위 매핑이 건물 특성과 LEED 버전 전환 원칙에 **의미적으로** 부합하는지 판단하세요.
수학적 오류가 아니라 **의미적 부적절함**을 찾으세요.
JSON으로만 응답하세요."""

    # ══════════════════════════════════════════════════════════════════════
    # LLM 검증 프롬프트 (기존) — 할루시네이션·수치 오류에 초점
    # ══════════════════════════════════════════════════════════════════════
    else:
        system_prompt = """당신은 LEED 인증 심사 전문가입니다.
아래는 LLM이 재매핑한 v5 결과입니다. LLM 출력의 할루시네이션과 수치 오류를 점검하세요.

반드시 JSON 형식으로만 응답하세요:
{
  "is_valid": true/false,
  "validation_score": 0.0~1.0,
  "issues": ["문제점1", "문제점2"],
  "feedback": "LLM Mapper에게 전달할 개선 지시"
}"""

        user_prompt = f"""다음 LLM 재매핑 결과를 검증하세요.

[원본]
버전: {project.get('version', '?')}
인증등급: {project.get('certification_level', '?')}
원본 총점: {project.get('total_score_raw', '?')}
원본 카테고리: {json.dumps(project.get('categories', {}), ensure_ascii=False)}

[{target_label}]
v5 총점: {mapping.get('total_score_v5', '?')}
카테고리별: {json.dumps(mapping.get('mapped_categories', {}), ensure_ascii=False)}
근거: {mapping.get('mapping_rationale', '')}

[검증 기준]
1. 각 카테고리 ≤ v5 최대값 (LT=16,SS=10,WE=12,EA=33,MR=13,IEQ=16,IN=6,RP=4,IP=2)
2. 인증등급 일관성: Certified=40~49, Silver=50~59, Gold=60~79, Platinum=80+
3. 달성률 드리프트 ≤ 20%
4. 원본 버전에 없는 카테고리(v2.2→LT/RP/IP, v2009→IP)에 0 이상 점수 부여 금지

validation_score >= 0.8 → is_valid=true
JSON으로만 응답하세요."""

    try:
        response = _invoke_llm_with_retry(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        # Rate limit 초과 → 강제 통과 (rule fallback과 동일 처리)
        result = {
            "is_valid": True,
            "validation_score": 0.5,
            "issues": [f"LLM Validator 호출 실패({type(e).__name__}) - 강제 승인"],
            "feedback": "",
            "iteration": current_iter,
        }
        return {
            **state,
            "validation_result": result,
            "logs": [f"[LLM Validator] 호출 실패 ({type(e).__name__}) - rule fallback 강제 승인"],
        }

    try:
        resp_text = response.content.strip()
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()

        parsed_resp = json.loads(resp_text)
        result = {
            "is_valid":         parsed_resp.get("is_valid", False),
            "validation_score": parsed_resp.get("validation_score", 0.0),
            "issues":           parsed_resp.get("issues", []),
            "feedback":         parsed_resp.get("feedback", ""),
            "iteration":        current_iter,
            "target":           target,     # V2: 어느 대상을 검증했는지
        }
        status_str = "PASS" if result["is_valid"] else "FAIL"
        log = (
            f"[LLM Validator - {target} 경로 Iter {current_iter}] {status_str} "
            f"(score={result['validation_score']:.2f})"
        )

    except Exception as e:
        # 파싱 실패 → 수학적 체크만으로 판정
        _val_v5_max = _get_v5_max(state.get("project", {}).get("leed_system", ""))
        mapped = mapping.get("mapped_categories", {})
        fallback_issues = [
            f"{cat} 초과: {s:.2f} > {_val_v5_max.get(cat, 0)}"
            for cat, s in mapped.items()
            if s > _val_v5_max.get(cat, 0) + 0.01
        ]
        is_valid = len(fallback_issues) == 0
        result = {
            "is_valid":         is_valid,
            "validation_score": 0.85 if is_valid else 0.5,
            "issues":           fallback_issues,
            "feedback":         "; ".join(fallback_issues),
            "iteration":        current_iter,
        }
        log = f"[LLM Validator Iter {current_iter}] 파싱 실패 ({e}) - 수학적 폴백"

    return {
        **state,
        "validation_result": result,
        "logs":              [log],
    }


# =============================================================================
# Node 7: Finalize
# =============================================================================

def finalize_node(state: LEEDStandardizationState) -> LEEDStandardizationState:
    """
    [Finalize Node]
    역할: 검증 통과된 매핑 결과를 최종 v5 표준화 데이터로 저장.
          rule_mapper 경로와 llm_mapper 경로 모두 이 노드에서 수렴.

    출력 필드:
        - project 메타데이터
        - v5 카테고리별 점수 (score_v5_{CAT})
        - 원본 카테고리별 점수 (score_orig_{CAT})
        - 표준화에 사용된 경로 (standardization_track: "rule" | "llm")
        - 달성률 정보
    """
    project = state.get("project", {})
    # V2: validation_target 기준으로 최종 결과 선택
    # - target="rule" (llm_validator가 rule 결과 PASS) → rule_mapping_result 채택
    # - target="llm"  (LLM 재매핑 완료 또는 진행 중)   → mapping_result 채택
    target = state.get("validation_target", "rule")
    mode = "rule" if target == "rule" else "llm"

    if target == "rule":
        mapping = state.get("rule_mapping_result", {})
    else:
        mapping = state.get("mapping_result", state.get("rule_mapping_result", {}))

    mapped = mapping.get("mapped_categories", {})
    total_v5 = mapping.get("total_score_v5", sum(mapped.values()))

    _fin_v5_max = _get_v5_max(project.get("leed_system", ""))
    _fin_v5_total = float(sum(_fin_v5_max.values()))

    # 달성률
    ver_max_total = sum(VERSION_BD_C_MAX.get(project.get("version", "v4"),
                                              VERSION_BD_C_MAX["v4"]).values())
    raw_total = float(project.get("total_score_raw", 0))
    ratio_orig = round(raw_total / max(ver_max_total, 1), 4)
    ratio_v5 = round(total_v5 / _fin_v5_total, 4)

    final_data = {
        # 식별 정보
        "project_id":              project.get("project_id", ""),
        "project_name":            project.get("project_name", ""),
        "leed_system":             project.get("leed_system", ""),
        "building_type":           project.get("building_type", ""),
        "gross_area_sqm":          project.get("gross_area_sqm", 0),
        # 원본 정보
        "original_version":        project.get("version", ""),
        "certification_level":     project.get("certification_level", ""),
        "total_score_original":    raw_total,
        "achievement_ratio_original": ratio_orig,
        # v5 매핑 결과
        "total_score_v5":          total_v5,
        "achievement_ratio_v5":    ratio_v5,
        # ── ML feature용 카테고리별 달성률 (0~1) ──────────────────────────
        # ratio_{cat} = score_v5_{cat} / V5_MAX[cat] = 원본 acquired / 원본 possible
        #
        # 왜 이렇게 계산해도 같은가?
        #   score_v5 = (awarded / old_max) * v5_max
        #   → score_v5 / v5_max = awarded / old_max = 달성률
        #
        # 즉 버전이 달라도 "이 카테고리에서 몇 %를 달성했냐"는 값은 동일하게 보존됨.
        # ML 모델에는 이 ratio 필드를 feature로 사용할 것.
        **{f"ratio_{cat}": round(score / _fin_v5_max.get(cat, 1), 4)
           for cat, score in mapped.items()},
        # 카테고리별 v5 절대점수 (논문 방법론 기술용 - ML feature로는 미사용)
        **{f"score_v5_{cat}": score for cat, score in mapped.items()},
        # 카테고리별 원본 점수 (비교용)
        **{f"score_orig_{cat}": project.get("categories", {}).get(cat, 0)
           for cat in mapped},
        # 메타
        "standardization_track":   mode,
        "standardization_iterations": state.get("current_iteration", 0),
        "mapping_rationale":       mapping.get("mapping_rationale", ""),
        # Option A: LLM 리뷰 메타데이터 (점수는 바꾸지 않음, 분석용 signal)
        "llm_review_target":       (state.get("validation_result") or {}).get("target"),
        "llm_review_is_valid":     (state.get("validation_result") or {}).get("is_valid"),
        "llm_review_score":        (state.get("validation_result") or {}).get("validation_score"),
        "llm_review_issues":       "; ".join((state.get("validation_result") or {}).get("issues", []))[:500],
        "llm_review_feedback":     ((state.get("validation_result") or {}).get("feedback") or "")[:500],
    }

    log = (
        f"[Finalize] 완료 ({mode} 경로) - "
        f"v5={total_v5:.1f}/110 | "
        f"달성률 원본={ratio_orig:.1%} → v5={ratio_v5:.1%}"
    )
    return {
        **state,
        "final_v5_data": final_data,
        "status":        "completed",
        "logs":          [log],
    }
