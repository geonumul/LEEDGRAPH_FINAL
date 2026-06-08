"""
LangGraph 그래프 구성 (V2: LLM 의무 검증 파이프라인)

아키텍처:
    [공통] pdf_ingest → csv_match → rule_mapper → hallucination_checker
                                                       │
                           ┌───────────────────────────┼──────────────────────┐
                  math PASS│                  math FAIL│             no API KEY│
                           ▼                           ▼                       ▼
                   [llm_validator]              [llm_mapper]              [finalize]
                   target=rule                        │                        (rule 결과)
                           │                          ▼
                   ┌───────┴───────┐            [llm_validator]
             PASS  │       │  FAIL │            target=llm
                   ▼       ▼       │                  │
              [finalize] [llm_mapper]           ┌─────┴─────┐
              (rule결과) │     (target→llm)     │PASS │FAIL+│
                         ▼                      ▼     ▼     │
                   [llm_validator]         finalize llm_mapper(loop)
                   target=llm              (llm결과)
                         │
                      ... (이하 LLM loop)

설계 원칙 (V2 변경):
    - Rule은 주 계산 주체 (결정론·재현성)
    - LLM은 **모든** 매핑 결과의 검증자 (의미적 타당성)
    - rule 검증 실패 시 → LLM 재매핑 → LLM 검증 loop (Option A: LLM 판단 존중)
    - OPENAI_API_KEY 없으면 기존처럼 finalize 직행 (graceful degradation)
"""

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, END
from .state import LEEDStandardizationState
from .nodes import (
    pdf_ingest_node,
    csv_match_node,
    rule_mapper_node,
    hallucination_checker_node,
    llm_mapper_node,
    llm_validator_node,
    finalize_node,
)


# =============================================================================
# 조건부 엣지 함수
# =============================================================================

def route_after_hallucination_check(state: LEEDStandardizationState) -> str:
    """
    hallucination_checker 결과에 따라 다음 노드 결정 (V2).

    V2 변경: PASS 브랜치가 "finalize" → "llm_validator" 로 변경
        (rule 결과도 LLM 의무 검증 거치게 됨)

    라우팅:
        no API KEY      → "finalize"      (LLM 없이 rule 결과 그대로)
        math PASS       → "llm_validator" (Rule 결과 LLM 의미 검증, target=rule)
        math FAIL       → "llm_mapper"    (Rule 계산 자체가 이상 → 구제 재매핑)
    """
    import os
    math_result = state.get("math_validation_result", {})
    # OPENAI_API_KEY 없으면 LLM 호출 불가 → graceful degradation
    if not os.environ.get("OPENAI_API_KEY"):
        return "finalize"
    if math_result.get("passed", False):
        return "llm_validator"     # V2 신규: rule 결과도 LLM 검증
    return "llm_mapper"


def route_after_llm_validation(state: LEEDStandardizationState) -> str:
    """
    llm_validator 결과에 따라 다음 노드 결정.

    Option A (LLM = Advisor) 구조:
        - 항상 "finalize"로 이동
        - LLM 판단은 메타데이터로만 기록되며 Rule 결과는 유지됨
        - 재매핑(llm_mapper loop) 없음 — LLM이 점수를 바꾸지 않음

    배경: 본 연구는 LEED 등급 결정 요인 분석이 본론이며 NLP 논문이 아님.
          LLM 재매핑 정확도가 방법론 주요 검증 대상이 되는 것을 피하기 위해,
          LLM은 "전문가 리뷰" 역할만 담당하고 점수는 결정론적 Rule을 유지.

    예외: math FAIL 구제 경로에서 llm_mapper를 통해 온 경우
          (validation_target="llm"), 이는 Rule이 계산 자체를 실패한 경우이므로
          LLM 재매핑 결과를 그대로 사용 (finalize 동일).
    """
    return "finalize"


# =============================================================================
# 그래프 빌드
# =============================================================================

def build_standardization_graph() -> StateGraph:
    """
    LEED 버전 표준화 LangGraph 워크플로우 구성.

    노드 목록:
        pdf_ingest           - PDF 파싱 (LLM 없음)
        csv_match            - 건물 목록 CSV 매칭 (LLM 없음)
        rule_mapper          - 결정론적 버전 매핑 (LLM 없음)
        hallucination_checker - 수학적 검증 (LLM 없음)
        llm_mapper           - LLM 기반 매핑 (폴백, 토큰 소모)
        llm_validator        - LLM 기반 검증 (폴백, 토큰 소모)
        finalize             - 최종 데이터 저장 (LLM 없음)
    """
    graph = StateGraph(LEEDStandardizationState)

    # ── 노드 등록 ─────────────────────────────────────────────────────────
    graph.add_node("pdf_ingest",            pdf_ingest_node)
    graph.add_node("csv_match",             csv_match_node)
    graph.add_node("rule_mapper",           rule_mapper_node)
    graph.add_node("hallucination_checker", hallucination_checker_node)
    graph.add_node("llm_mapper",            llm_mapper_node)
    graph.add_node("llm_validator",         llm_validator_node)
    graph.add_node("finalize",              finalize_node)

    # ── 엣지: Track 1 (결정론적) ──────────────────────────────────────────
    graph.set_entry_point("pdf_ingest")
    graph.add_edge("pdf_ingest",  "csv_match")
    graph.add_edge("csv_match",   "rule_mapper")
    graph.add_edge("rule_mapper", "hallucination_checker")

    # hallucination_checker 분기 (V2): PASS→llm_validator, FAIL→llm_mapper, no API→finalize
    graph.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination_check,
        {
            "finalize":      "finalize",       # no API KEY (graceful degradation)
            "llm_validator": "llm_validator",  # V2 신규: rule 결과 LLM 의무 검증
            "llm_mapper":    "llm_mapper",     # math FAIL → 재매핑
        },
    )

    # ── 엣지: LLM 재매핑 → LLM 검증 (loop) ────────────────────────────────
    graph.add_edge("llm_mapper", "llm_validator")

    # llm_validator 분기 (Option A): 항상 finalize (LLM 재매핑 제거)
    graph.add_conditional_edges(
        "llm_validator",
        route_after_llm_validation,
        {
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# =============================================================================
# 실행 함수
# =============================================================================

def run_standardization(
    project_data: dict = None,
    pdf_path: str = None,
    directory_df=None,
    max_iterations: int = 3,
) -> dict:
    """
    단일 프로젝트 버전 표준화 실행.

    Args:
        project_data:  직접 구성된 project dict (pdf_path 없을 때 사용)
        pdf_path:      Scorecard PDF 경로 (있으면 pdf_ingest → csv_match로 자동 구성)
        directory_df:  이미 로딩된 CSV DataFrame (반복 실행 시 재사용하여 I/O 절약)
        max_iterations: LLM 폴백 경로 최대 반복 횟수 (기본 3)

    Returns:
        dict: 최종 LangGraph State (final_v5_data, logs, status 등 포함)
    """
    graph = build_standardization_graph()

    initial_state: LEEDStandardizationState = {
        # 입력
        "pdf_path":              pdf_path,
        "directory_df":          directory_df,
        # 초기화
        "parsed_pdf":            None,
        "matched_building":      None,
        "project":               project_data,
        "rule_mapping_result":   None,
        "math_validation_result": None,
        "mapping_result":        None,
        "validation_result":     None,
        # 제어
        "validation_mode":       "rule",
        "validation_target":     "rule",   # V2: llm_validator 초기 검증 대상
        "max_iterations":        max_iterations,
        "current_iteration":     0,
        # 출력
        "final_v5_data":         None,
        "status":                "pending",
        "logs":                  [],
    }

    return graph.invoke(initial_state)


def run_batch_standardization(
    pdf_dir: str = None,
    project_list: list = None,
    directory_df=None,
    max_iterations: int = 3,
    verbose: bool = True,
) -> list:
    """
    일괄 표준화 실행.

    두 가지 입력 모드:
        1. pdf_dir: 폴더 내 모든 PDF 자동 처리 (Scorecard_*.pdf)
        2. project_list: 직접 구성된 project dict 리스트

    Args:
        pdf_dir:       PDF 폴더 경로 (모드 1)
        project_list:  project dict 리스트 (모드 2)
        directory_df:  CSV DataFrame (반복 시 재사용 - I/O 절약)
        max_iterations: LLM 폴백 최대 반복
        verbose:       진행 상황 출력 여부

    Returns:
        list: 성공한 final_v5_data 딕셔너리 리스트
    """
    from pathlib import Path
    from src.data.loader import LEEDDataLoader

    # directory_df 1회만 로딩
    if directory_df is None:
        try:
            directory_df = LEEDDataLoader().load_project_directory()
        except Exception as e:
            print(f"[경고] CSV 로딩 실패 ({e}) - PDF 데이터만 사용")

    # 처리 대상 목록 구성
    if pdf_dir:
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        inputs = [{"pdf_path": str(p)} for p in pdf_files]
        total = len(pdf_files)
        if total == 0:
            print(f"[경고] PDF 없음: {pdf_dir}")
            return []
    elif project_list:
        inputs = [{"project_data": p} for p in project_list]
        total = len(project_list)
    else:
        print("[오류] pdf_dir 또는 project_list 중 하나를 지정하세요.")
        return []

    results = []
    track_counts = {"rule": 0, "llm": 0, "failed": 0}

    for i, inp in enumerate(inputs):
        label = inp.get("pdf_path", inp.get("project_data", {}).get("project_id", f"#{i+1}"))
        if verbose:
            print(f"[{i+1}/{total}] 처리 중: {label}")

        try:
            final_state = run_standardization(
                pdf_path=inp.get("pdf_path"),
                project_data=inp.get("project_data"),
                directory_df=directory_df,
                max_iterations=max_iterations,
            )

            if final_state.get("final_v5_data"):
                results.append(final_state["final_v5_data"])
                track = final_state.get("validation_mode", "rule")
                track_counts[track] = track_counts.get(track, 0) + 1

                if verbose:
                    v5 = final_state["final_v5_data"]["total_score_v5"]
                    iters = final_state.get("current_iteration", 0)
                    print(f"  완료 ({track} 경로) - v5={v5:.1f}, LLM반복={iters}")
            else:
                track_counts["failed"] += 1
                if verbose:
                    print(f"  실패: {final_state.get('status')}")

        except Exception as e:
            track_counts["failed"] += 1
            if verbose:
                print(f"  오류: {e}")

    print(
        f"\n일괄 처리 완료: {len(results)}/{total}개 성공 "
        f"(rule경로={track_counts['rule']}, "
        f"llm경로={track_counts.get('llm', 0)}, "
        f"실패={track_counts['failed']})"
    )
    return results
