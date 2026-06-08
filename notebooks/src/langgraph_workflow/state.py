"""
LangGraph State 정의
- LEED 버전 표준화 워크플로우에서 공유되는 상태
"""

from typing import TypedDict, Annotated, Optional
import operator


class ProjectData(TypedDict):
    """개별 LEED 프로젝트 데이터"""
    project_id: str
    project_name: str
    version: str                    # 원본 버전 (v2.2, v3, v4 등)
    building_type: str
    gross_area_sqm: float
    certification_level: str        # 원본 등급
    categories: dict                # 원본 카테고리별 점수
    total_score_raw: float          # 원본 총점


class MappingResult(TypedDict):
    """Mapper Agent의 카테고리 매핑 결과"""
    mapped_categories: dict         # v5 기준으로 매핑된 카테고리 점수
    mapping_rationale: str          # 매핑 근거 설명
    proportional_scores: dict       # 비율 환산 점수
    total_score_v5: float           # v5 환산 총점


class ValidationResult(TypedDict):
    """Validator Agent의 검증 결과"""
    is_valid: bool                  # 검증 통과 여부
    validation_score: float         # 검증 품질 점수 (0~1)
    issues: list                    # 발견된 문제점 목록
    feedback: str                   # Mapper에게 전달할 피드백
    iteration: int                  # 현재 반복 횟수


class MathValidationResult(TypedDict):
    """수학적(규칙 기반) 검증 결과 - LLM 없이 순수 Python으로 계산"""
    passed: bool                    # 검증 통과 여부
    issues: list                    # 발견된 문제 목록
    achievement_ratio_original: float   # 원본 달성률 (achieved/max)
    achievement_ratio_v5: float         # v5 매핑 후 달성률
    ratio_drift: float              # 달성률 변화량 (절댓값)


class LEEDStandardizationState(TypedDict):
    """
    LangGraph 전체 워크플로우 공유 State.

    흐름 (1단계: 결정론적 경로 - LLM 없음):
        pdf_ingest → csv_match → rule_mapper → hallucination_checker
                                                    ↓ PASS
                                                 finalize → END

    흐름 (2단계: LLM 폴백 - hallucination_checker 실패 시):
        hallucination_checker FAIL → llm_mapper → llm_validator
                                         ↑              ↓ FAIL (반복)
                                         └──────────────┘ (최대 3회)
                                                       ↓ PASS
                                                    finalize → END

    설계 원칙:
        - rule_mapper: 수식 기반 결정론적 매핑, 토큰 소모 없음
        - hallucination_checker: 수학적 제약 검증, 토큰 소모 없음
        - llm_mapper/llm_validator: 규칙으로 해결 불가한 엣지케이스만 처리
    """
    # 입력: PDF 경로 또는 직접 project dict
    pdf_path: Optional[str]
    directory_df: Optional[object]      # pd.DataFrame (LangGraph 직렬화 제외)

    # PDF 파싱 결과
    parsed_pdf: Optional[dict]

    # CSV 매칭 결과
    matched_building: Optional[dict]

    # 입력 데이터 (csv_match_node에서 구성됨)
    project: Optional[ProjectData]

    # ── 결정론적 경로 ──────────────────────────────────────
    # rule_mapper_node 출력
    rule_mapping_result: Optional[MappingResult]

    # hallucination_checker_node 출력
    math_validation_result: Optional[MathValidationResult]

    # ── LLM 폴백 경로 ──────────────────────────────────────
    # llm_mapper_node 출력 (llm_validator도 이 값 사용)
    mapping_result: Optional[MappingResult]

    # llm_validator_node 출력
    validation_result: Optional[ValidationResult]

    # ── 공통 제어 ──────────────────────────────────────────
    # 현재 실행 경로: "rule" | "llm"
    # - 의미: 최종 finalize에서 어느 결과를 채택할지 결정 (rule_mapping_result vs mapping_result)
    # - 갱신 시점: rule_mapper_node에서 "rule"로 세팅, llm_mapper_node에서 "llm"으로 변경
    validation_mode: str

    # LLM validator가 현재 검증 중인 대상: "rule" | "llm"
    # - 의미: llm_validator_node가 어느 매핑을 검증하고 있는지
    #   "rule" → rule_mapping_result 검증 (Phase 2 신규)
    #   "llm"  → mapping_result 검증 (기존 LLM 재매핑 결과)
    # - 갱신 시점: 초기값 "rule", llm_mapper_node 거치면 "llm"으로 전환
    validation_target: str

    # LLM 경로 반복 횟수 (rule 경로는 반복 없음)
    max_iterations: int
    current_iteration: int

    # 최종 결과
    final_v5_data: Optional[dict]
    status: str                     # "pending" | "completed" | "failed"

    # 로그 (Annotated → append 방식으로 누적)
    logs: Annotated[list, operator.add]
