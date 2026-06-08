# LEEDGRAPH

LangGraph 기반 LEED 인증 데이터 표준화 및 등급 결정요인 분석 파이프라인.
LEED v2.0~v4.1 의 인증 자료를 v5 공통 기준으로 정렬하고, 표준화된 점수에
SHAP 기반 변수 중요도 분석을 적용한다.

## Overview

한국 LEED 인증 건물 460건의 PDF 스코어카드를 LangGraph 워크플로우로 파싱·표준화하고,
RandomForest 분류기와 SHAP 분석으로 인증 등급(Certified·Silver·Gold·Platinum)을
가장 크게 결정하는 카테고리를 식별한다.

## Repository structure

```
LEEDGRAPH/
├── data/
│   ├── scorecards/                 원본 PDF 스코어카드 460건 + placeholder 5건
│   ├── rubrics/                    버전별 루브릭(xlsx) + mapping_rules.yaml (107 규칙)
│   ├── project_directory.csv       USGBC 프로젝트 디렉토리 (456 행)
│   ├── project_features.parquet    v5 표준화 결과 (460 행, 분석 입력)
│   └── standardized_credits.parquet 크레딧 단위 매핑 로그
├── notebooks/
│   ├── 01_전처리.ipynb              PDF → 표준화 parquet 생성
│   ├── 02_데이터분석.ipynb          EDA + 모델 비교 + SHAP
│   └── src/                        파이프라인 모듈 (data loader, langgraph_workflow)
├── scripts/                        표·그림 재생성 스크립트
├── results/
│   ├── tables/                     논문 표 CSV
│   └── figures/                    논문 그림 PNG
├── docs/                           전처리·파이프라인 상세 설명
└── requirements.txt
```

## Data

- 표본: 한국 LEED 인증 460건 (USGBC Public LEED Project Directory 한국 전수)
- 원천: 프로젝트 디렉토리 CSV 456 행 + 건물별 PDF 스코어카드 465 건 (실분석 460)
- 산출: `data/project_features.parquet` (460 행 × 28 컬럼, 규칙 기반 v5 표준화)
- 출처/라이선스: 스코어카드 PDF 는 USGBC Public LEED Project Directory 의 공개 자료.

## Pipeline

LangGraph 노드 구성 (`notebooks/src/langgraph_workflow/`).

```
Step 1  Input     460 PDF scorecards (v2.0 ~ v4.1)
Step 2  Parse     pdf_ingest + csv_match (pdfplumber + 정규식)
Step 3  Map       rule_mapper (107 매핑 규칙) + hallucination_checker (5 수학 제약)
                    math PASS  →  finalize
                    math FAIL  →  llm_mapper → llm_validator → finalize  (LLM 구제 경로)
Step 4  Output    project_features.parquet (v5 표준화 점수, N=460)
Step 5  Analyze   RandomForest + SHAP (5-Fold CV)
Step 6  Result    카테고리별 등급 결정 기여도
```

본 저장소에 동봉된 `data/project_features.parquet` 는 규칙 기반(rule-only) 산출본으로
LLM 호출 없이 재현 가능하다. LLM 구제 경로는 방법론상 구현되어 있으며 `OPENAI_API_KEY`
설정 시 활성화된다.

## Requirements

- Python 3.11+
- 주요 의존성: LangChain·LangGraph, pdfplumber, pandas, scikit-learn, xgboost, lightgbm, shap, matplotlib
- 전체 목록은 `requirements.txt`

```bash
pip install -r requirements.txt
```

LLM 검증 경로(선택)를 사용하려면 `.env` 에 `OPENAI_API_KEY` 를 설정한다.

## Usage

전처리·분석은 노트북 두 개로 진행한다.

```bash
# 1. 표준화 parquet 생성 (LLM 미사용 시 약 5분)
jupyter notebook notebooks/01_전처리.ipynb

# 2. EDA + 모델 비교 + SHAP
jupyter notebook notebooks/02_데이터분석.ipynb
```

표·그림 개별 재생성 스크립트는 `scripts/` 참고.

```bash
python scripts/regenerate_features.py             # parquet 재생성 (규칙 기반)
python scripts/generate_table_3_4_consistency.py  # 카테고리 합 일치 검증
python scripts/generate_table_3_4_drift.py        # 달성률 drift 표
python scripts/generate_table4_distribution.py    # 등급/버전 분포 표
python scripts/generate_pipeline_figure.py        # 파이프라인 다이어그램
```

## Results

`results/tables/` 와 `results/figures/` 에 저장된 산출물 기준.

- 표준화: 460 건 전수, 카테고리 합 일치율 100.0%, 평균 drift 11.55%p
- 분류 성능: RandomForest CV Accuracy 0.8457 ± 0.0081, weighted F1 0.8401 ± 0.0089
- 변수 중요도 (mean |SHAP|): ratio_EA > ratio_EQ > ratio_WE

## License

MIT License — `LICENSE` 참고.
