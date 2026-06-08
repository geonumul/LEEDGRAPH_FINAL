"""
LEED 데이터 로더
- USGBC PublicLEEDProjectDirectory.xlsx 로딩
- 개별 Scorecard PDF 파싱
"""

import os
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────
# LEED 버전별 카테고리 최대 점수 정의
# ─────────────────────────────────────────────────────────
LEED_VERSION_MAX_SCORES = {
    "v1.0 pilot": {
        "SS": 14,
        "WE": 5,
        "EA": 17,
        "MR": 13,
        "IEQ": 15,
        "IN": 5,
        "TOTAL": 69,
    },
    "v2.0": {
        "SS": 14,
        "WE": 5,
        "EA": 17,
        "MR": 13,
        "IEQ": 15,
        "IN": 5,
        "TOTAL": 69,
    },
    "v2.2": {
        "SS": 14,   # Sustainable Sites
        "WE": 5,    # Water Efficiency
        "EA": 17,   # Energy & Atmosphere
        "MR": 13,   # Materials & Resources
        "IEQ": 15,  # Indoor Environmental Quality
        "IN": 5,    # Innovation & Design
        "TOTAL": 69,
    },
    "v2009": {      # LEED 2009 = v3
        "SS": 26,
        "WE": 10,
        "EA": 35,
        "MR": 14,
        "IEQ": 15,
        "IN": 6,
        "RP": 4,
        "TOTAL": 110,
    },
    "v3": {
        "SS": 26,
        "WE": 10,
        "EA": 35,
        "MR": 14,
        "IEQ": 15,
        "IN": 6,
        "RP": 4,    # Regional Priority
        "TOTAL": 110,
    },
    "v4": {
        "LT": 16,   # Location & Transportation
        "SS": 10,
        "WE": 11,
        "EA": 33,
        "MR": 13,
        "IEQ": 16,
        "IN": 6,
        "RP": 4,
        "IP": 2,    # Integrative Process
        "TOTAL": 110,
    },
    "v4.1": {
        "LT": 16,
        "SS": 10,
        "WE": 12,
        "EA": 33,
        "MR": 13,
        "IEQ": 16,
        "IN": 6,
        "RP": 4,
        "IP": 2,
        "TOTAL": 110,
    },
    "v5": {
        "LT": 16,
        "SS": 10,
        "WE": 12,
        "EA": 33,
        "MR": 13,
        "IEQ": 16,
        "IN": 6,
        "RP": 4,
        "IP": 2,
        "TOTAL": 110,
    },
}

# LEED 등급 기준 (v4/v5 기준)
LEED_GRADE_THRESHOLDS = {
    "Certified": (40, 49),
    "Silver": (50, 59),
    "Gold": (60, 79),
    "Platinum": (80, 110),
}


class LEEDDataLoader:
    """LEED 프로젝트 데이터 로더"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ─────────────────────────────────────────────
    # 1. PublicLEEDProjectDirectory 로딩
    # ─────────────────────────────────────────────
    def load_project_directory(
        self, filename: str = "project_directory.csv"
    ) -> pd.DataFrame:
        """
        USGBC 공개 프로젝트 디렉토리 로딩.

        주요 컬럼:
            - ID, ProjectName, Country, LEEDSystemVersion
            - CertLevel, PointsAchieved, CertDate, GrossFloorArea
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            for ext in (".csv", ".xlsx"):
                alt = filepath.with_suffix(ext)
                if alt.exists():
                    filepath = alt
                    break
            else:
                raise FileNotFoundError(
                    f"파일을 찾을 수 없습니다: {filepath}\n"
                    "USGBC에서 데이터를 다운로드하여 data/ 폴더에 저장해주세요."
                )

        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, engine="openpyxl")

        print(f"프로젝트 디렉토리 로딩 완료: {len(df)}개 프로젝트")
        return df

    def load_korea_projects(
        self, filename: str = "PublicLEEDProjectDirectory.csv"
    ) -> pd.DataFrame:
        """한국 LEED 프로젝트만 필터링 (CSV가 이미 한국 전용이면 전체 반환)"""
        df = self.load_project_directory(filename)

        if "Country" in df.columns:
            korea_mask = df["Country"].str.contains("Korea|KR|한국", case=False, na=False)
            korea_df = df[korea_mask].copy()
            if len(korea_df) > 0:
                print(f"한국 프로젝트 필터링 완료: {len(korea_df)}개")
                return korea_df

        # 이미 한국 전용 데이터인 경우
        print(f"전체 데이터 사용: {len(df)}개")
        return df

    def match_scorecard_to_directory(
        self, parsed_pdf: dict, directory_df: pd.DataFrame
    ) -> Optional[dict]:
        """
        파싱된 Scorecard PDF를 프로젝트 디렉토리 CSV와 매칭.

        매칭 우선순위:
            1. project_id 정확 매칭 (가장 신뢰)
            2. 건물명 유사도 매칭 (fallback)

        Returns:
            dict: 매칭된 행 정보 또는 None
        """
        id_col = "ID" if "ID" in directory_df.columns else directory_df.columns[0]
        name_col = "ProjectName" if "ProjectName" in directory_df.columns else None

        # 1순위: project_id 매칭
        pid = str(parsed_pdf.get("project_id", "")).strip()
        if pid:
            match = directory_df[directory_df[id_col].astype(str).str.strip() == pid]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["_match_method"] = "id"
                return row

        # 2순위: 건물명 매칭 (소문자 포함 비교)
        if name_col and parsed_pdf.get("project_name"):
            pdf_name = parsed_pdf["project_name"].strip().lower()
            name_match = directory_df[
                directory_df[name_col].str.lower().str.strip() == pdf_name
            ]
            if not name_match.empty:
                row = name_match.iloc[0].to_dict()
                row["_match_method"] = "name"
                return row

        return None

    # ─────────────────────────────────────────────
    # 2. Scorecard PDF 파싱
    # ─────────────────────────────────────────────
    def parse_scorecard_pdf(self, pdf_path: str) -> dict:
        """
        LEED Scorecard PDF에서 카테고리별 점수 추출.

        Returns:
            dict: {
                "project_id": str,
                "project_name": str,
                "location": str,
                "leed_system": str,
                "version": str,
                "certification_level": str,
                "award_date": str,
                "total_awarded": int,
                "total_possible": int,
                "total_score": int,
                "categories": {
                    "SS": {"awarded": int, "possible": int}, ...
                },
                "credits": {
                    "Credit: Site Mgmt": {"awarded": int, "possible": int}, ...
                },
            }
        """
        result = {
            "project_id": "",
            "project_name": "",
            "location": "",
            "leed_system": "",
            "version": "unknown",
            "certification_level": "",
            "award_date": "",
            "total_awarded": 0,
            "total_possible": 0,
            "total_score": 0,
            "categories": {},
            "credits": {},
            "raw_text": "",
        }

        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"

            result["raw_text"] = full_text
            result.update(self._extract_scorecard_info(full_text))

        return result

    def _fix_doubled_chars(self, text: str) -> str:
        """USGBC PDF 스코어카드의 이중 문자 인코딩 수정.

        두 단계로 처리:
        - 1단계: 대문자 연속 doubled (2쌍 이상) → SSUUSSTTAAIINNAABBLLEE → SUSTAINABLE
          (1쌍만인 EE 등은 건드리지 않아 'LEED' 보존)
        - 2단계: 숫자/특수문자 doubled (1쌍 이상) → :: → :, 22 → 2, // → /
        """
        def fix(m):
            s = m.group(0)
            return "".join(s[i] for i in range(0, len(s), 2))

        text = re.sub(r"(?:([A-Z])\1){2,}", fix, text)
        text = re.sub(r"(?:([0-9:/&])\1)+", fix, text)
        return text

    def _extract_scorecard_info(self, text: str) -> dict:
        """Scorecard 텍스트에서 정보 추출 (USGBC 형식)"""
        info = {
            "project_id": "",
            "project_name": "",
            "location": "",
            "leed_system": "",
            "version": "unknown",
            "certification_level": "",
            "award_date": "",
            "total_awarded": 0,
            "total_possible": 0,
            "total_score": 0,
            "categories": {},
            "credits": {},
        }

        cleaned = self._fix_doubled_chars(text)

        # 버전 추출: "(v4)", "(v4.1)", "(v2009)" 등
        # USGBC 스코어카드 PDF에서 버전은 "LEED O+M: ... (v4)" 형태로 표기됨
        ver_match = re.search(r"\(v([\d.]+|2009|2008)\)", text, re.IGNORECASE)
        if ver_match:
            ver = ver_match.group(1)
            if ver == "4.1":
                info["version"] = "v4.1"
            elif ver == "4":
                info["version"] = "v4"
            elif ver == "5":
                info["version"] = "v5"
            elif ver == "2009" or ver == "2008":
                # LEED 2009 = v3 구조와 동일 (SS에 교통 항목 포함)
                info["version"] = "v2009"
            elif ver.startswith("3"):
                info["version"] = "v3"
            elif ver == "2.2":
                info["version"] = "v2.2"
            elif ver == "2.0":
                info["version"] = "v2.0"
            elif ver in ("1.0", "1"):
                info["version"] = "v1.0 pilot"

        # LEED 시스템명 추출
        sys_match = re.search(r"(LEED\s+[\w+\s:&/,-]+?\(v[\d.]+\))", text, re.IGNORECASE)
        if sys_match:
            info["leed_system"] = sys_match.group(1).strip()

        # 인증 등급 추출
        grade_match = re.search(r"\b(PLATINUM|GOLD|SILVER|CERTIFIED)\b", text, re.IGNORECASE)
        if grade_match:
            info["certification_level"] = grade_match.group(1).capitalize()

        # 수상 날짜 추출
        date_match = re.search(
            r"AWARDED\s+((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4})",
            text, re.IGNORECASE,
        )
        if date_match:
            info["award_date"] = date_match.group(1)

        # 프로젝트 ID 및 위치 추출: "1000182052, Gyeonggi-do"
        id_match = re.search(r"^(\d{7,}),\s*(.+)$", text, re.MULTILINE)
        if id_match:
            info["project_id"] = id_match.group(1)
            info["location"] = id_match.group(2).strip()
            # 건물명은 프로젝트 ID 줄 이후 첫 번째 비어있지 않은 줄
            rest = text[id_match.end():]
            name_match = re.search(r"^\s*([A-Za-z].{2,80})$", rest, re.MULTILINE)
            if name_match:
                info["project_name"] = name_match.group(1).strip()

        # 총점 추출 (cleaned 텍스트에서 "TOTAL 64 / 110")
        total_match = re.search(r"TOTAL\s+(\d+)\s*/\s*(\d+)", cleaned, re.IGNORECASE)
        if total_match:
            info["total_awarded"] = int(total_match.group(1))
            info["total_possible"] = int(total_match.group(2))
            info["total_score"] = int(total_match.group(1))

        # 카테고리별 점수 추출 (cleaned 텍스트에서 "CATEGORY AWARDED: X / Y")
        # 주의: LEED 버전/시스템(BD+C, O+M 등)마다 헤더 표기가 다를 수 있음
        # → 핵심 키워드만 매칭하고 중간 단어는 [\w\s&]* 로 허용
        cat_patterns = {
            "SS": r"SUSTAINABLE\s+SITES?\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            "WE": r"WATER\s+EFFICIENCY\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            # ENERGY & ATMOSPHERE or ENERGY AND ATMOSPHERE
            "EA": r"ENERGY[\w\s&]+ATMOSPHERE\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            # MATERIALS & RESOURCES or MATERIAL & RESOURCES
            "MR": r"MATERIALS?[\w\s&]*RESOURCES?\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            # IEQ 버그 수정: "INDOOR ENVIRONMENTAL QUALITY AWARDED"에서
            # 원래 패턴(\w*가 스페이스 불가)이 QUALITY를 못 잡아 IEQ 미추출됨
            # → [\w\s]+ 로 "ENVIRONMENTAL QUALITY" 전체를 포괄적으로 매칭
            "IEQ": r"INDOOR\s+ENVIRONMENTAL[\w\s]+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            "IN": r"INNOVATION\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            # REGIONAL PRIORITY CREDITS AWARDED (버전마다 중간 단어 다름)
            "RP": r"REGIONAL\s+PRIORITY[\w\s]*AWARDED:\s*(\d+)\s*/\s*(\d+)",
            "LT": r"LOCATION[\w\s&]+TRANSPORTATION\s+AWARDED:\s*(\d+)\s*/\s*(\d+)",
            # INTEGRATIVE PROCESS CREDITS AWARDED (헤더에 "CREDITS"가 끼는 변형)
            # → RP와 동일하게 [\w\s]* 로 중간 단어 허용 (미허용 시 IP 통째 누락)
            "IP": r"INTEGRATIVE\s+PROCESS[\w\s]*AWARDED:\s*(\d+)\s*/\s*(\d+)",
        }

        categories = {}
        for cat, pattern in cat_patterns.items():
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                categories[cat] = {
                    "awarded": int(m.group(1)),
                    "possible": int(m.group(2)),
                }
        info["categories"] = categories

        # 개별 크레딧 점수 추출 (원본 텍스트에서)
        credit_pattern = r"^(Prereq|Credit)\s+(.+?)\s+(\d+)\s*/\s*(\d+)\s*$"
        credits = {}
        for m in re.finditer(credit_pattern, text, re.MULTILINE):
            key = f"{m.group(1)}: {m.group(2).strip()}"
            credits[key] = {
                "awarded": int(m.group(3)),
                "possible": int(m.group(4)),
            }
        info["credits"] = credits

        return info

    def load_scorecard_batch(self, pdf_dir: str) -> pd.DataFrame:
        """
        폴더 내 모든 Scorecard PDF를 일괄 파싱.

        Args:
            pdf_dir: PDF 파일들이 있는 디렉토리 경로

        Returns:
            DataFrame: 전체 파싱 결과
        """
        pdf_path = Path(pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            print(f"[경고] PDF 파일을 찾을 수 없습니다: {pdf_dir}")
            return pd.DataFrame()

        records = []
        for pdf_file in pdf_files:
            try:
                parsed = self.parse_scorecard_pdf(str(pdf_file))
                flat = {
                    "file_name": pdf_file.name,
                    "project_id": parsed["project_id"],
                    "project_name": parsed["project_name"],
                    "location": parsed["location"],
                    "leed_system": parsed["leed_system"],
                    "version": parsed["version"],
                    "certification_level": parsed["certification_level"],
                    "award_date": parsed["award_date"],
                    "total_awarded": parsed["total_awarded"],
                    "total_possible": parsed["total_possible"],
                }
                for cat, scores in parsed["categories"].items():
                    flat[f"{cat}_awarded"] = scores["awarded"]
                    flat[f"{cat}_possible"] = scores["possible"]
                records.append(flat)
                print(f"  완료: {pdf_file.name}")
            except Exception as e:
                print(f"  오류: {pdf_file.name}: {e}")

        df = pd.DataFrame(records)
        print(f"\n총 {len(df)}개 Scorecard 파싱 완료")
        return df

    @staticmethod
    def create_sample_data() -> pd.DataFrame:
        """
        실제 데이터가 없을 때 테스트용 샘플 데이터 생성.
        실제 LEED 한국 프로젝트 통계를 반영한 더미 데이터.
        """
        import numpy as np

        np.random.seed(42)
        n = 451

        # 버전 분포 (한국 LEED 인증 현황 반영)
        versions = np.random.choice(
            ["v2.2", "v3", "v4", "v4.1"],
            size=n,
            p=[0.05, 0.30, 0.50, 0.15],
        )

        records = []
        for i, ver in enumerate(versions):
            max_scores = LEED_VERSION_MAX_SCORES[ver]

            # 카테고리별 점수 생성 (정규분포 기반, 최대값 범위 내)
            record = {
                "project_id": f"KR-{i+1:04d}",
                "version": ver,
                "building_type": np.random.choice(
                    ["Office", "Commercial", "Residential", "Mixed-Use", "Industrial"],
                    p=[0.40, 0.25, 0.15, 0.15, 0.05],
                ),
                "gross_area_sqm": np.random.lognormal(mean=10.2, sigma=0.8),
            }

            # 버전별 카테고리 점수 생성
            for cat, max_pt in max_scores.items():
                if cat == "TOTAL":
                    continue
                achieved_ratio = np.random.beta(a=3, b=2)  # 60~80% 달성 경향
                record[f"score_{cat}"] = min(
                    int(max_pt * achieved_ratio), max_pt
                )

            # 총점 계산
            score_cols = [k for k in record if k.startswith("score_")]
            record["total_score_raw"] = sum(record[k] for k in score_cols)

            # 등급 결정
            for grade, (lo, hi) in LEED_GRADE_THRESHOLDS.items():
                # 총점을 100점 기준으로 환산 후 등급 결정
                normalized = (
                    record["total_score_raw"] / max_scores["TOTAL"] * 110
                )
                if lo <= normalized <= hi:
                    record["certification_level"] = grade
                    break
            else:
                record["certification_level"] = (
                    "Platinum" if normalized > 80 else "Certified"
                )

            records.append(record)

        df = pd.DataFrame(records)
        print(f"샘플 데이터 생성 완료: {len(df)}개 프로젝트")
        return df
