"""외국인 사장님 금융 근거 API.

실행 (F1 폴더에서):  uvicorn api.main:app --reload
문서:               http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M  # noqa: E402
from src import refresh as R  # noqa: E402

app = FastAPI(
    title="외국인 사장님 금융 근거 API",
    version="0.1.0",
    description="케이뱅크·카카오뱅크 '사장님을 위한 금융 아이디어 공모전' — 타겟 수·수익·상환능력·정착 근거를 계산한다.",
)


def _clean(obj):
    """NaN/inf를 JSON에서 null로 바꾼다."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- Q1
@app.get("/target/counts", tags=["Q1 타겟 수"], summary="하한·중심·상한·예비 타겟 인원")
def target_counts():
    return _clean(M.target_counts())


@app.get("/target/projection", tags=["Q1 타겟 수"], summary="중심 추정(외국인 자영업자) 연도별 전망")
def target_projection(year: int = Query(2028, ge=2025, le=2035)):
    return {"year": year, "owners": M.project_core(year), "cagr": M.core_cagr(),
            "owners_range": {m: M.project_core(year, m) for m in ("endpoint", "recent3", "trend")}}


@app.get("/target/trend", tags=["Q1 타겟 수"], summary="외국인 자영업자 수 2012년부터 연도별 (KOSIS)")
def target_trend():
    return {"series": M.self_employed_series(), "cagr": M.core_cagr_range()}


@app.get("/target/profile", tags=["Q1 타겟 수"], summary="사장님 프로필: 국적·업종·매출·개업 시기 (카드 가맹점 10,323곳)")
def target_profile():
    return M.owner_profile()


@app.get("/target/latent", tags=["Q1 타겟 수"], summary="잠재 창업 수요(자영업률 격차)")
def target_latent(benchmark_rate: float = Query(M.KR_SELF_EMP_RATE, gt=0, lt=1)):
    return M.latent_startup_demand(benchmark_rate)


@app.get("/target/regions", tags=["Q1 타겟 수"], summary="사업 가능 외국인 상위 시군구")
def target_regions(n: int = Query(10, ge=1, le=228)):
    return M.top_regions(n)


# ---------------------------------------------------------------- Q2
class RevenueBody(BaseModel):
    target_year: int = Field(2028, ge=2025, le=2035)
    penetration: float = Field(0.05, ge=0, le=1, description="대출 침투율")
    avg_loan_won: float = Field(30_000_000, gt=0)
    spread: float = Field(0.035, ge=0, le=0.3, description="예대마진")
    credit_loss: float = Field(0.010, ge=0, le=0.3, description="대손비용률")
    remit_penetration: float = Field(0.05, ge=0, le=1)
    remit_fee: float = Field(0.010, ge=0, le=0.1)


@app.get("/revenue/scenarios", tags=["Q2 수익"], summary="보수·기준·낙관 시나리오")
def revenue_scenarios():
    return {name: M.revenue(inp) for name, inp in M.SCENARIOS.items()}


@app.post("/revenue/simulate", tags=["Q2 수익"], summary="가정값을 바꿔 연 수익 계산")
def revenue_simulate(body: RevenueBody):
    return M.revenue(M.RevenueInput(**body.model_dump()))


@app.get("/assumptions/evidence", tags=["Q2 수익"], summary="가정값(대출액·마진·대손율·금리·수수료)과 공개 자료 비교")
def assumptions_evidence():
    return _clean(M.assumption_evidence())


@app.get("/assumptions/target-delinquency", tags=["Q2 수익"], summary="타겟 구성으로 가중한 개인사업자 연체율 (대손율 가정 근거)")
def assumptions_target_delinquency():
    return _clean(M.target_weighted_delinquency())


@app.get("/market/funnel", tags=["Q2 수익"], summary="대출·송금 시장 퍼널 (수익 시나리오와 같은 숫자)")
def market_funnel(scenario: str = Query("기준", pattern="^(보수|기준|낙관)$")):
    return _clean(M.market_funnel(scenario))


@app.get("/market/remittance-savings", tags=["Q2 수익"], summary="송금 고객 1인당 연 절감액 (비교 대상별)")
def market_remittance_savings(our_fee: float = Query(None, ge=0, le=0.1, description="우리 수수료율. 비우면 기준 시나리오 값")):
    return _clean(M.remittance_savings(our_fee))


@app.get("/target/province-conversion", tags=["Q1 타겟 수"], summary="시도별 예비 사장님 1만 명당 외국인 카드 가맹점 수")
def target_province_conversion():
    return M.province_conversion()


@app.get("/settlement/visa-change", tags=["Q4 정착"], summary="체류자격을 바꾼 외국인 비율 추이 (전체·체류자격별)")
def settlement_visa_change():
    return _clean(M.visa_change_trend())


@app.get("/draft/settlement-pipeline", tags=["최종 초안"], summary="정착 사다리: KIIP 참여·영주·귀화 추이")
def draft_settlement_pipeline():
    return _clean(M.settlement_pipeline())


@app.get("/draft/repayment", tags=["최종 초안"], summary="비자 스텝업: 단계별 금리의 DSR 통과 비율과 고객 연 이자")
def draft_repayment():
    return _clean(M.draft_repayment())


@app.get("/draft/rate-benchmarks", tags=["최종 초안"], summary="초안 금리와 시장 금리 비교")
def draft_rate_benchmarks():
    return _clean(M.rate_benchmarks())


@app.get("/draft/mentor-pool", tags=["최종 초안"], summary="선배 귀화인 멘토 후보: 시군구별 국적 취득 5년+ 주민")
def draft_mentor_pool(top: int = Query(15, ge=1, le=60)):
    return _clean(M.mentor_pool(top))


@app.get("/draft/unit-economics", tags=["최종 초안"], summary="두 상품의 단위경제성 (운영비 차감 전)")
def draft_unit_economics():
    return _clean(M.draft_unit_economics())


# ---------------------------------------------------------------- Q3
@app.get("/repayment/dsr", tags=["Q3 상환능력"], summary="대출 조건별 DSR 통과 비율(체류자격별)")
def repayment_dsr(
    loan_won: float = Query(30_000_000, gt=0),
    annual_rate: float = Query(0.06, ge=0, le=0.3),
    months: int = Query(60, ge=1, le=360),
    dsr: float = Query(0.40, gt=0, le=1),
):
    return M.dsr_pass(loan_won, annual_rate, months, dsr)


# ---------------------------------------------------------------- Q4
@app.get("/settlement/region", tags=["Q4 정착"], summary="시군구: 사업 가능 외국인 비중 ↔ 장기체류·귀화 상관")
def settlement_region():
    try:
        return _clean(M.settlement_corr())
    except FileNotFoundError:
        raise HTTPException(503, "notebooks/target_evidence.ipynb를 먼저 실행해 settlement_corr_region.csv를 만든다")


@app.get("/settlement/visa-groups", tags=["Q4 정착"], summary="가족정착형 vs 취업·학업형 비율차 검정")
def settlement_visa_groups():
    return _clean(M.visa_group_tests())


# ---------------------------------------------------------------- 시장 지표·데이터 갱신
@app.get("/market/fx", tags=["시장 지표"], summary="최근 환율(원/달러·원/위안)과 변동폭")
def market_fx(months: int = Query(12, ge=2, le=130)):
    return _clean(M.fx_recent(months))


@app.get("/market/remittances", tags=["시장 지표"], summary="한국발 개인송금 지급액(내국인 포함 전체)")
def market_remittances(years: int = Query(10, ge=1, le=66)):
    return M.remittances_paid(years)


@app.get("/data/status", tags=["데이터 갱신"], summary="자료별 최신 시점과 자동 갱신 여부")
def data_status():
    return R.status()


@app.post("/data/refresh", tags=["데이터 갱신"], summary="공개 원자료 새로 받아오기 (인터넷 필요, 수십 초 걸린다)")
def data_refresh(sources: list[str] | None = Query(None, description="ecos, worldbank, moj 중 선택. 비우면 전부")):
    try:
        return R.refresh_all(sources)
    except ValueError as e:
        raise HTTPException(422, str(e))
