"""공모전 근거 지표 계산 모듈 — 노트북(target_evidence)과 API(api/main.py)가 같은 함수를 쓴다.

입력은 data/processed, data/curated의 CSV다. 시장 지표(환율·송금)만 data/raw의 최신 파일을 직접 읽는다.
모든 금액 단위는 원(KRW), 인원은 명이다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CUR = ROOT / "data" / "curated"

# 외부 비교값·가정 기본값 (근거는 notebooks/target_evidence 참조)
KR_SELF_EMP_RATE = 0.196            # 국가데이터처 2025.8 비임금근로 부가조사: 자영업자/취업자
CARD_MERCHANTS = 10_323              # 이민정책연구원 통계브리프 2025-14 (신한카드 외국인 개인사업자 가맹점)
REMIT_BIN_MID_MAN = [50, 150, 250, 400, 750, 1250, 1750, 2500]  # 송금액 구간 대표값(만원)
INCOME_BINS = [("inc_lt100", 0, 100), ("inc_100_200", 100, 200), ("inc_200_300", 200, 300), ("inc_300p", 300, None)]


# ---------------------------------------------------------------- 로딩
@lru_cache(maxsize=None)
def _survey() -> pd.DataFrame:
    from . import load as L
    return L.load_survey()   # KOSIS 통계표(없으면 손으로 옮긴 CSV)


def _sv(table: str, year: int | None = None) -> pd.DataFrame:
    d = _survey()[_survey().table == table]
    return d if year is None else d[d.year == year]


@lru_cache(maxsize=None)
def _employment() -> pd.DataFrame:
    return _sv("employment_status").pivot(index="year", columns="category", values="value")  # 천 명


@lru_cache(maxsize=None)
def _visa() -> pd.DataFrame:
    return pd.read_csv(PROC / "visa_indicators_2025.csv")


@lru_cache(maxsize=None)
def _nts() -> pd.DataFrame:
    return pd.read_csv(PROC / "nts_foreign_income_tax.csv")


# ---------------------------------------------------------------- Q1 타겟 수
@lru_cache(maxsize=None)
def _self_emp() -> pd.DataFrame:
    """외국인 자영업자 시계열 2012~ (KOSIS DT_2FB007F). 보도자료(천 명 반올림)보다 한 자리 더 정밀하다."""
    from . import load as L
    return L.load_self_employed_series().set_index("year")


def core_self_employed(year: int = 2025) -> float:
    return float(_self_emp().total[year])


def core_cagr_range() -> dict:
    """연평균 증가율 세 가지. 표본조사라 해마다 출렁이므로 계산법에 따라 값이 조금 다르다.

    trend    전체 기간(2012~) 로그선형 추세 — 모든 해를 쓰고 시작·끝 해에 덜 민감하다 (대표값)
    endpoint 첫 해→마지막 해 단순 CAGR
    recent3  최근 3년(2022→2025) CAGR
    """
    s = _self_emp().total
    y0, y1 = int(s.index.min()), int(s.index.max())
    slope = np.polyfit(s.index.values, np.log(s.values), 1)[0]
    return {"trend": float(np.exp(slope) - 1),
            "endpoint": float((s[y1] / s[y0]) ** (1 / (y1 - y0)) - 1),
            "recent3": float((s[y1] / s[y1 - 3]) ** (1 / 3) - 1),
            "period": f"{y0}~{y1}"}


def core_cagr() -> float:
    return core_cagr_range()["trend"]


def project_core(year: int, method: str = "trend") -> float:
    """2025 중심 추정을 연평균 증가율(기본: 2012~2025 추세)로 연장."""
    return core_self_employed(2025) * (1 + core_cagr_range()[method]) ** (year - 2025)


def self_employed_series() -> list[dict]:
    d = _self_emp().reset_index()
    return [{"year": int(r.year), "employer": float(r.employer), "solo": float(r.solo), "total": float(r.total),
             "share_of_employed_pct": float(r.total / r.employed * 100)} for r in d.itertuples()]


def target_counts() -> dict:
    nts = _nts().set_index("year")
    vts = pd.read_csv(PROC / "visa_group_timeseries.csv", parse_dates=["date"]).set_index("date")
    pool = vts["사업 가능 체류자격 합계"]
    yrs = (pool.index[-1] - pool.index[0]).days / 365.25
    return {
        "lower": {"label": "신한카드 외국인 개인사업자 가맹점", "count": CARD_MERCHANTS, "as_of": "2025 브리프"},
        "core": {"label": "외국인 자영업자(고용주+1인)", "count": core_self_employed(), "as_of": "2025.5", "cagr": core_cagr(),
                 "cagr_range": core_cagr_range()},
        "core_employer": {"label": "직원을 둔 외국인 사장님", "count": float(_self_emp().employer[2025]), "as_of": "2025.5"},
        "upper": {"label": "외국인 종합소득세 신고자", "count": float(nts.filers[2024]), "as_of": "2024",
                  "cagr": (nts.filers[2024] / nts.filers[2014]) ** 0.1 - 1},
        "pool": {"label": "사업 가능 체류자격 체류자", "count": float(pool.iloc[-1]), "as_of": "2026.9",
                 "cagr": (pool.iloc[-1] / pool.iloc[0]) ** (1 / yrs) - 1},
        "core_2028": project_core(2028),
        "core_2028_range": {m: project_core(2028, m) for m in ("endpoint", "recent3", "trend")},
    }


def latent_startup_demand(benchmark_rate: float = KR_SELF_EMP_RATE) -> dict:
    emp = _sv("employed_by_visa", 2025).set_index("category").value * 1e3
    elig = float(emp[["재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)"]].sum())
    rate = core_self_employed() / elig
    return {"eligible_employed": elig, "foreign_rate_upper": rate, "benchmark_rate": benchmark_rate,
            "latent_owners": max(0.0, (benchmark_rate - rate) * elig)}


def top_regions(n: int = 10) -> list[dict]:
    reg = pd.read_csv(PROC / "region_business_eligible.csv").sort_values("biz", ascending=False).reset_index(drop=True)
    reg["cum_share"] = reg.biz.cumsum() / reg.biz.sum()
    out = reg.head(n)[["label", "biz", "총합계", "LQ", "cum_share"]].rename(columns={"biz": "business_eligible", "총합계": "foreign_total"})
    return out.to_dict(orient="records")


def owner_profile() -> dict:
    """외국인 사장님 프로필 — 신한카드 외국인 개인사업자 가맹점 10,323곳(이민정책연구원 통계브리프 No. 2025-14).

    조사(KOSIS)에는 사장님의 국적·업종·매출이 없어 카드 가맹점 자료로 본다. 한 카드사 기준이고 체류자격 정보가 없다.
    """
    from . import load as L
    b = L.load_mrtc_brief()

    def tab(name):
        return b[b.table == name]

    def shares(d):
        tot = d.value.sum()
        return [{"name": r.row, "count": int(r.value), "share_pct": float(r.value / tot * 100)}
                for r in d.sort_values("value", ascending=False).itertuples()]

    reg = tab("registration").pivot(index="row", columns="col", values="value")
    rev = tab("revenue_industry").pivot(index="row", columns="col", values="value")
    order = ["3억 원 이하", "3억~5억 원", "5억~10억 원", "10억~30억 원", "30억 원 초과"]
    rev = rev.loc[order]
    total = float(reg.values.sum())
    nat_reg = tab("nationality_registration")
    months = nat_reg[nat_reg.col == "평균 영업기간"].set_index("row").value
    return {
        "source": "이민정책연구원 통계브리프 No. 2025-14 (신한카드, 2025년 6월 말 기준)",
        "merchants": int(total), "active": int(reg["가맹유지"].sum()), "closed": int(reg["가맹해지"].sum()),
        "closed_share_pct": float(reg["가맹해지"].sum() / total * 100),
        "nationality": shares(tab("nationality")),
        "industry": shares(tab("industry")),
        "revenue_bands": [{"band": k, "count": int(v), "share_pct": float(v / total * 100)} for k, v in rev.sum(axis=1).items()],
        "small_share_by_industry_pct": {c: float(rev.loc["3억 원 이하", c] / rev[c].sum() * 100) for c in rev.columns},
        "cohorts": [{"registered": k, "count": int(r.sum()), "share_pct": float(r.sum() / total * 100),
                     "still_active_pct": float(r["가맹유지"] / r.sum() * 100)} for k, r in reg.iterrows()],
        "avg_months_in_business": {k: float(v) for k, v in months.items()},
        "top_regions": [{"name": r.row, "count": int(r.value)} for r in
                        tab("top_region")[tab("top_region").col == "전체 가맹점"].itertuples()],
        "capital_area_share_pct": float(tab("region_industry").query("row in ['서울', '인천', '경기']").value.sum() / total * 100),
    }


# ---------------------------------------------------------------- Q2 수익
def mean_annual_remittance() -> float:
    ra = _sv("remit_amount", 2025).value.values / 100
    return float((ra * np.array(REMIT_BIN_MID_MAN)).sum() * 1e4)


def remit_senders() -> float:
    pop15 = _sv("population", 2025).value.iloc[0] * 1e3
    return float(pop15 * _sv("remit_freq", 2025).set_index("category").loc["송금함", "value"] / 100)


@dataclass
class RevenueInput:
    target_year: int = 2028
    penetration: float = 0.05          # 대출 침투율
    avg_loan_won: float = 30_000_000
    spread: float = 0.035              # 예대마진
    credit_loss: float = 0.010         # 대손비용률
    remit_penetration: float = 0.05
    remit_fee: float = 0.010


def revenue(inp: RevenueInput) -> dict:
    n = core_self_employed() if inp.target_year <= 2025 else project_core(inp.target_year)
    borrowers = n * inp.penetration
    book = borrowers * inp.avg_loan_won
    loan_rev = book * (inp.spread - inp.credit_loss)
    users = remit_senders() * inp.remit_penetration
    remit_rev = users * mean_annual_remittance() * inp.remit_fee
    return {"input": asdict(inp), "target_owners": n, "borrowers": borrowers, "loan_book_won": book,
            "loan_revenue_won": loan_rev, "remit_users": users, "remit_revenue_won": remit_rev,
            "total_revenue_won": loan_rev + remit_rev}


SCENARIOS = {
    "보수": RevenueInput(2025, 0.03, 20_000_000, 0.030, 0.015, 0.03, 0.005),
    "기준": RevenueInput(2028, 0.05, 30_000_000, 0.035, 0.010, 0.05, 0.010),
    "낙관": RevenueInput(2028, 0.10, 50_000_000, 0.040, 0.007, 0.10, 0.015),
}


# ---------------------------------------------------------------- 가정값 근거
def assumption_evidence() -> dict:
    """수익 시나리오·DSR 가정값을 공개 자료와 나란히 놓는다. 가정값을 바꾸지는 않는다(팀이 정한다).

    비교 대상은 내국인 포함 전체 개인사업자·은행권 자료다. 외국인 사장님만의 금리·연체율 자료는 없다.
    """
    from . import load as L
    debt = L.load_owner_debt()
    y = int(debt.year.max())
    d = debt[debt.year == y].set_index(["dim", "segment"])

    def row(dim, seg):
        r = d.loc[(dim, seg)]
        return {"segment": f"{dim}: {seg}" if dim != "전체" else "전체 개인사업자",
                "avg_loan_biz_manwon": float(r.avg_loan_biz), "avg_loan_bank_manwon": float(r.avg_loan_bank),
                "median_loan_manwon": float(r.median_loan), "delinq_biz_pct": float(r.delinq_biz),
                "delinq_bank_pct": float(r.delinq_bank), "delinq_nonbank_pct": float(r.delinq_nonbank)}

    like_target = [row("전체", "전체"), row("매출액별", "3천만원 미만"), row("매출액별", "5~10천만원 미만"),
                   row("매출액별", "15~30천만원 미만"), row("사업기간별", "3년 미만"), row("종사자 유무별", "종사자 없음"),
                   row("산업분류별", "숙박 및 음식점업")]
    tot = debt[(debt.dim == "전체")].set_index("year")
    rates = L.load_ecos_lending_rates()
    latest = rates[rates.period == rates.period.max()].set_index("item").rate.to_dict()
    ext = L.load_assumption_evidence()
    base = SCENARIOS["기준"]
    return {
        "as_of_year": y,
        "current_assumptions": {name: asdict(inp) for name, inp in SCENARIOS.items()},
        "owner_debt_like_target": like_target,
        "delinq_trend_biz_loan_pct": {int(k): float(v) for k, v in tot.delinq_biz.items()},
        "bank_lending_rates_pct": {"period": rates.period.max(), **{k: float(v) for k, v in latest.items()}},
        "external": ext.to_dict(orient="records"),
        "sensitivity_base_total_eok": {
            "현재 기준(마진 3.5%·대손 1.0%)": revenue(base)["total_revenue_won"] / 1e8,
            "마진 2.5%·대손 1.5%": revenue(RevenueInput(**{**asdict(base), "spread": 0.025, "credit_loss": 0.015}))["total_revenue_won"] / 1e8,
            "마진 3.5%·대손 2.0%": revenue(RevenueInput(**{**asdict(base), "credit_loss": 0.020}))["total_revenue_won"] / 1e8,
            "마진 3.0%·대손 1.5%": revenue(RevenueInput(**{**asdict(base), "spread": 0.030, "credit_loss": 0.015}))["total_revenue_won"] / 1e8,
        },
    }


# ---------------------------------------------------------------- 분석 A: 타겟 가중 연체율
# 카드 가맹점(통계브리프) 업종 → KOSIS 「개인사업자 부채」 산업분류 대응 (카드사 업종과 표준산업분류가 달라 근사다)
_IND_TO_KSIC = {
    "요식/유흥": "숙박 및 음식점업", "미용": "협회 및 단체 수리 및 기타 개인서비스업", "음/식료품": "도매 및 소매업",
    "교육/학원": "교육 서비스업", "의료": "보건업 및 사회복지 서비스업", "유통": "도매 및 소매업",
    "스포츠/문화/레저": "예술 스포츠 및 여가관련 서비스업", "가정생활/서비스": "협회 및 단체 수리 및 기타 개인서비스업",
    "의류/잡화": "도매 및 소매업", "여행/교통": "운수 및 창고업", "가전/가구": "도매 및 소매업",
    "자동차/주유": "도매 및 소매업", "기타": None,   # None = 전체 개인사업자 평균
}
# 카드 가맹점 연매출 구간 → KOSIS 매출액 구간. 3억 원 이하는 KOSIS 5개 구간을 덮지만 그 안의 분포를 몰라
# 낮은 값(min)·가운데(mean)·높은 값(max) 세 가지로 계산한다.
_REV_TO_KOSIS = {
    "3억 원 이하": ["3천만원 미만", "3~5천만원 미만", "5~10천만원 미만", "10~15천만원 미만", "15~30천만원 미만"],
    "3억~5억 원": ["30~50천만원 미만"], "5억~10억 원": ["50~100천만원 미만"],
    "10억~30억 원": ["100천만원 이상"], "30억 원 초과": ["100천만원 이상"],
}
# 카드 가맹 등록 시기(2025.6 기준 업력) → KOSIS 사업기간 구간 비중 (낮음, 가운데, 높음 = '3년 미만' 비중 가정)
_COHORT_TO_TENURE = {
    "2020~2024년": {"3년 미만": (0.5, 0.6, 0.7), "3~10년 미만": None},      # 업력 0.5~5.5년. 최근 개업이 많아 3년 미만 50~70%
    "2010~2019년": {"3~10년 미만": (0.4, 0.4, 0.4), "10년 이상": None},     # 업력 5.5~15.5년. 고르게 분포한다고 보면 40%가 10년 미만
    "2000~2009년": {"10년 이상": (1.0, 1.0, 1.0)},
    "1994~1999년": {"10년 이상": (1.0, 1.0, 1.0)},
}


def target_weighted_delinquency(year: int | None = None) -> dict:
    """우리 타겟(외국인 사장님 가게)의 구성으로 국내 개인사업자 연체율을 가중 평균한다 — 대손비용률 가정의 근거.

    구성은 이민정책연구원 통계브리프(신한카드 외국인 개인사업자 가맹점 10,323곳)와 KOSIS 조사(1인 사업자 비중)에서,
    구간별 연체율은 KOSIS 「개인사업자 부채」(내국인 포함, 대출잔액 기준)에서 가져온다.
    사업자대출(전 금융권)과 은행 대출 두 가지 연체율로 계산한다. 연체율은 대손비용률과 같은 개념이 아니다.
    """
    from . import load as L
    debt = L.load_owner_debt()
    y = int(debt.year.max()) if year is None else year
    d = debt[debt.year == y].set_index(["dim", "segment"])
    nat = {k: float(d.loc[("전체", "전체"), f"delinq_{k}"]) for k in ("biz", "bank", "nonbank")}

    def rate(dim, seg, kind):
        return nat[kind] if seg is None else float(d.loc[(dim, seg), f"delinq_{kind}"])

    b = L.load_mrtc_brief()
    dims = {}

    # 1) 매출 구간
    rev = b[b.table == "revenue_industry"].groupby("row").value.sum()
    out = {}
    for kind in ("biz", "bank", "nonbank"):
        vals = []
        for pick in (min, lambda xs: sum(xs) / len(xs), max):
            vals.append(sum(rev[band] * pick([rate("매출액별", s, kind) for s in segs])
                            for band, segs in _REV_TO_KOSIS.items()) / rev.sum())
        out[kind] = {"low": vals[0], "mid": vals[1], "high": vals[2]}
    dims["매출 구간"] = {**out, "weights": {k: float(v / rev.sum()) for k, v in rev.items()},
                      "note": "카드 매출 기준 구간을 KOSIS 전체 연매출 구간에 맞췄다. 3억 원 이하 안의 분포를 몰라 범위로 낸다"}

    # 2) 업력
    reg = b[b.table == "registration"].groupby("row").value.sum()
    out = {}
    for kind in ("biz", "bank", "nonbank"):
        vals = []
        for i in range(3):
            tot = 0.0
            for cohort, split in _COHORT_TO_TENURE.items():
                (seg1, share), = [(k, v) for k, v in split.items() if v is not None]
                rest = [k for k, v in split.items() if v is None]
                r = share[i] * rate("사업기간별", seg1, kind) + (1 - share[i]) * (rate("사업기간별", rest[0], kind) if rest else 0)
                tot += reg[cohort] * r
            vals.append(tot / reg.sum())
        out[kind] = {"low": min(vals), "mid": vals[1], "high": max(vals)}
    dims["업력"] = {**out, "weights": {k: float(v / reg.sum()) for k, v in reg.items()},
                  "note": "가맹 등록 시기로 업력을 추정했다. 2020년 이후 개업의 '3년 미만' 비중을 50~70%로 가정"}

    # 3) 1인 사업자 비중 (외국인 자영업자, KOSIS 2025)
    se = L.load_self_employed_series().set_index("year")
    yy = int(se.index.max())
    solo = float(se.solo[yy] / se.total[yy])
    out = {kind: {k: solo * rate("종사자 유무별", "종사자 없음", kind) + (1 - solo) * rate("종사자 유무별", "종사자 있음", kind)
                  for k in ("low", "mid", "high")} for kind in ("biz", "bank", "nonbank")}
    dims["직원 유무"] = {**out, "weights": {"1인(직원 없음)": solo, "직원 있음": 1 - solo},
                     "note": f"외국인 자영업자 중 고용원 없는 비중 {solo:.1%} (KOSIS {yy})"}

    # 4) 업종
    ind = b[b.table == "industry"].set_index("row").value
    out = {kind: {k: float(sum(ind[i] * rate("산업분류별", _IND_TO_KSIC[i], kind) for i in ind.index) / ind.sum())
                  for k in ("low", "mid", "high")} for kind in ("biz", "bank", "nonbank")}
    dims["업종"] = {**out, "weights": {k: float(v / ind.sum()) for k, v in ind.items()},
                  "note": "카드사 업종을 표준산업분류에 근사 대응(요식/유흥 → 숙박 및 음식점업 등)"}

    mids = [v["biz"]["mid"] for v in dims.values()]
    lows = [v["biz"]["low"] for v in dims.values()]
    highs = [v["biz"]["high"] for v in dims.values()]
    trend = debt[(debt.dim == "전체") & (debt.segment == "전체")].set_index("year").delinq_biz
    return {
        "as_of_year": y,
        "national": nat,
        "dims": dims,
        "summary_biz": {"low": min(lows), "mid_range": [min(mids), max(mids)], "high": max(highs)},
        "national_trend_biz": {int(k): float(v) for k, v in trend.items()},
        "current_credit_loss_pct": {k: v.credit_loss * 100 for k, v in SCENARIOS.items()},
    }


# ---------------------------------------------------------------- 분석 C: 시장 퍼널 (수익 시나리오와 같은 숫자)
def market_funnel(scenario: str = "기준") -> dict:
    """대출·송금 두 상품의 시장 퍼널. 숫자는 revenue()와 같은 계산에서 나온다(옛 fig14와 달리 t04와 어긋나지 않는다).

    대출: 사업 가능 체류자격 체류자(예비) → 외국인 사장님(시나리오 연도) → 대출 고객(침투율) → 순이자 수익
    송금: 15세 이상 상주 외국인 → 해외 송금 외국인 → 송금 고객(침투율) → 수수료 수익
    단계마다 자료의 정의·시점이 달라 단계 사이 비율을 '전환율'로 읽으면 안 된다.
    """
    inp = SCENARIOS[scenario]
    r = revenue(inp)
    tc = target_counts()
    pop15 = float(_sv("population", 2025).value.iloc[0] * 1e3)
    loan = [
        {"stage": "예비 사장님", "people": tc["pool"]["count"], "def": "사업 가능 체류자격 체류자", "as_of": tc["pool"]["as_of"]},
        {"stage": "외국인 사장님", "people": r["target_owners"], "def": "자영업자(고용주+1인)" + (" 전망" if inp.target_year > 2025 else ""),
         "as_of": str(inp.target_year)},
        {"stage": "대출 고객", "people": r["borrowers"], "def": f"침투율 {inp.penetration:.0%} (가정)", "as_of": str(inp.target_year),
         "revenue_won": r["loan_revenue_won"], "book_won": r["loan_book_won"]},
    ]
    remit = [
        {"stage": "상주 외국인", "people": pop15, "def": "15세 이상 상주 외국인", "as_of": "2025.5"},
        {"stage": "송금 외국인", "people": remit_senders(), "def": "해외 송금을 하는 외국인", "as_of": "2025.5"},
        {"stage": "송금 고객", "people": r["remit_users"], "def": f"침투율 {inp.remit_penetration:.0%} (가정)", "as_of": "2025.5",
         "revenue_won": r["remit_revenue_won"]},
    ]
    return {"scenario": scenario, "input": r["input"], "loan": loan, "remit": remit,
            "total_revenue_won": r["total_revenue_won"], "mean_annual_remittance_won": mean_annual_remittance()}


# ---------------------------------------------------------------- 분석 D: 시도별 사장님 전환 지수
_SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "대전광역시": "대전",
               "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원", "충청북도": "충북",
               "충청남도": "충남", "전북특별자치도": "전북", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
               "전남광주통합특별시": "광주·전남"}


def province_conversion() -> list[dict]:
    """시도별 '예비 사장님 1만 명당 외국인 카드 가맹점 수'와 전국 대비 지수(전국 = 100).

    가맹점: 이민정책연구원 통계브리프 표 5(신한카드, 2025.6, 해지 2,288곳 포함). 예비 사장님: 법무부 사업 가능 체류자격(2026.9).
    광주·전남은 2026년 7월 통합 행정구역에 맞춰 합쳤다. 한 카드사 기준이고 두 자료의 시점·대상이 달라 '전환율'이 아니라 상대 지수다.
    """
    from . import load as L
    card = L.load_mrtc_brief("region_industry").groupby("row").value.sum()
    card = card.rename(index={"광주": "광주·전남", "전남": "광주·전남"}).groupby(level=0).sum()
    reg = pd.read_csv(PROC / "region_business_eligible.csv").groupby("시도")[["biz", "총합계"]].sum()
    reg.index = reg.index.map(_SIDO_SHORT)
    d = reg.join(card.rename("merchants"), how="inner")
    assert len(d) == 16 and d.merchants.sum() == 10_323
    nat = d.merchants.sum() / d.biz.sum() * 1e4
    d["per_10k"] = d.merchants / d.biz * 1e4
    d["index"] = d.per_10k / nat * 100
    d = d.sort_values("per_10k", ascending=False)
    return [{"sido": k, "merchants": int(r.merchants), "business_eligible": float(r.biz), "foreign_total": float(r["총합계"]),
             "per_10k": float(r.per_10k), "index": float(r["index"])} for k, r in d.iterrows()]


# ---------------------------------------------------------------- 분석 F: 정착 흐름 (체류자격 변경)
def visa_change_trend() -> dict:
    """체류자격을 바꾼 적이 있는 외국인 비율(%) — 전체와 체류자격별(현재 자격 기준), 2012~ (KOSIS DT_2FK005R).

    해마다 다른 표본이라 같은 사람을 따라간 값이 아니다. 계속 체류 희망(DT_2FK008R)은 영주(F-5)에게 묻지 않는다.
    """
    from . import load as L
    tot = L.kosis_wide("DT_2FK005R")["T200"]
    by = L.kosis_wide("DT_2FK005R", by="visa")["T200"].unstack("visa")
    wish = L.kosis_wide("DT_2FK008R", who=None, by="visa")["T200"].unstack("visa")
    y = int(tot.index.max())
    return {"total": {int(k): float(v) for k, v in tot.items()},
            "by_visa": {v: {int(k): float(x) for k, x in by[v].dropna().items()} for v in by.columns},
            "wish_stay_latest": {v: float(wish.loc[y, v]) for v in wish.columns}, "latest_year": y}


# ---------------------------------------------------------------- 분석 E: 고객 절감액 (송금)
REMIT_FREQ_MID = {"T210": 1, "T220": 2, "T230": 3, "T240": 4, "T250": 5, "T260": 6, "T270": 9, "T280": 12}


def remittance_savings(our_fee: float | None = None, year: int = 2025) -> dict:
    """송금 고객 1인당 연 비용을 '우리 수수료'와 '지금 쓰는 방법'으로 비교한다.

    1인 연 송금액은 조사의 송금액 구간 분포(mean_annual_remittance), 1년 송금 횟수는 횟수 구간의 대표값으로 구한다
    (12회 이상은 12회로 둔다 → 1회 금액은 실제보다 크게, 건당 수수료 비율은 작게 나온다).
    비교 대상: ① 시중은행 건당 수수료 1만·3만 원(환율 마진 제외 — 블로그 출처, 저신뢰)
              ② World Bank 한국발 송금 총비용 5.36~5.90%(수수료+환율 마진, 200달러 기준이라 큰 금액엔 과대).
    환율 마진 자료가 없어 결과가 비교 대상에 따라 크게 갈린다.
    """
    from . import load as L
    fee = SCENARIOS["기준"].remit_fee if our_fee is None else our_fee
    f = L.kosis_wide("DT_2FI004F").loc[year]
    senders = float(f["T200"])
    n_per_year = float(sum(f[k] * m for k, m in REMIT_FREQ_MID.items()) / senders)
    annual = mean_annual_remittance()
    per_transfer = annual / n_per_year
    ours = annual * fee
    alts = []
    for label, cost in (("시중은행 건당 1만 원(환율 마진 제외)", 10_000 * n_per_year),
                        ("시중은행 건당 3만 원(환율 마진 제외)", 30_000 * n_per_year),
                        ("World Bank 평균 총비용 5.36%(베트남행)", annual * 0.0536),
                        ("World Bank 평균 총비용 5.90%(중국행)", annual * 0.0590)):
        alts.append({"label": label, "annual_cost_won": cost, "cost_pct": cost / annual * 100, "saving_won": cost - ours})
    return {"year": year, "annual_remittance_won": annual, "transfers_per_year": n_per_year, "per_transfer_won": per_transfer,
            "our_fee_pct": fee * 100, "our_annual_cost_won": ours, "alternatives": alts,
            "saving_range_won": [min(a["saving_won"] for a in alts), max(a["saving_won"] for a in alts)]}


# ---------------------------------------------------------------- 팀 최종 초안 근거 (비자 스텝업 마이너스 통장 · 선배 귀화인 P2P)
# 초안(data/raw/team/proposal_draft_final.pdf)에 적힌 예시값. 초안 스스로 "사업성 검토를 위한 가정"이라고 밝힌 값이다.
# 이용률·대출 기간처럼 초안에 없는 값은 우리가 둔 가정이고 note에 적었다. 바꾸면 노트북·API가 같이 바뀐다.
DRAFT_ASSUME = {
    "p1_limit_won": 15_000_000,            # 초안: 초기 한도 최대 1,500만 원 (1,000만~1,500만)
    "p1_utilization": 0.70,                # 우리 가정: 한도 평균 이용률 (초안에 없음)
    "p1_rate_entry": 0.15,                 # 초안: 초기 진입 연 14~16%
    "p1_cut_stage2": 0.005,                # 초안: 3개월 성실 상환·무체납 → 0.5%p
    "p1_cut_kiip_step": 0.003,             # 초안: KIIP 단계별 0.3%p (누적 한도)
    "p1_kiip_steps": 3,                    # 우리 가정: 누적 한도를 3단계(0.9%p)로 둔다
    "p1_cut_stage4": 0.015,                # 초안: KIIP 5단계·영주·귀화 신청 → 추가 1.5%p
    "p1_rate_mid": 0.10,                   # 초안: 중간 단계 연 9~11%
    "p1_rate_final": 0.0525,               # 초안: 국적 취득 후 제휴 은행 연 4.5~6%
    "p1_dsr_months": 60,                   # 우리 가정: 한도대출을 5년 분할로 보고 DSR 계산
    "p2_loan_won": 20_000_000,             # 초안: 총 2,000만 원 모집
    "p2_mentor_share": 0.10,               # 초안: 멘토 10% 후순위
    "p2_rate": 0.125,                      # 초안: 차주 금리 연 11~14%
    "p2_months": 12,                       # 우리 가정: 12개월 원리금균등 상환 (초안에 없음)
    "p2_platform_fee": 0.03,               # 초안: 대출 실행액의 3% (2.0~3.5%)
    "p2_mgmt_fee": 0.12,                   # 초안: 투자자 이자 수익의 12% (10~15%)
}


def draft_stage_rates() -> list[dict]:
    """비자 스텝업: 단계별 금리 (초안 예시값으로 계산)."""
    a = DRAFT_ASSUME
    r1 = a["p1_rate_entry"]
    r2 = r1 - a["p1_cut_stage2"]
    r3 = r2 - a["p1_cut_kiip_step"] * a["p1_kiip_steps"]
    r4 = r3 - a["p1_cut_stage4"]
    return [{"stage": "1 진입", "rate": r1, "condition": "운영 6개월+, 최근 3개월 매출, 기본 심사"},
            {"stage": "2 성실 이행", "rate": r2, "condition": "3개월 성실 상환·무체납"},
            {"stage": "3 정착 진전", "rate": r3, "condition": f"KIIP {a['p1_kiip_steps']}개 단계 수료"},
            {"stage": "4 중간 목표", "rate": r4, "condition": "KIIP 5단계 합격 또는 영주·귀화 신청"},
            {"stage": "5 최종 전환", "rate": a["p1_rate_final"], "condition": "국적 취득 후 제휴 은행 별도 심사"}]


def draft_repayment() -> dict:
    """비자 스텝업: 단계별 금리에서 한도 1,500만 원의 DSR 40% 통과 비율과 고객 연 이자."""
    a = DRAFT_ASSUME
    bal = a["p1_limit_won"] * a["p1_utilization"]
    out = []
    for st in draft_stage_rates():
        d = dsr_pass(a["p1_limit_won"], st["rate"], a["p1_dsr_months"], 0.40)
        out.append({**st, "monthly_payment_won": d["monthly_payment_won"], "required_monthly_income_won": d["required_monthly_income_won"],
                    "pass_by_group": d["by_group"], "annual_interest_won": bal * st["rate"]})
    return {"limit_won": a["p1_limit_won"], "avg_balance_won": bal, "stages": out,
            "saving_to_mid_won": out[0]["annual_interest_won"] - out[3]["annual_interest_won"],
            "saving_to_final_won": out[0]["annual_interest_won"] - out[4]["annual_interest_won"]}


def rate_benchmarks() -> dict:
    """초안 금리(14~16% / 9~11% / 4.5~6% / P2P 11~14%)를 시장 금리와 나란히 놓는다."""
    from . import load as L
    ev = L.load_assumption_evidence()
    ext = ev[ev.assumption.isin(["금리 비교", "대출금리"])]
    rates = L.load_ecos_lending_rates()
    latest = rates[rates.period == rates.period.max()].set_index("item").rate
    bench = [{"label": r.metric, "rate": float(r.value), "as_of": r.as_of, "grade": r.grade, "url": r.url}
             for r in ext.itertuples() if r.unit == "%" and "연체율" not in r.metric]
    bench.append({"label": "예금은행 일반신용대출 금리(신규)", "rate": float(latest["일반신용대출"]), "as_of": rates.period.max(),
                  "grade": "공식 통계", "url": "https://ecos.bok.or.kr (121Y006)"})
    draft = [{"label": "초안 비자 스텝업 진입", "low": 14.0, "high": 16.0}, {"label": "초안 비자 스텝업 중간", "low": 9.0, "high": 11.0},
             {"label": "초안 제휴 은행 대환", "low": 4.5, "high": 6.0}, {"label": "초안 커뮤니티 P2P", "low": 11.0, "high": 14.0}]
    return {"benchmarks": sorted(bench, key=lambda x: x["rate"]), "draft": draft}


def settlement_pipeline() -> dict:
    """정착 사다리: KIIP 참여자, 영주(F-5) 체류자, 귀화 허가, 영주권자의 국적 취득 계획, KIIP 평균 학습기간."""
    from . import load as L
    kiip = L.load_kosis("DT_1B040A28").pivot_table(index="year", columns="ITM_NM", values="value")
    nat = L.load_kosis("DT_1B040A18").pivot_table(index="year", columns="C1_NM", values="value")
    ts = L.load_visa_timeseries()
    f5 = ts[ts.visa == "F5"].set_index("date")["count"]
    plan = L.load_kosis("DT_2FS030R")
    plan = plan[(plan.C1_NM.str.strip() == "계") & (plan.ITM_NM.str.strip() == "있음")].set_index("year").value
    learn = L.load_kosis("DT_2FQ005R")
    learn = learn[(learn.C1_NM == "외국인") & (learn.C2_NM.str.strip() == "계") & (learn.ITM_NM.str.strip() == "평균 학습기간")].set_index("year").value
    nw = L.kosis_wide("DT_2FB007F", who="귀화허가자")
    npop = L.kosis_wide("DT_2FF001F", who="귀화허가자")["T100"]
    fw = L.kosis_wide("DT_2FB007F", who="외국인")
    y = int(nw.index.max())
    return {
        "kiip_participants": {int(k): float(v) for k, v in kiip["교육참여자"].dropna().items()},
        "kiip_institutions": {int(k): float(v) for k, v in kiip["운영기관수"].dropna().items()},
        "naturalized_per_year": {int(k): float(v) for k, v in nat["귀화"].dropna().items()},
        "f5_stock": {d.strftime("%Y-%m"): float(v) for d, v in f5.items()},
        "f5_plan_naturalize_pct": {int(k): float(v) for k, v in plan.items()},
        "kiip_avg_learning_months": {int(k): float(v) for k, v in learn.items()},
        "recent_naturalized": {"year": y, "population_k": float(npop[y]), "employed_k": float(nw.loc[y, "T100"]),
                               "self_employed_k": float(nw.loc[y, "T310"] + nw.loc[y, "T320"]),
                               "self_emp_share_of_employed_pct": float((nw.loc[y, "T310"] + nw.loc[y, "T320"]) / nw.loc[y, "T100"] * 100),
                               "foreign_self_emp_share_of_employed_pct": float((fw.loc[y, "T310"] + fw.loc[y, "T320"]) / fw.loc[y, "T100"] * 100)},
    }


_SIDO_CODE = {"11": "서울", "21": "부산", "22": "대구", "23": "인천", "24": "전남", "25": "대전", "26": "울산", "29": "세종",
              "31": "경기", "32": "강원", "33": "충북", "34": "충남", "35": "전북", "36": "전남", "37": "경북", "38": "경남", "39": "제주"}


def mentor_pool(top: int = 15) -> dict:
    """선배 귀화인 멘토 후보 지도: 시군구별 '국적 취득 5년 이상' 주민(행안부 2024) vs 예비 사장님(법무부 2026.9).

    광주 구는 2026 통합 행정구역 표기(전남)에 맞춘다. 인천 2026 개편 구는 매칭이 안 될 수 있다.
    멘토 후보 수 추정은 최근 5년 귀화자의 고용률·자영업 비중을 5년 이상 귀화자에게 그대로 적용한 거친 값이다.
    """
    from . import load as L
    d = L.load_kosis("DT_110025_A040_A")
    y = int(d.year.max())
    d = d[(d.year == y)]
    code = d.C1.str.replace("11101HJG", "", regex=False)
    d = d.assign(code=code)
    sgg = d[d.code.str.len() == 5].copy()
    sgg["label"] = sgg.code.str[:2].map(_SIDO_CODE) + " " + sgg.C1_NM.str.strip()
    long_ = sgg[sgg.C2_NM.str.strip().isin(["5년이상~10년미만", "10년이상"])].groupby("label").value.sum()
    allnat = sgg[sgg.C2_NM.str.strip() == "계"].groupby("label").value.sum()
    reg = pd.read_csv(PROC / "region_business_eligible.csv").set_index("label").biz
    df = pd.DataFrame({"naturalized_5y_plus": long_, "naturalized_all": allnat}).join(reg.rename("prospective"), how="left")
    df["per_100_prospective"] = df.naturalized_5y_plus / df.prospective * 100
    df = df.sort_values("naturalized_5y_plus", ascending=False)
    t0 = d[d.code == "00"]
    tot = t0.set_index(t0.C2_NM.str.strip()).value
    pipe = settlement_pipeline()["recent_naturalized"]
    emp_rate = pipe["employed_k"] / pipe["population_k"]
    se_share = pipe["self_emp_share_of_employed_pct"] / 100
    n5 = float(tot["5년이상~10년미만"] + tot["10년이상"])
    hubs = {"안산(경기 안산시)": "경기 안산시", "대림(서울 영등포구)": "서울 영등포구", "대림·가리봉(서울 구로구)": "서울 구로구",
            "수원(경기 수원시)": "경기 수원시", "이태원(서울 용산구)": "서울 용산구"}
    return {
        "year": y, "naturalized_total": float(tot["계"]), "naturalized_5y_plus": n5,
        "mentor_candidates_est": n5 * emp_rate * se_share,
        "est_note": f"5년 이상 귀화자 × 최근 귀화자 고용률 {emp_rate:.0%} × 취업자 중 자영업 {se_share:.1%} (거친 추정)",
        "top": [{"label": k, **{c: (None if pd.isna(v) else float(v)) for c, v in r.items()}} for k, r in df.head(top).iterrows()],
        "draft_hubs": [{"hub": h, "label": lab, **({c: (None if pd.isna(v) else float(v)) for c, v in df.loc[lab].items()} if lab in df.index else {})}
                       for h, lab in hubs.items()],
    }


def draft_unit_economics(loss_rates: tuple = (0.011, 0.021, 0.029, 0.05, 0.10)) -> dict:
    """두 상품의 고객·대출 1건당 단위경제성 (운영비·모집비 차감 전).

    비자 스텝업: 평균 잔액 × (금리 − 조달금리 − 손실률). 조달금리는 저축은행 정기예금(1년) 최신값(ECOS)을 쓴다.
    P2P: 플랫폼 수익 = 실행액 × 중개 수수료 + 투자자 이자 × 관리 수수료. 멘토 후순위(10%)가 먼저 손실을 진다.
    손실률은 연체율을 대리값으로 쓴다(연체율 ≠ 손실률). 단위경제성만 보며 고객 수는 정하지 않는다.
    """
    from . import load as L
    a = DRAFT_ASSUME
    fund = None
    for p in sorted((L.RAW / "ecos").glob("ecos_121Y004_BEBBBE01_*.json")):
        rows = __import__("json").load(open(p, encoding="utf-8"))["StatisticSearch"]["row"]
        fund = float(rows[-1]["DATA_VALUE"]) / 100
        fund_as_of = rows[-1]["TIME"]
    twd = target_weighted_delinquency()
    nb = twd["dims"]
    nb_mid = sorted(v["nonbank"]["mid"] for v in nb.values())
    bal = a["p1_limit_won"] * a["p1_utilization"]
    p1 = []
    for st in draft_stage_rates()[:4]:
        for lr in (nb_mid[0] / 100, nb_mid[-1] / 100):
            p1.append({"stage": st["stage"], "rate": st["rate"], "loss_rate": lr,
                       "margin_won": bal * (st["rate"] - fund - lr), "margin_pct": (st["rate"] - fund - lr) * 100})
    # P2P: 원리금균등 12개월
    P, r, n = a["p2_loan_won"], a["p2_rate"] / 12, a["p2_months"]
    pay = P * r / (1 - (1 + r) ** -n)
    interest = pay * n - P
    plat = P * a["p2_platform_fee"] + interest * a["p2_mgmt_fee"]
    mentor_amt = P * a["p2_mentor_share"]
    mentor_int = interest * a["p2_mentor_share"] * (1 - a["p2_mgmt_fee"])
    p2 = []
    for lr in loss_rates:
        loss = P * lr
        m_loss = min(loss, mentor_amt)
        p2.append({"loss_rate": lr, "loss_won": loss, "mentor_net_won": mentor_int - m_loss,
                   "mentor_return_pct": (mentor_int - m_loss) / mentor_amt * 100,
                   "investor_loss_won": max(0.0, loss - mentor_amt),
                   # 멘토가 손해를 보지 않으려면 전체 이자 중 멘토 몫이 얼마여야 하나 (지금은 원금 비율과 같은 10%)
                   "mentor_interest_share_needed": min(loss, mentor_amt) / (interest * (1 - a["p2_mgmt_fee"]))})
    return {"funding_rate": fund, "funding_as_of": fund_as_of, "p1_avg_balance_won": bal, "p1": p1,
            "p1_loss_proxy_pct": {"low": nb_mid[0], "high": nb_mid[-1], "note": "타겟 가중 비은행 개인사업자 연체율(관점별 가운데 값의 최소~최대)"},
            "p2": {"loan_won": P, "total_interest_won": interest, "platform_revenue_won": plat,
                   "platform_fee_won": P * a["p2_platform_fee"], "mgmt_fee_won": interest * a["p2_mgmt_fee"],
                   "mentor_amount_won": mentor_amt, "mentor_interest_net_won": mentor_int,
                   "mentor_breakeven_loss_rate": mentor_int / P, "by_loss": p2},
            "assume": DRAFT_ASSUME}


# ---------------------------------------------------------------- Q3 상환능력
def monthly_payment(principal: float, annual_rate: float = 0.06, months: int = 60) -> float:
    r = annual_rate / 12
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def required_income(principal: float, annual_rate: float = 0.06, months: int = 60, dsr: float = 0.40) -> float:
    return monthly_payment(principal, annual_rate, months) / dsr


def _share_above(row: pd.Series, need_man: float) -> float:
    earners = sum(row[c] for c, *_ in INCOME_BINS)
    above = 0.0
    for c, lo, hi in INCOME_BINS:
        if hi is None:
            above += row[c] if need_man <= lo else row[c] * 0.5  # 300만+ 개방구간: 기준 초과 시 절반만(보수)
        elif need_man <= lo:
            above += row[c]
        elif need_man < hi:
            above += row[c] * (hi - need_man) / (hi - lo)
    return above / earners * 100


def dsr_pass(principal: float, annual_rate: float = 0.06, months: int = 60, dsr: float = 0.40) -> dict:
    need_man = required_income(principal, annual_rate, months, dsr) / 1e4
    v = _visa().copy()
    v["pass_pct"] = v.apply(lambda r: _share_above(r, need_man), axis=1)
    v["earners_k"] = v.pop_k * (100 - v.inc_none) / 100
    groups = {g: float(np.average(d.pass_pct, weights=d.earners_k)) for g, d in v.groupby("grp")}
    return {"monthly_payment_won": monthly_payment(principal, annual_rate, months),
            "required_monthly_income_won": need_man * 1e4,
            "by_visa": v[["visa", "visa_label", "grp", "pass_pct", "utility_miss"]].to_dict(orient="records"),
            "by_group": groups}


# ---------------------------------------------------------------- Q4 정착
def settlement_corr() -> list[dict]:
    return pd.read_csv(PROC / "settlement_corr_region.csv").to_dict(orient="records")


def visa_group_tests() -> list[dict]:
    return pd.read_csv(PROC / "test_A1_visa_groups.csv").to_dict(orient="records")


# ---------------------------------------------------------------- 시장 지표 (갱신되는 원자료를 바로 읽는다)
def fx_recent(months: int = 12) -> dict:
    """한국은행 월평균 환율: 최근 n개월 값과 변동폭. src/refresh.py로 새로 받으면 바로 반영된다."""
    from . import load as L
    fx = L.load_ecos_monthly_fx().tail(months)
    out = {"latest_period": fx.date.iloc[-1].strftime("%Y-%m"), "months": int(len(fx)), "series": {}}
    for col in ("KRW/USD", "KRW/CNY"):
        s = fx[col]
        out["series"][col] = {
            "latest": float(s.iloc[-1]), "mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()),
            "range_pct": float((s.max() - s.min()) / s.mean() * 100),   # 기간 내 고저 차이 / 평균
            "change_pct": float((s.iloc[-1] / s.iloc[0] - 1) * 100),    # 기간 첫 달 대비
        }
    out["monthly"] = [{"period": d.strftime("%Y-%m"), "KRW/USD": float(u), "KRW/CNY": float(c)}
                      for d, u, c in zip(fx.date, fx["KRW/USD"], fx["KRW/CNY"])]
    return out


def remittances_paid(years: int = 10) -> list[dict]:
    """World Bank: 한국에서 해외로 나간 개인송금(달러). 외국인만의 송금이 아니라 내국인 포함 전체다."""
    from . import load as L
    rem = L.load_wb_remittances_paid().tail(years)
    return [{"year": int(y), "usd": float(v), "usd_billion": float(v) / 1e9} for y, v in zip(rem.year, rem.usd)]
