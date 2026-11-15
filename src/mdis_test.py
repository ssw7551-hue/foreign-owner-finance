"""개인 단위 본검정: '자녀가 있는 외국인은 거주안정성·상환능력이 높다'.

필요 자료: MDIS(https://mdis.kostat.go.kr) 「이민자체류실태및고용조사」 2025 공개용 마이크로데이터.
  1) MDIS 회원가입(무료) → 다운로드 서비스 → 이민자체류실태및고용조사 → 2025 → 외국인 파일 + 코드북(레이아웃) 다운로드
  2) 데이터 파일(csv 또는 txt)을 data/raw/mdis/ 에 저장
  3) 아래 COLUMN_MAP / CODE_MAP 을 코드북의 실제 변수명·코드값으로 채운다 (지금 값은 자리표시자)

파일이 없거나 COLUMN_MAP 이 채워지지 않으면 run() 은 안내만 출력하고 아무 결과도 만들지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MDIS_DIR = ROOT / "data" / "raw" / "mdis"

# 코드북을 보고 채울 것: 분석용 이름 -> MDIS 변수명
COLUMN_MAP = {
    "weight": None,          # 가중치
    "has_child": None,       # 자녀 유무
    "child_where": None,     # 자녀 거주지(모두 국내/일부 국내/모두 국외)
    "stay_years": None,      # 한국 체류기간(구간 코드)
    "wish_stay": None,       # 계속 체류 희망 여부
    "tenure": None,          # 점유형태(자기 집/전월세/무상)
    "income": None,          # 지난 1년 월평균 총소득(구간 코드)
    "hardship_utility": None,  # 경제적 어려움 - 공과금 기한 내 미납(복수응답 해당 여부)
    "visa": None, "sex": None, "age": None, "edu": None, "region": None,
}

# 코드북을 보고 채울 것: 각 결과변수를 1/0으로 만드는 코드값
CODE_MAP = {
    "has_child_yes": None,          # 예: 1
    "child_all_korea": None,        # 예: 1
    "stay_10y_codes": None,         # 예: [6]
    "wish_stay_yes": None,          # 예: 1
    "tenure_own": None,             # 예: 1
    "income_300p_codes": None,      # 예: [5]
    "income_none_codes": None,      # 예: [1]
    "utility_yes": None,            # 예: 1
}

OUTCOMES = {  # 이름: (설명, 좋은 방향 +1 / 나쁜 방향 -1)
    "stay_10y": ("체류 10년 이상", +1),
    "wish_stay": ("계속 체류 희망", +1),
    "own_home": ("자기 집 거주", +1),
    "inc_300p": ("월소득 300만 원 이상(소득자)", +1),
    "utility_miss": ("공과금 미납 경험", -1),
}


def _find_file():
    if not MDIS_DIR.exists():
        return None
    files = sorted([p for p in MDIS_DIR.iterdir() if p.suffix.lower() in (".csv", ".txt", ".sav", ".dta")])
    return files[0] if files else None


def ready() -> tuple[bool, str]:
    f = _find_file()
    if f is None:
        return False, f"MDIS 파일 없음: {MDIS_DIR} 에 마이크로데이터를 저장하세요."
    missing = [k for k, v in COLUMN_MAP.items() if v is None]
    if missing:
        return False, f"src/mdis_test.py 의 COLUMN_MAP 미설정: {missing}"
    if any(v is None for v in CODE_MAP.values()):
        return False, "src/mdis_test.py 의 CODE_MAP 미설정"
    return True, str(f)


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".sav":
        return pd.read_spss(path)
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path)
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없음: {path}")


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    c, k = COLUMN_MAP, CODE_MAP
    d = pd.DataFrame({
        "w": pd.to_numeric(raw[c["weight"]], errors="coerce"),
        "child": (raw[c["has_child"]] == k["has_child_yes"]).astype(int),
        "child_kr": (raw[c["child_where"]] == k["child_all_korea"]).astype(int),
        "stay_10y": raw[c["stay_years"]].isin(k["stay_10y_codes"]).astype(int),
        "wish_stay": (raw[c["wish_stay"]] == k["wish_stay_yes"]).astype(float),
        "own_home": (raw[c["tenure"]] == k["tenure_own"]).astype(int),
        "utility_miss": (raw[c["hardship_utility"]] == k["utility_yes"]).astype(int),
        "visa": raw[c["visa"]].astype(str), "sex": raw[c["sex"]].astype(str),
        "age": pd.to_numeric(raw[c["age"]], errors="coerce"), "edu": raw[c["edu"]].astype(str),
        "region": raw[c["region"]].astype(str),
    })
    earner = ~raw[c["income"]].isin(k["income_none_codes"])
    d["inc_300p"] = np.where(earner, raw[c["income"]].isin(k["income_300p_codes"]).astype(float), np.nan)
    d.loc[raw[c["wish_stay"]].isna(), "wish_stay"] = np.nan  # 영주(F-5) 비해당
    d["child_kr"] = d["child_kr"] * d["child"]  # 자녀 없는 경우 0
    return d.dropna(subset=["w", "age"])


def bivariate(d: pd.DataFrame, exposure="child") -> pd.DataFrame:
    """가중 비율(노출 1 vs 0), 비율차, 카이제곱(가중치를 표본 크기로 재조정)."""
    from scipy import stats
    rows = []
    for y, (label, _) in OUTCOMES.items():
        s = d.dropna(subset=[y])
        w = s.w * len(s) / s.w.sum()  # 가중합 = 표본수 (p값 과대 방지)
        p1 = np.average(s[y][s[exposure] == 1], weights=w[s[exposure] == 1])
        p0 = np.average(s[y][s[exposure] == 0], weights=w[s[exposure] == 0])
        tab = pd.crosstab(s[exposure], s[y], values=w, aggfunc="sum").fillna(0).values
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"결과": label, "노출=1 (%)": p1 * 100, "노출=0 (%)": p0 * 100,
                     "차이(%p)": (p1 - p0) * 100, "χ²": chi2, "p": p, "n": len(s)})
    return pd.DataFrame(rows)


def adjusted_or(d: pd.DataFrame, exposure="child") -> pd.DataFrame:
    """가중 로지스틱 회귀(GLM Binomial). 통제: 성·연령·연령²·학력·체류자격·지역(+소득 이외 결과는 소득300+ 제외)."""
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    rows = []
    for y, (label, _) in OUTCOMES.items():
        s = d.dropna(subset=[y]).copy()
        s["wn"] = s.w * len(s) / s.w.sum()
        f = f"{y} ~ {exposure} + C(sex) + age + I(age**2) + C(edu) + C(visa) + C(region)"
        m = smf.glm(f, data=s, family=sm.families.Binomial(), freq_weights=s["wn"]).fit(cov_type="HC1")
        b, se = m.params[exposure], m.bse[exposure]
        rows.append({"결과": label, "OR": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se),
                     "p": m.pvalues[exposure], "n": int(m.nobs)})
    return pd.DataFrame(rows)


def run():
    ok, msg = ready()
    if not ok:
        print("[Part B 건너뜀]", msg)
        print("→ MDIS 공개용 마이크로데이터를 받으면 이 셀이 개인 단위 카이제곱·가중 로지스틱 회귀를 자동 실행합니다.")
        return None
    d = prepare(read_raw(Path(msg)))
    return {"data": d,
            "biv_child": bivariate(d, "child"), "biv_child_kr": bivariate(d, "child_kr"),
            "or_child": adjusted_or(d, "child"), "or_child_kr": adjusted_or(d, "child_kr")}
