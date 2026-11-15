"""Loaders that turn the raw public files in data/raw/ into tidy DataFrames.

Every function reads only local files (downloaded once, see data/SOURCES.md),
so the notebook runs offline and reproducibly.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "curated"

# 체류자격 groups used across the analysis.
# "사업 가능" = 체류자격상 자영업/개인사업이 허용되거나 사업 목적 자격.
VISA_GROUPS = {
    "영주·거주(F-2·F-5)": ["F2", "F5"],
    "재외동포(F-4)": ["F4"],
    "결혼이민(F-6)": ["F6"],
    "기업투자·무역경영(D-8·D-9)": ["D8", "D9"],
    "유학·구직(D-2·D-4·D-10)": ["D2", "D4", "D10"],
    "비전문·방문취업(E-9·H-2)": ["E9", "H2"],
    "전문인력(E-1~E-7)": ["E1", "E2", "E3", "E4", "E5", "E6", "E7"],
}
BUSINESS_ELIGIBLE = ["F2", "F4", "F5", "F6", "D8", "D9"]

_TOTAL_WORDS = {"총합계", "총계", "한계", "소계", "계"}


def _visa_code(col: str) -> str:
    """'D9(무역경영)' -> 'D9'; leaves non-visa columns unchanged."""
    m = re.match(r"^([A-Z]\d{1,2})\(", str(col))
    return m.group(1) if m else str(col)


# ---------------------------------------------------------------- MOJ: national
def load_stay_by_nationality(path: Path | str) -> pd.DataFrame:
    """MOJ '국적(지역) 및 체류자격별 장단기체류외국인' table -> one row per nationality.

    Returns columns: 대륙, 국적, 총합계, and one column per visa code (B1 ... H2, 기타).
    """
    df = pd.read_excel(path, header=2)
    df.columns = [_visa_code(c) for c in df.columns]
    df = df.rename(columns={df.columns[0]: "대륙", df.columns[1]: "국적", df.columns[2]: "성별"})
    df = df[df["대륙"].notna() & ~df["대륙"].isin(["총합계"])]
    df = df[~df["국적"].isin(list(_TOTAL_WORDS))]
    # first row per (대륙, 국적) is the both-sex total
    df = df[df["성별"].isin(list(_TOTAL_WORDS))]
    keep = [c for c in df.columns if c not in ("성별",) and not str(c).startswith("Unnamed")]
    out = df[keep].copy()
    num = [c for c in out.columns if c not in ("대륙", "국적")]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    return out.reset_index(drop=True)


def load_stay_totals(path: Path | str) -> pd.Series:
    """Grand-total row (all nationalities) of the same MOJ table, indexed by visa code."""
    df = pd.read_excel(path, header=2)
    df.columns = [_visa_code(c) for c in df.columns]
    row = df.iloc[0]
    s = row.iloc[3:]
    s = s[[not str(i).startswith("Unnamed") for i in s.index]]
    return pd.to_numeric(s, errors="coerce").fillna(0)


def load_visa_timeseries() -> pd.DataFrame:
    """Year-end (Dec) visa totals 2020-2025 plus the latest month, long format.

    Columns: date, visa, count
    """
    rows = []
    for p in sorted(glob.glob(str(RAW / "moj" / "stay_*.xlsx"))):
        m = re.search(r"stay_(\d{4})_(\d{2})", p)
        y, mo = int(m.group(1)), int(m.group(2))
        if mo not in (12,) and not (y == 2026):
            continue  # keep Dec snapshots + the newest month
        s = load_stay_totals(p)
        for k, v in s.items():
            rows.append({"date": pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0), "visa": k, "count": v})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- MOJ: regional
SIDO_TO_GEO = {
    "서울특별시": ["11"], "부산광역시": ["21"], "대구광역시": ["22", "37"], "인천광역시": ["23"],
    "전남광주통합특별시": ["24", "36"], "광주광역시": ["24"], "전라남도": ["36"],
    "대전광역시": ["25"], "울산광역시": ["26"], "세종특별자치시": ["29"], "경기도": ["31"],
    "강원특별자치도": ["32"], "충청북도": ["33"], "충청남도": ["34"], "전북특별자치도": ["35"],
    "경상북도": ["37"], "경상남도": ["38"], "제주특별자치도": ["39"],
}


def _city_key(name: str) -> str:
    """Collapse 일반구: '수원시 권선구' / '수원시권선구' -> '수원시'."""
    name = str(name).replace(" ", "")
    m = re.match(r"^(.+?시)(.+구)$", name)
    if m and not name.endswith("특별시"):
        return m.group(1)
    return name


def load_region_table(path: Path | str) -> pd.DataFrame:
    """MOJ regional table (등록+거소, 지역별×체류자격 or ×국적) -> one row per 시군구 (both sexes)."""
    df = pd.read_excel(path, header=0)
    df.columns = [_visa_code(c) for c in df.columns]
    df = df.rename(columns={df.columns[0]: "시도", df.columns[1]: "시군구", df.columns[2]: "성별"})
    df = df[(df["성별"] == "총계") & (df["시도"] != "총합계")].copy()
    # 시도 without sub-districts (세종) only have a '총계' row: keep it as its own city
    has_sub = df[df["시군구"] != "총계"]["시도"].unique()
    df = df[(df["시군구"] != "총계") | ~df["시도"].isin(has_sub)].copy()
    df.loc[df["시군구"] == "총계", "시군구"] = df["시도"]
    num = [c for c in df.columns if c not in ("시도", "시군구", "성별")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.drop(columns="성별")
    df["city"] = df["시군구"].map(_city_key).replace(INCHEON_2026)
    return df.reset_index(drop=True)


# 인천 행정체제 개편(2026.7): 신설 구 -> 2013 경계 기준 권역으로 합산 (지도 매칭용)
INCHEON_2026 = {"제물포구": "중구·동구", "영종구": "중구·동구", "미추홀구": "남구",
                "서해구": "서구", "검단구": "서구"}


def aggregate_to_city(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.groupby(["시도", "city"], as_index=False)[list(num)].sum()


# ---------------------------------------------------------------- geo
def load_geo_features() -> list[dict]:
    """Municipal polygons with a (geo province code, city key) attached."""
    g = json.load(open(RAW / "geo" / "skorea_municipalities_2013_simple.json", encoding="utf-8"))
    feats = []
    for f in g["features"]:
        p = f["properties"]
        name = p["name"]
        if name in ("청원군",):  # merged into 청주시 (2014)
            name = "청주시"
        if name == "세종시":
            name = "세종특별자치시"
        if p["code"][:2] == "23" and name in ("중구", "동구"):
            name = "중구·동구"
        feats.append({"prov": p["code"][:2], "city": _city_key(name), "geometry": f["geometry"]})
    return feats


def match_geo(city_df: pd.DataFrame, feats: list[dict]) -> tuple[dict, list]:
    """Return {feature index -> city_df row index} and the list of unmatched city rows."""
    lookup = {}
    for i, r in city_df.iterrows():
        for prov in SIDO_TO_GEO.get(r["시도"], []):
            lookup[(prov, r["city"])] = i
    mapping, used = {}, set()
    for k, f in enumerate(feats):
        i = lookup.get((f["prov"], f["city"]))
        if i is not None:
            mapping[k] = i
            used.add(i)
    unmatched = [city_df.loc[i, ["시도", "city"]].tolist() for i in city_df.index if i not in used]
    return mapping, unmatched


# ---------------------------------------------------------------- KOSTAT survey (curated)
def load_survey(source: str = "kosis") -> pd.DataFrame:
    """이민자체류실태및고용조사 요약(긴 형식). 기본은 KOSIS 통계표(모든 연도, 소수 첫째 자리)다.

    source="curated"면 보도자료에서 손으로 옮긴 CSV(2022~2025 일부)를 읽는다. 두 자료는 반올림 범위 안에서 같다
    (tests/test_api.py가 확인한다). KOSIS 파일이 없으면 손으로 옮긴 CSV로 돌아간다.
    """
    if source == "kosis" and newest(RAW / "kosis", "DT_2FB007F", ".json").exists():
        return load_survey_kosis()
    return pd.read_csv(CURATED / "survey_immigrants.csv")


# ---------------------------------------------------------------- KOSTAT survey (KOSIS 통계표, src/refresh.py가 받는다)
def load_kosis(tbl: str) -> pd.DataFrame:
    """KOSIS 통계표 원자료(JSON) → DataFrame. year(int), value(float; 비공개·'-'는 NaN) 열을 덧붙인다."""
    d = pd.DataFrame(json.load(open(newest(RAW / "kosis", tbl, ".json"), encoding="utf-8")))
    d["year"] = d["PRD_DE"].astype(int)
    d["value"] = pd.to_numeric(d["DT"], errors="coerce")
    return d


def load_self_employed_series() -> pd.DataFrame:
    """외국인 자영업자 수 2012~ (KOSIS DT_2FB007F 종사상 지위별 취업자, 외국인·남녀 계). 단위: 명.

    Columns: year, employer(고용원 있는 자영업자), solo(고용원 없는 자영업자), total, employed(취업자 합계)
    귀화허가자와 무급가족종사자는 넣지 않는다. 표본조사라 연도별 값에 오차가 있다.
    """
    d = load_kosis("DT_2FB007F")
    d = d[(d.C1 == "10") & (d.C2 == "100")]
    p = d.pivot_table(index="year", columns="ITM_ID", values="value") * 1e3
    out = pd.DataFrame({"employer": p["T310"], "solo": p["T320"], "employed": p["T100"]})
    out["total"] = out.employer + out.solo
    return out.reset_index()[["year", "employer", "solo", "total", "employed"]]


_VISA_LABEL = {"E9": "비전문취업(E-9)", "H2": "방문취업(H-2)", "E1_7": "전문인력(E-1~E-7)", "D2_D4": "유학생(D-2·D-4)",
               "F4": "재외동포(F-4)", "F5": "영주(F-5)", "F6": "결혼이민(F-6)", "ETC": "기타"}
_VISA_KEY = {"비전문취업": "E9", "방문취업": "H2", "전문인력": "E1_7", "유학생": "D2_D4",
             "재외동포": "F4", "영주": "F5", "결혼이민": "F6", "기타": "ETC"}


def _visa_key(name: str) -> str | None:
    """KOSIS 체류자격 이름('결혼이민(F-6 F-2-1)' 등 표마다 표기가 다르다) → 짧은 코드."""
    return _VISA_KEY.get(str(name).split("(")[0].strip())


def _kosis_wide(tbl: str, who: str | None = "외국인", by: str = "계") -> pd.DataFrame:
    """표 하나를 year × ITM_ID로 펼친다. who=대상별(외국인 등), by=나머지 분류의 값('계' 또는 'visa'=체류자격별).

    나머지 분류는 표마다 이름이 다르다(특성별·체류자격별·국적별 …). 한 칸에 값이 둘 이상 겹치면 평균을 내지 않고 오류를 낸다.
    """
    d = load_kosis(tbl)
    obj = {d[c.replace("_NM", "_OBJ_NM")].iloc[0]: c for c in ("C1_NM", "C2_NM") if c in d}
    if who is not None and "대상별" in obj:
        d = d[d[obj["대상별"]] == who]
    rest = [c for name, c in obj.items() if name != "대상별"]
    col = rest[0] if rest else None
    code = col[:-3] if col else None     # 'C2_NM' -> 'C2' (분류 코드 열)
    if by == "visa":
        d = d.assign(visa=d[col].map(_visa_key)).dropna(subset=["visa"])
        # 한 표에 성별(1xx)·체류자격(2xx)·국적(3xxx)·지역(4xx) 묶음이 함께 있으면 '기타'가 여러 번 나온다.
        # 체류자격 묶음(210~280)만 남긴다. 281 이후는 '기타'의 세부 항목이다.
        if (d[code].str.fullmatch(r"2[1-8]0")).any():
            d = d[d[code].str.fullmatch(r"2[1-8]0")]
        index = ["year", "visa"]
    else:
        if col is not None:
            d = d[d[col].str.strip() == by]
        index = ["year"]
        if by == "계" and col is not None:
            # 묶음마다 '계'가 따로 있다. 값이 서로 같은지 확인하고 하나만 남긴다.
            spread = d.groupby(index + ["ITM_ID"])["value"].agg(lambda s: s.max() - s.min())
            if (spread > 0.11).any():
                raise ValueError(f"{tbl}: 묶음별 '계' 값이 서로 다르다 (최대 차이 {spread.max():.2f})")
            d = d.sort_values(code).drop_duplicates(index + ["ITM_ID"])
    if d.duplicated(index + ["ITM_ID"]).any():
        raise ValueError(f"{tbl}: 같은 칸에 값이 여러 개다. who/by 조건을 더 좁힌다 (who={who!r}, by={by!r})")
    return d.pivot_table(index=index, columns="ITM_ID", values="value")


kosis_wide = _kosis_wide


def load_survey_kosis() -> pd.DataFrame:
    """이민자체류실태및고용조사 KOSIS 통계표 → `curated/survey_immigrants.csv`와 같은 긴 형식(외국인 기준).

    Columns: table, year, category, value, unit, source.  범주 이름은 손으로 옮긴 CSV와 같게 맞췄다.
    보도자료 2개 연도만 옮겼던 것과 달리 표가 있는 모든 연도가 들어온다(취업 2012~, 소득·송금 2017~ 2년 주기).
    """
    rows = []

    def add(table, wide, mapping, unit, tbl, scale=None):
        for cat, ids in mapping.items():
            ids = [ids] if isinstance(ids, str) else ids
            v = wide[ids].sum(axis=1, min_count=1)
            if scale is not None:
                v = v / wide[scale] * 100
            rows.extend({"table": table, "year": int(y), "category": cat, "value": float(x), "unit": unit,
                         "source": f"KOSIS {tbl}"} for y, x in v.dropna().items())

    add("employment_status", _kosis_wide("DT_2FB007F"), {
        "취업자": "T100", "임금근로자": "T200", "상용근로자": "T210", "임시·일용근로자": "T220", "비임금근로자": "T300",
        "고용원이 있는 자영업자": "T310", "고용원이 없는 자영업자": "T320", "무급가족종사자": "T330"}, "천명", "DT_2FB007F")
    add("industry", _kosis_wide("DT_2FB021F"), {
        "농림어업": "T200", "광·제조업": "T300", "건설업": "T400", "도소매·숙박·음식점업": "T500",
        "전기·운수·통신·금융업": "T600", "사업·개인·공공서비스업": "T700"}, "천명", "DT_2FB021F")
    add("employed_by_visa", _kosis_wide("DT_2FB001F", who=None), {
        "비전문취업(E-9)": "T200", "방문취업(H-2)": "T300", "전문인력(E-1~E-7)": "T400", "유학생(D-2·D-4)": "T500",
        "재외동포(F-4)": "T600", "영주(F-5)": "T700", "결혼이민(F-6)": "T800", "기타": "T900"}, "천명", "DT_2FB001F")
    add("income", _kosis_wide("DT_2FI001F"), {
        "없음": "T200", "100만 원 미만": "T300", "100~200만 원": "T400", "200~300만 원": "T500", "300만 원 이상": "T600"},
        "%", "DT_2FI001F", scale="T100")
    add("spending", _kosis_wide("DT_2FI006F"), {
        "생활비": "T200", "주거비": "T300", "국내외 송금": "T400", "저축": "T500", "기타": "T600"}, "%", "DT_2FI006F")
    add("remit_freq", _kosis_wide("DT_2FI004F"), {
        "송금함": "T200", "연 1~11회": ["T210", "T220", "T230", "T240", "T250", "T260", "T270"], "연 12회 이상": "T280"},
        "%", "DT_2FI004F", scale="T100")
    add("remit_amount", _kosis_wide("DT_2FI005F"), {
        "100만 원 미만": "T0200", "100~200만 원": "T0300", "200~300만 원": "T0400", "300~500만 원": "T0500",
        "500~1000만 원": "T0600", "1000~1500만 원": "T0700", "1500~2000만 원": "T0800", "2000만 원 이상": "T0900"},
        "% of senders", "DT_2FI005F", scale="T0100")
    add("hardship", _kosis_wide("DT_2FI015F"), {"경험함": "T200"}, "%", "DT_2FI015F", scale="T100")
    add("hardship", _kosis_wide("DT_2FI016R"), {
        "병원비 부담으로 진료 못함": "T200", "학비 마련 어려움": "T300", "비자발적 실직": "T400", "공과금 기한 내 미납": "T500"},
        "% of 경험자", "DT_2FI016R")
    add("population", _kosis_wide("DT_2FF001F"), {"15세 이상 상주 외국인": "T100"}, "천명", "DT_2FF001F")
    return pd.DataFrame(rows)


def load_survey_by_visa_kosis(year: int = 2025) -> pd.DataFrame:
    """체류자격별 지표를 KOSIS에서 읽는다. 열 이름은 `curated/survey_by_visa_2025.csv`와 같다.

    체류자격별 소득 분포(inc_*)는 KOSIS 공개표에 없어 포함하지 않는다(보도자료 부록에만 있다).
    """
    def at(tbl, who="외국인"):
        return _kosis_wide(tbl, who=who, by="visa").xs(year, level="year")

    ten, spend, hard, htype, stay = at("DT_2FF015F"), at("DT_2FI006F"), at("DT_2FI015F"), at("DT_2FI016R"), at("DT_2FK003R")
    wish = at("DT_2FK008R", who=None)
    out = pd.DataFrame({
        "pop_k": ten["T100"], "own_home_k": ten["T200"].fillna(0), "rent_k": ten["T300"], "free_k": ten["T400"],
        "remit_share": spend["T400"], "saving_share": spend["T500"],
        "hardship": hard["T200"] / hard["T100"] * 100, "hardship_utility": htype["T500"],
        "stay_lt6m": stay["T200"].fillna(0), "stay_6m_1y": stay["T300"], "stay_1_3y": stay["T400"], "stay_3_5y": stay["T500"],
        "stay_5_10y": stay["T600"], "stay_10yp": stay["T700"], "wish_stay": wish["T200"].reindex(ten.index),
    })
    out = out.reindex(list(_VISA_LABEL)).reset_index().rename(columns={"index": "visa"})
    out.insert(1, "visa_label", out.visa.map(_VISA_LABEL))
    return out


def load_children_kosis() -> pd.DataFrame:
    """자녀 유·무 및 자녀 거주지 (KOSIS DT_2FF012F, 2017~). 열 이름은 `curated/survey_children.csv`와 같다."""
    d = load_kosis("DT_2FF012F")
    rows = []
    for who, by, label in (("외국인", "계", "외국인"), ("외국인", "남자", "외국인 남자"), ("외국인", "여자", "외국인 여자"),
                           ("귀화허가자", "계", "귀화허가자")):
        w = d[(d.C1_NM == who) & (d.C2_NM == by)].pivot_table(index="year", columns="ITM_ID", values="value")
        for y, r in w.iterrows():
            rows.append({"year": int(y), "group": label, "pop_k": r["T100"], "with_children_k": r["T300"],
                         "all_in_korea_k": r["T310"], "partly_korea_k": r["T320"], "all_abroad_k": r["T330"],
                         "no_children_k": r["T200"], "source": "KOSIS DT_2FF012F"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 이민정책연구원 통계브리프 (curated)
def load_mrtc_brief(table: str | None = None) -> pd.DataFrame:
    """이민정책연구원 통계브리프 No. 2025-14 표(신한카드 외국인 개인사업자 가맹점 10,323곳, 2025.6 기준).

    Columns: table, row, col, value, unit, source.  table: registration, nationality, nationality_registration,
    industry, region_industry, revenue_industry, top_region, correlation_ln.
    한 카드사 자료라 전체 외국인 사장님을 대표하지 않는다. 10,323곳에는 가맹 해지 2,288곳이 들어 있다.
    """
    d = pd.read_csv(CURATED / "mrtc_brief_2025_14.csv")
    return d if table is None else d[d.table == table].reset_index(drop=True)


# ---------------------------------------------------------------- 가정값 근거 (KOSIS 개인사업자 부채 + 공시·보도)
_DEBT_DIMS = {"매출액별": ("DT_1PB_1005", "DT_1PB_1015"), "사업기간별": ("DT_1PB_1006", "DT_1PB_1016"),
              "산업분류별": ("DT_1PB_1004", "DT_1PB_1014"), "종사자 유무별": ("DT_1PB_1007", "DT_1PB_1017")}
# 분류 코드: 11 전체, 12 가계대출, 13 사업자대출, 14 은행, 15 비은행(평균대출·대출잔액 기준 연체율) / 21 = 중위대출
_DEBT_COLS = {"11": "", "13": "_biz", "14": "_bank", "15": "_nonbank"}


def load_owner_debt(year: int | None = None) -> pd.DataFrame:
    """국내 개인사업자의 평균대출·연체율 (국가데이터처 일자리행정통계 「개인사업자 부채」, KOSIS, 2017~2024).

    Columns: year, dim(전체·매출액별·사업기간별·산업분류별·종사자 유무별), segment,
             avg_loan(만원)·median_loan·avg_loan_biz·avg_loan_bank·avg_loan_nonbank,
             delinq(%, 대출잔액 기준)·delinq_biz·delinq_bank·delinq_nonbank
    내국인 포함 전체 개인사업자다(외국인만 따로 없다). 평균대출은 가계대출과 사업자대출을 합한 값이다.
    한 표 안에 '전체'가 평균(코드 11)과 중위(코드 21)로 두 번 나오므로 반드시 코드로 구분한다.
    """
    def one(tbl, prefix, dim):
        d = load_kosis(tbl)
        code = "C2" if "C2" in d else "C1"
        seg = d["C1_NM"].str.strip() if code == "C2" else "전체"
        d = d.assign(segment=seg, dim=dim)
        out = d[d[code].isin(_DEBT_COLS)].assign(col=lambda x: prefix + x[code].map(_DEBT_COLS))
        if prefix == "avg_loan":
            med = d[d[code] == "21"].assign(col="median_loan")
            out = pd.concat([out, med])
        return out[["year", "dim", "segment", "col", "value"]]

    parts = [one("DT_1PB_1001", "avg_loan", "전체"), one("DT_1PB_1011", "delinq", "전체")]
    for dim, (loan_t, del_t) in _DEBT_DIMS.items():
        parts += [one(loan_t, "avg_loan", dim), one(del_t, "delinq", dim)]
    long = pd.concat(parts)
    assert not long.duplicated(["year", "dim", "segment", "col"]).any()
    wide = long.pivot(index=["year", "dim", "segment"], columns="col", values="value").reset_index()
    wide.columns.name = None
    return wide if year is None else wide[wide.year == year].reset_index(drop=True)


def load_assumption_evidence() -> pd.DataFrame:
    """가정값 근거 중 공시·보도·국제기구 자료에서 옮긴 수치. grade 열에 신뢰 등급을 적었다."""
    return pd.read_csv(CURATED / "assumption_evidence.csv")


def load_ecos_lending_rates() -> pd.DataFrame:
    """한국은행 ECOS 121Y006 예금은행 대출금리(신규취급액 기준, 월). Columns: period, item, rate."""
    rows = []
    for p in sorted((RAW / "ecos").glob("ecos_121Y006_*.json")):
        for r in json.load(open(p, encoding="utf-8")).get("StatisticSearch", {}).get("row", []):
            rows.append({"period": r["TIME"][:4] + "-" + r["TIME"][4:], "item": re.sub(r"\s*\d\)$", "", r["ITEM_NAME1"]),
                         "rate": float(r["DATA_VALUE"])})
    return pd.DataFrame(rows).drop_duplicates(["period", "item"], keep="last")


# ---------------------------------------------------------------- NTS (e-나라지표)
def load_nts_foreign_income_tax() -> pd.DataFrame:
    """외국인 종합소득세 신고 인원·세액 (국세청, e-나라지표 2849). Columns: year, filers, tax_eok."""
    t = pd.read_html(RAW / "index_go" / "t284902.html")[0]
    t = t.set_index(t.columns[0])
    vals = t.iloc[1:5].reset_index(drop=True)  # 인원, 증가율, 세액, 증가율
    years = [int(c) for c in t.columns]
    return pd.DataFrame({
        "year": years,
        "filers": pd.to_numeric(vals.iloc[0].values),
        "filers_growth_pct": pd.to_numeric(vals.iloc[1].values),
        "tax_eok": pd.to_numeric(vals.iloc[2].values),
    })


# ---------------------------------------------------------------- FX / remittance
def newest(folder: Path, stem: str, ext: str) -> Path:
    """가장 새 원자료 파일. `stem.ext`(처음 받은 것)와 `stem_YYYYMMDD.ext`(src/refresh.py가 받은 것) 중 최신."""
    dated = sorted(p for p in folder.glob(f"{stem}_*{ext}") if re.fullmatch(r"\d{8}", p.stem[len(stem) + 1:]))
    return dated[-1] if dated else folder / f"{stem}{ext}"


def load_ecos_monthly_fx() -> pd.DataFrame:
    """ECOS 731Y004 monthly average KRW per USD / CNY. Columns: date, KRW/USD, KRW/CNY."""
    out = {}
    for code, label in (("0000001", "KRW/USD"), ("0000053", "KRW/CNY")):
        rows = []
        for line in open(newest(RAW / "fx", f"ecos_731Y004_{code}", ".jsonl"), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for r in json.loads(line).get("StatisticSearch", {}).get("row", []):
                rows.append((pd.Timestamp(r["TIME"][:4] + "-" + r["TIME"][4:] + "-01"), float(r["DATA_VALUE"])))
        out[label] = pd.Series(dict(rows))
    df = pd.DataFrame(out).sort_index()
    df.index.name = "date"
    return df.reset_index()


WB_ISO = {"CN": "중국(CNY)", "VN": "베트남(VND)", "UZ": "우즈베키스탄(UZS)", "TH": "태국(THB)",
          "NP": "네팔(NPR)", "KH": "캄보디아(KHR)", "PH": "필리핀(PHP)", "ID": "인도네시아(IDR)",
          "MN": "몽골(MNT)"}


def load_wb_fx_cross() -> pd.DataFrame:
    """Annual LCU per USD (World Bank PA.NUS.FCRF) -> local currency per 1 KRW, wide by country."""
    d = json.load(open(newest(RAW / "worldbank", "fx_annual", ".json"), encoding="utf-8"))[1]
    df = pd.DataFrame([(r["country"]["id"], int(r["date"]), r["value"]) for r in d if r["value"] is not None],
                      columns=["iso", "year", "lcu_per_usd"])
    w = df.pivot(index="year", columns="iso", values="lcu_per_usd")
    krw = w.pop("KR")
    cross = w.div(krw, axis=0)  # local currency units per 1 KRW
    return cross.rename(columns=WB_ISO).drop(columns=[c for c in cross.columns if c in ("US",)], errors="ignore")


def load_wb_remittances_paid() -> pd.DataFrame:
    d = json.load(open(newest(RAW / "worldbank", "kor_remit_paid", ".json"), encoding="utf-8"))[1]
    df = pd.DataFrame([(int(r["date"]), r["value"]) for r in d if r["value"] is not None],
                      columns=["year", "usd"]).sort_values("year")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- children hypothesis (survey + MOIS)
def load_children() -> pd.DataFrame:
    """자녀 유·무 및 자녀 거주지 (이민자체류실태및고용조사 보도자료 통계표, 천명)."""
    if newest(RAW / "kosis", "DT_2FF012F", ".json").exists():
        return load_children_kosis()   # 2017~, 손으로 옮긴 CSV와 반올림 범위 안에서 같다
    return pd.read_csv(CURATED / "survey_children.csv")


def load_survey_by_visa() -> pd.DataFrame:
    """체류자격별 거주안정성·상환능력 대리지표 (2025 보도자료 통계표, %·천명).

    Derived columns:
      own_home      자기 집 거주 비율(%)
      utility_miss  공과금 기한 내 미납 경험률(인구 대비 %) = 경제적 어려움 경험률 × 공과금 비중 / 100
      inc_300p_earn 소득 있는 사람 중 월 300만+ 비율(%)
    """
    d = pd.read_csv(CURATED / "survey_by_visa_2025.csv")
    if newest(RAW / "kosis", "DT_2FF015F", ".json").exists():
        # 점유형태·지출·경제적 어려움·체류기간·계속체류는 KOSIS 값으로 바꾼다. 체류자격별 소득 분포(inc_*)만
        # KOSIS 공개표에 없어 보도자료에서 옮긴 값을 쓴다.
        k = load_survey_by_visa_kosis(2025)
        inc = d[["visa"] + [c for c in d.columns if c.startswith("inc_")] + ["source_pages"]]
        d = k.merge(inc, on="visa", how="left")
    d["own_home"] = d.own_home_k / d.pop_k * 100
    d["utility_miss"] = d.hardship * d.hardship_utility / 100
    d["inc_300p_earn"] = d.inc_300p / (100 - d.inc_none) * 100
    return d


_MOIS_FILE = RAW / "mois" / "mois_foreign_residents_2024.xlsx"
_SIDO_SUFFIX = ("특별시", "광역시", "특별자치시", "도", "특별자치도")
_STAY_LABELS = ["1년미만", "1년~2년미만", "2년~3년미만", "3년~4년미만", "4년~5년미만", "5년~10년미만", "10년이상", "미상"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan  # '*'(비공개)·'-' 등


def _is_sido(name: str) -> bool:
    return name.endswith(_SIDO_SUFFIX) and name not in ("세종시",)


def _is_ilban_gu(name: str, seen_cities: set) -> bool:
    m = re.match(r"^(.+?시)(.+구)$", name)
    return bool(m) and m.group(1) in seen_cities


def load_mois_region() -> pd.DataFrame:
    """행정안전부 「2024 지방자치단체 외국인주민 현황」('24.11.1.) → 시군구 1행.

    Columns: 시도, 시군구, total_pop, fr_total(B), nonnat(C), worker, marriage, student, compatriot,
             other, naturalized(D), children(E), stay_10y, stay_5_10y, stay_known
    일반구 행(예: 수원시장안구)은 상위 시 행과 중복이므로 제외.
    """
    import openpyxl
    wb = openpyxl.load_workbook(_MOIS_FILE, read_only=True, data_only=True)

    # -- sheet 1-2: 유형 및 지역별(시군구)
    rows = list(wb["1-2. 유형 및 지역별(시.군.구)"].iter_rows(values_only=True, max_col=30))
    recs, sido, seen = [], None, set()
    for r in rows:
        name = (str(r[0]).strip() if r[0] is not None else "")
        if not name or name in ("구분", "전국") or "1-2." in name:
            continue
        if _is_sido(name):
            sido = name
            continue
        if _is_ilban_gu(name, seen):
            continue
        if name.endswith("시"):
            seen.add(name)
        recs.append({
            "시도": sido, "시군구": name, "total_pop": _num(r[1]), "fr_total": _num(r[3]),
            "nonnat": _num(r[6]), "worker": _num(r[9]), "marriage": _num(r[12]), "student": _num(r[15]),
            "compatriot": _num(r[18]), "other": _num(r[21]), "naturalized": _num(r[24]), "children": _num(r[27]),
        })
    df = pd.DataFrame(recs)

    # -- sheet 3-2: 체류기간별(시군구) — 지역명 행 뒤에 기간별 행이 이어지는 블록 구조
    rows = list(wb["3-2. 체류기간별(시.군.구)"].iter_rows(values_only=True, max_col=2))
    stay, cur, sido = {}, None, None
    for r in rows:
        name = (str(r[0]).strip() if r[0] is not None else "")
        if not name:
            continue
        if name in _STAY_LABELS:
            if cur is not None:
                stay.setdefault(cur, {})[name] = _num(r[1])
            continue
        if name == "전국" or "3-2." in name or name == "구분":
            cur = None
            continue
        if _is_sido(name):
            sido, cur = name, (name, "__SIDO__")  # 하위 시군구가 없는 시도(세종)는 시도 블록을 사용
            continue
        cur = (sido, name)
    for (sd, nm) in [k for k in stay if k[1] == "__SIDO__"]:
        has_sub = any(k[0] == sd and k[1] != "__SIDO__" for k in stay)
        v = stay.pop((sd, nm))
        if not has_sub:
            sub = df[df["시도"] == sd]["시군구"]
            if len(sub) == 1:
                stay[(sd, sub.iloc[0])] = v
    s = pd.DataFrame([{"시도": k[0], "시군구": k[1], **v} for k, v in stay.items()])
    s["stay_10y"] = s["10년이상"]
    s["stay_5_10y"] = s["5년~10년미만"]
    s["stay_known"] = s[[c for c in _STAY_LABELS if c != "미상"]].sum(axis=1, min_count=1)
    df = df.merge(s[["시도", "시군구", "stay_10y", "stay_5_10y", "stay_known"]], on=["시도", "시군구"], how="left")
    return df
