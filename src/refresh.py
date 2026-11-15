"""공개 원자료 새로 받아오기 — 한국은행 환율, World Bank 송금·환율, 법무부 체류외국인 통계.

규칙
- 원자료를 덮어쓰지 않는다. 새로 받은 파일은 이름 끝에 받은 날짜(YYYYMMDD)를 붙여 따로 저장한다.
  (법무부 월보는 `stay_연도_월.xlsx`처럼 통계 기준월이 이름이고, 이미 있는 달은 다시 받지 않는다.)
- 받은 내용이 지금 가진 최신 파일과 같으면 저장하지 않는다. 행 수가 줄었으면 오류로 보고 저장하지 않는다.
- `src/load.py`의 환율·송금 로더는 가장 새 파일을 읽는다 (`load.newest`).
- 법무부 새 달 파일은 받아만 둔다. 노트북은 2026.9 기준 파일에 고정되어 있으므로 숫자를 바꾸려면 노트북을 고쳐 다시 실행한다.

실행 (F1 폴더에서):  python -m src.refresh            # 전부
                     python -m src.refresh ecos moj   # 일부만
한국은행 인증키가 있으면 환경변수 ECOS_API_KEY에 넣는다. 없으면 공개 sample 키(1회 10행)를 쓴다.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import httpx

from . import load as L

UA = {"User-Agent": "Mozilla/5.0 (public-data refresh)"}
TIMEOUT = 40.0

ECOS_ITEMS = {"0000001": "원/달러", "0000053": "원/위안"}
ECOS_START = "201601"
WB_BASE = "https://api.worldbank.org/v2"
WB_FX_COUNTRIES = "CN;ID;KH;KR;MN;NP;PH;TH;US;UZ;VN"
MOJ_HOST = "https://www.moj.go.kr"
MOJ_LIST = MOJ_HOST + "/immigration/1569/subview.do"
MOJ_NEW = L.RAW / "moj" / "new"   # 새 달 파일을 받아 두는 곳 (분석은 data/raw/moj 바로 아래 파일만 읽는다)
# 게시글 제목 → (첨부파일 이름에 들어가는 말, 저장할 파일 접두어)
MOJ_TARGETS = [
    (r"(\d{4})년 (\d{1,2})월 출입국외국인정책 통계월보", [("체류외국인 원본 통계표", "stay")]),
    (r"지역별 외국인\(등록\+거소\) 현황\((\d{4})년 (\d{1,2})월말", [("자격별", "regres_region_visa"), ("국적별", "regres_region_nat")]),
]


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _client() -> httpx.Client:
    return httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True)


def _get_retry(client: httpx.Client, url: str, tries: int = 3, timeout: float = 90.0) -> httpx.Response:
    """느린 서버(World Bank)용: 시간 초과·일시 오류면 몇 번 더 시도한다."""
    last = None
    for _ in range(tries):
        try:
            r = client.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
    raise last


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------- 한국은행 ECOS
def _ecos_rows_from_file(path: Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows += json.loads(line).get("StatisticSearch", {}).get("row", [])
    return rows


def _ecos_fetch(client: httpx.Client, code: str, end: str) -> list[dict]:
    key = _env_key("ECOS_API_KEY") or "sample"
    step = 10 if key == "sample" else 1000
    pages, start, total = [], 1, None
    while total is None or start <= total:
        url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{start}/{start + step - 1}"
               f"/731Y004/M/{ECOS_START}/{end}/{code}/0000100")
        body = client.get(url).json()
        if "StatisticSearch" not in body:
            raise RuntimeError(f"ECOS 응답 오류: {json.dumps(body, ensure_ascii=False)[:200]}")
        total = int(body["StatisticSearch"]["list_total_count"])
        pages.append(body)
        start += step
    return pages


def refresh_ecos(client: httpx.Client | None = None) -> dict:
    own = client is None
    client = client or _client()
    out = {}
    try:
        end = date.today().strftime("%Y%m")
        for code, label in ECOS_ITEMS.items():
            stem = f"ecos_731Y004_{code}"
            cur = L.newest(L.RAW / "fx", stem, ".jsonl")
            old = {r["TIME"]: r["DATA_VALUE"] for r in _ecos_rows_from_file(cur)}
            pages = _ecos_fetch(client, code, end)
            new = {r["TIME"]: r["DATA_VALUE"] for p in pages for r in p["StatisticSearch"]["row"]}
            if len(new) < len(old):
                raise RuntimeError(f"{label}: 받은 행({len(new)})이 기존({len(old)})보다 적어 저장하지 않는다")
            if new == old:
                out[label] = {"changed": False, "latest": max(new), "file": cur.name}
                continue
            path = L.RAW / "fx" / f"{stem}_{_today()}.jsonl"
            _save(path, ("\n".join(json.dumps(p, ensure_ascii=False) for p in pages) + "\n").encode("utf-8"))
            out[label] = {"changed": True, "latest": max(new), "previous_latest": max(old),
                          "new_months": len(set(new) - set(old)), "file": path.name}
    finally:
        if own:
            client.close()
    return out


# ---------------------------------------------------------------- World Bank
def _wb_points(doc) -> dict:
    return {(r["country"]["id"], r["date"]): r["value"] for r in doc[1] if r["value"] is not None}


def refresh_worldbank(client: httpx.Client | None = None) -> dict:
    own = client is None
    client = client or _client()
    jobs = {
        "한국발 개인송금": ("kor_remit_paid", f"{WB_BASE}/country/KOR/indicator/BM.TRF.PWKR.CD.DT?format=json&per_page=100"),
        "송금 상대국 환율": ("fx_annual", f"{WB_BASE}/country/{WB_FX_COUNTRIES}/indicator/PA.NUS.FCRF?format=json&per_page=1000&date=2005:{date.today().year}"),
    }
    out = {}
    try:
        for label, (stem, url) in jobs.items():
            cur = L.newest(L.RAW / "worldbank", stem, ".json")
            old = _wb_points(json.load(open(cur, encoding="utf-8")))
            r = _get_retry(client, url)
            doc = r.json()
            if not (isinstance(doc, list) and len(doc) == 2 and doc[1]):
                raise RuntimeError(f"World Bank 응답 오류({label}): {r.text[:200]}")
            new = _wb_points(doc)
            if len(new) < len(old):
                raise RuntimeError(f"{label}: 받은 값({len(new)})이 기존({len(old)})보다 적어 저장하지 않는다")
            latest = max(d for _, d in new)
            if new == old:
                out[label] = {"changed": False, "latest": latest, "file": cur.name}
                continue
            path = L.RAW / "worldbank" / f"{stem}_{_today()}.json"
            _save(path, r.content)
            out[label] = {"changed": True, "latest": latest, "previous_latest": max(d for _, d in old),
                          "new_or_revised_values": sum(1 for k, v in new.items() if old.get(k) != v), "file": path.name}
    finally:
        if own:
            client.close()
    return out


# ---------------------------------------------------------------- 법무부 통계월보 게시판
_A = r'href="(/bbs/immigration/227/\d+/{kind}\.do)[^"]*"[^>]*>(.*?)</a>'


def _links(html: str, kind: str) -> list[tuple[str, str]]:
    return [(m.group(1), re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip())
            for m in re.finditer(_A.format(kind=kind), html, re.S)]


def _moj_newest(prefix: str) -> tuple[int, int]:
    """분석이 쓰는 폴더(data/raw/moj)에 있는 가장 최근 (연, 월)."""
    found = [re.search(rf"{prefix}_(\d{{4}})_(\d{{2}})\.xlsx$", p.name) for p in (L.RAW / "moj").glob(f"{prefix}_*.xlsx")]
    return max(((int(m.group(1)), int(m.group(2))) for m in found if m), default=(0, 0))


def refresh_moj(client: httpx.Client | None = None) -> dict:
    """게시판 첫 페이지에서 통계월보·지역별 현황 글을 찾아, 지금 가진 것보다 새 달의 엑셀만 new/ 폴더에 받는다."""
    global MOJ_NEW
    MOJ_NEW = L.RAW / "moj" / "new"
    own = client is None
    client = client or _client()
    out = {"downloaded": [], "already_have": [], "note": ""}
    try:
        posts = _links(client.get(MOJ_LIST).text, "artclView")
        if not posts:
            raise RuntimeError("법무부 게시판 목록을 읽지 못했다 (양식 변경 가능)")
        for pattern, wants in MOJ_TARGETS:
            for href, title in posts:
                m = re.search(pattern, title)
                if not m:
                    continue
                y, mo = int(m.group(1)), int(m.group(2))
                names = {prefix: f"{prefix}_{y}_{mo:02d}.xlsx" for _, prefix in wants}
                if all((y, mo) <= _moj_newest(prefix) for prefix in names):
                    continue  # 이미 가진 달이거나 그보다 예전 달
                targets = {prefix: MOJ_NEW / n for prefix, n in names.items()}
                if all(p.exists() for p in targets.values()):
                    out["already_have"] += [f"new/{p.name}" for p in targets.values()]
                    continue
                files = _links(client.get(MOJ_HOST + href).text, "download")
                for word, prefix in wants:
                    path = targets[prefix]
                    if path.exists() or (y, mo) <= _moj_newest(prefix):
                        continue
                    hit = [h for h, name in files if word in name and name.endswith(".xlsx")]
                    if not hit:
                        raise RuntimeError(f"'{title}' 글에서 '{word}' 엑셀 첨부를 찾지 못했다")
                    data = client.get(MOJ_HOST + hit[0]).content
                    if data[:2] != b"PK":
                        raise RuntimeError(f"{path.name}: 받은 파일이 엑셀 형식이 아니다")
                    _save(path, data)
                    out["downloaded"].append(f"new/{path.name}")
        out["analysis_uses"] = "%d-%02d" % _moj_newest("stay")
        if out["downloaded"] or out["already_have"]:
            out["note"] = ("새 달 파일은 data/raw/moj/new/에 받아 두었다. 분석과 핵심 숫자는 data/raw/moj/의 파일 기준이라 그대로다. "
                           "새 달로 바꾸려면 파일을 data/raw/moj/로 옮기고 notebooks/foreign_entrepreneur_finance.py의 "
                           "파일 이름·기준월 표기를 고쳐 다시 실행한다.")
        else:
            out["note"] = "게시판에 지금 가진 것보다 새 달 통계가 아직 없다."
    finally:
        if own:
            client.close()
    return out


# ---------------------------------------------------------------- 통계청 KOSIS (인증키 필요)
KOSIS_ORG = "101"
KOSIS_TABLES = {  # 이민자체류실태및고용조사(상주인구기준) 통계표
    "DT_2FB007F": "종사상 지위별 취업자",
    "DT_2FB001F": "체류자격별 취업자",
    "DT_2FB002F": "국적별 취업자",
    "DT_2FB021F": "산업별 취업자(11차)",
    "DT_2FA002F": "체류자격별 경제활동인구",
    "DT_2FF001F": "15세 이상 국내 상주인구",
    "DT_2FF012F": "자녀 유무 및 자녀 거주지",
    "DT_2FF015F": "점유 형태",
    "DT_2FI001F": "월평균 총소득",
    "DT_2FI006F": "총소득 대비 지출부문별 구성비",
    "DT_2FI004F": "해외송금 여부 및 연간 송금횟수",
    "DT_2FI005F": "연간 해외 송금액",
    "DT_2FI015F": "경제적 어려움 경험 유무",
    "DT_2FI016R": "경험한 경제적 어려움 유형",
    "DT_2FK003R": "한국 체류기간",
    "DT_2FK008R": "계속 체류 희망 여부",
    # 2026-11-15 추가: 타겟 프로필·제도권 편입·정착 보조 지표
    "DT_2FA003F": "국적별 경제활동인구",
    "DT_2FB003F": "거주지역별 취업자",
    "DT_2FB009F": "사업체 종사자 규모별 취업자",
    "DT_2FB010F": "동일 직장 근속기간별 취업자",
    "DT_2FI002R": "국민연금 가입유무",
    "DT_2FI003F": "건강보험 가입유무",
    "DT_2FK001R": "한국 최초 입국시점",
    "DT_2FK005R": "체류자격 변경 유무 및 변경 횟수",
    "DT_2FF007F": "배우자 유무 및 배우자 거주지",
    "DT_2FP004R": "본인의 소득 만족도",
    "DT_2FP007R": "한국생활에서 어려운 사항(1순위)",
    # 2026-11-15 추가: 일자리행정통계 「개인사업자 부채」 — 수익 시나리오 가정값(평균 대출액·대손율)의 근거
    "DT_1PB_1001": "개인사업자의 대출 총괄",
    "DT_1PB_1011": "개인사업자의 연체율 총괄",
    "DT_1PB_1005": "개인사업자의 매출액별 대출",
    "DT_1PB_1015": "개인사업자의 매출액별 대출 연체율",
    "DT_1PB_1006": "개인사업자의 사업기간별 대출",
    "DT_1PB_1016": "개인사업자의 사업기간별 대출 연체율",
    "DT_1PB_1004": "개인사업자의 산업분류별 대출",
    "DT_1PB_1014": "개인사업자의 산업분류별 대출 연체율",
    "DT_1PB_1007": "개인사업자의 종사자 유무별 대출",
    "DT_1PB_1017": "개인사업자의 종사자 유무별 대출 연체율",
    # 2026-11-15 추가: 팀 최종 초안(비자 스텝업 마이너스 통장·선배 귀화인 P2P)의 근거
    "DT_1B040A18": "연도별 국적 처리 현황(귀화·회복 등)",
    "DT_1B040A28": "연도별 사회통합프로그램 운영기관 및 참여자",
    "DT_1B040A27": "지역별·이민자별 사회통합프로그램 참여",
    "DT_110025_A040_A": "시군구별 국적취득 경과 기간별 외국인 주민",
    "DT_2FS030R": "향후 한국국적 취득 계획(영주)",
    "DT_2FS029R": "영주권 취득 시기(영주)",
    "DT_2FQ005R": "사회통합프로그램 한국어 학습기간",
}
# 통계청(101)이 아닌 기관의 표: 111 = 법무부 출입국·외국인정책본부, 110 = 행정안전부
KOSIS_ORG_OF = {"DT_1B040A18": "111", "DT_1B040A28": "111", "DT_1B040A27": "111", "DT_110025_A040_A": "110"}


def _env_key(name: str) -> str | None:
    """환경변수 또는 F1/.env 파일에서 인증키를 읽는다. 키는 코드·문서·로그에 적지 않는다."""
    if os.environ.get(name):
        return os.environ[name]
    env = L.ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.strip().partition("=")
            if k == name and v:
                return v
    return None


def _kosis_fetch(client: httpx.Client, key: str, tbl: str) -> list[dict]:
    base = {"method": "getList", "apiKey": key, "orgId": KOSIS_ORG_OF.get(tbl, KOSIS_ORG), "tblId": tbl, "itmId": "ALL", "format": "json",
            "jsonVD": "Y", "startPrdDe": "2012", "endPrdDe": str(date.today().year)}
    last = None
    for prd in ("Y", "F"):       # Y = 매년, F = 2년 주기(소득·송금 문항)
        for n in (3, 2, 1):      # 표마다 분류 수가 달라서 많은 쪽부터 시도한다
            params = dict(base, prdSe=prd, **{f"objL{i}": "ALL" for i in range(1, n + 1)})
            body = client.get("https://kosis.kr/openapi/Param/statisticsParameterData.do", params=params, timeout=90.0).json()
            if isinstance(body, list):
                return body
            last = body
    raise RuntimeError(f"KOSIS 응답 오류({tbl}): {last}")


def refresh_kosis(client: httpx.Client | None = None) -> dict:
    key = _env_key("KOSIS_API_KEY")
    if not key:
        raise RuntimeError("KOSIS 인증키가 없다. F1/.env에 KOSIS_API_KEY=... 한 줄을 넣는다 (kosis.kr/openapi 에서 무료 발급)")
    own = client is None
    client = client or _client()
    folder = L.RAW / "kosis"
    out = {}
    try:
        for tbl, label in KOSIS_TABLES.items():
            try:
                rows = _kosis_fetch(client, key, tbl)
            except Exception as e:  # 한 표가 안 받아져도 나머지는 계속한다
                out[label] = {"changed": False, "error": _hide_keys(f"{type(e).__name__}: {e}")}
                continue
            cur = L.newest(folder, tbl, ".json")
            old = json.load(open(cur, encoding="utf-8")) if cur.exists() else None
            latest = max(r["PRD_DE"] for r in rows)
            if old is not None and len(rows) < len(old):
                raise RuntimeError(f"{label}: 받은 행({len(rows)})이 기존({len(old)})보다 적어 저장하지 않는다")
            if rows == old:
                out[label] = {"changed": False, "latest": latest, "rows": len(rows), "file": cur.name}
                continue
            path = folder / f"{tbl}_{_today()}.json"
            _save(path, json.dumps(rows, ensure_ascii=False).encode("utf-8"))
            out[label] = {"changed": True, "latest": latest, "rows": len(rows), "file": path.name}
    finally:
        if own:
            client.close()
    return out


# ---------------------------------------------------------------- 전체
SOURCES = {"ecos": refresh_ecos, "worldbank": refresh_worldbank, "moj": refresh_moj, "kosis": refresh_kosis}


def _hide_keys(text: str) -> str:
    """오류 문구에 주소가 섞여 나와도 인증키가 드러나지 않게 가린다."""
    for name in ("KOSIS_API_KEY", "ECOS_API_KEY"):
        k = _env_key(name)
        if k:
            text = text.replace(k, "***").replace(k.replace("=", "%3D"), "***")
    return text


def refresh_all(sources: list[str] | None = None) -> dict:
    """고른 출처(기본: 전부)를 새로 받는다. 한 곳이 실패해도 나머지는 계속한다."""
    names = sources or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise ValueError(f"알 수 없는 출처: {unknown} (가능: {list(SOURCES)})")
    result = {"ran_at": datetime.now().isoformat(timespec="seconds")}
    with _client() as client:
        for n in names:
            try:
                result[n] = {"ok": True, **SOURCES[n](client)}
            except Exception as e:  # 네트워크·양식 변경 등: 기존 파일은 그대로 둔다
                result[n] = {"ok": False, "error": _hide_keys(f"{type(e).__name__}: {e}")}
    return result


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")


def status() -> dict:
    """지금 분석이 읽는 원자료가 언제 것인지 (네트워크를 쓰지 않는다)."""
    fx = L.load_ecos_monthly_fx()
    rem = L.load_wb_remittances_paid()
    fx_file = L.newest(L.RAW / "fx", "ecos_731Y004_0000001", ".jsonl")
    rem_file = L.newest(L.RAW / "worldbank", "kor_remit_paid", ".json")
    stay = sorted(p.name for p in (L.RAW / "moj").glob("stay_*.xlsx"))
    months = sorted(re.search(r"stay_(\d{4})_(\d{2})", n).groups() for n in stay)
    return {
        "한국은행 환율(ECOS)": {"latest_period": fx.date.max().strftime("%Y-%m"), "months": int(len(fx)),
                             "file": fx_file.name, "fetched": _mtime(fx_file), "auto_refresh": True, "cycle": "매달"},
        "World Bank 한국발 송금": {"latest_period": str(int(rem.year.max())), "file": rem_file.name,
                                "fetched": _mtime(rem_file), "auto_refresh": True, "cycle": "연 1회"},
        "법무부 체류외국인": {"latest_period": "-".join(months[-1]), "files": len(stay),
                          "waiting_in_new_folder": sorted(p.name for p in (L.RAW / "moj" / "new").glob("*.xlsx")),
                          "auto_refresh": "새 달 파일만 new/ 폴더에 내려받는다 (분석 반영은 노트북 수정·재실행 필요)", "cycle": "매달"},
        "이민자체류실태및고용조사": {"latest_period": "2025-05", "auto_refresh": False, "cycle": "연 1회(12월 발표)",
                             "note": "보도자료 PDF를 data/curated CSV로 옮겨 적는다"},
        "국세청 외국인 종합소득세": {"latest_period": "2024", "auto_refresh": False, "cycle": "연 1회"},
        "행정안전부 외국인주민 현황": {"latest_period": "2024-11", "auto_refresh": False, "cycle": "연 1회"},
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(refresh_all(sys.argv[1:] or None), ensure_ascii=False, indent=2))
