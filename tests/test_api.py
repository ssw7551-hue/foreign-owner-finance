"""API·지표 회귀 테스트 — 노트북(target_evidence)에서 나온 숫자와 일치하는지 확인한다.

실행 (F1 폴더에서): pytest -q
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.main import app  # noqa: E402
from src import metrics as M  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_target_counts_match_notebook():
    d = client.get("/target/counts").json()
    assert d["core"]["count"] == 55_900
    assert d["upper"]["count"] == 253_050
    assert round(d["core_2028"]) == pytest.approx(73_213, abs=5)
    r = d["core_2028_range"]
    assert r["endpoint"] < r["recent3"] < r["trend"] and 71_000 < r["endpoint"] and r["trend"] < 74_000
    assert d["lower"]["count"] < d["core"]["count"] < d["upper"]["count"] < d["pool"]["count"]


def test_regions_cumulative_share():
    top = client.get("/target/regions", params={"n": 10}).json()
    assert top[0]["label"] == "경기 안산시"
    assert top[-1]["cum_share"] == pytest.approx(0.38, abs=0.01)


def test_monthly_payment_3000():
    assert M.monthly_payment(30_000_000) == pytest.approx(579_984, abs=5)


def test_dsr_group_pass_rates():
    d = client.get("/repayment/dsr", params={"loan_won": 30_000_000}).json()
    assert d["by_group"]["가족정착형"] == pytest.approx(90.9, abs=0.2)
    d5 = client.get("/repayment/dsr", params={"loan_won": 50_000_000}).json()
    assert d5["by_group"]["가족정착형"] > d5["by_group"]["취업·학업형"]


def test_revenue_scenarios_match_notebook():
    s = client.get("/revenue/scenarios").json()
    assert s["기준"]["total_revenue_won"] / 1e8 == pytest.approx(76.2, abs=0.2)
    assert s["보수"]["total_revenue_won"] < s["기준"]["total_revenue_won"] < s["낙관"]["total_revenue_won"]


def test_revenue_simulate_validation():
    assert client.post("/revenue/simulate", json={"penetration": 2}).status_code == 422
    r = client.post("/revenue/simulate", json={"penetration": 0.1}).json()
    assert r["borrowers"] == pytest.approx(M.project_core(2028) * 0.1)


def test_settlement_region():
    rows = client.get("/settlement/region").json()
    stay = next(r for r in rows if r["Y"].startswith("10년"))
    assert stay["spearman_rho"] > 0.8 and stay["n"] == 208


# ---------------------------------------------------------------- 시장 지표·데이터 갱신 (인터넷 없이 돈다)
def test_market_fx_reads_raw():
    d = client.get("/market/fx", params={"months": 12}).json()
    assert d["months"] == 12 and d["latest_period"] >= "2026-10"
    assert 1000 < d["series"]["KRW/USD"]["latest"] < 2000
    assert len(d["monthly"]) == 12


def test_market_remittances():
    rows = client.get("/market/remittances", params={"years": 3}).json()
    assert [r["year"] for r in rows] == sorted(r["year"] for r in rows) and len(rows) == 3
    assert rows[-1]["usd_billion"] > 5


def test_data_status_offline():
    d = client.get("/data/status").json()
    assert d["한국은행 환율(ECOS)"]["auto_refresh"] is True
    assert d["법무부 체류외국인"]["latest_period"] >= "2026-09"
    assert d["이민자체류실태및고용조사"]["auto_refresh"] is False


def test_refresh_rejects_unknown_source():
    assert client.post("/data/refresh", params={"sources": ["nope"]}).status_code == 422


def test_newest_prefers_dated_snapshot(tmp_path):
    from src import load as L
    (tmp_path / "fx_annual.json").write_text("[]")
    assert L.newest(tmp_path, "fx_annual", ".json").name == "fx_annual.json"
    (tmp_path / "fx_annual_20261114.json").write_text("[]")
    (tmp_path / "fx_annual_20270102.json").write_text("[]")
    (tmp_path / "fx_annual_backup.json").write_text("[]")  # 날짜가 아닌 꼬리표는 무시
    assert L.newest(tmp_path, "fx_annual", ".json").name == "fx_annual_20270102.json"


def test_refresh_worldbank_keeps_original_and_skips_shrunk(tmp_path, monkeypatch):
    """새 값이 오면 날짜 붙은 새 파일로 저장하고, 원본은 그대로 두며, 줄어든 응답은 저장하지 않는다."""
    import json
    import shutil
    from src import load as L, refresh as R
    wb = tmp_path / "worldbank"
    shutil.copytree(L.RAW / "worldbank", wb)
    for p in wb.glob("*_????????.json"):
        p.unlink()
    original = (wb / "kor_remit_paid.json").read_bytes()
    monkeypatch.setattr(L, "RAW", tmp_path)

    class Resp:
        def __init__(self, doc): self.content = json.dumps(doc).encode(); self.text = self.content.decode()
        def json(self): return json.loads(self.content)

    def fake(extra):
        def get(client, url, **kw):
            stem = "kor_remit_paid" if "BM.TRF" in url else "fx_annual"
            doc = json.load(open(wb / f"{stem}.json", encoding="utf-8"))
            if stem == "kor_remit_paid":
                doc[1] = doc[1][1:] if extra < 0 else ([dict(doc[1][0], date="2026", value=1.5e10)] * extra + doc[1])
            return Resp(doc)
        return get

    monkeypatch.setattr(R, "_get_retry", fake(1))
    out = R.refresh_worldbank(client=object())
    assert out["한국발 개인송금"]["changed"] and out["한국발 개인송금"]["latest"] == "2026"
    assert out["송금 상대국 환율"]["changed"] is False
    assert (wb / "kor_remit_paid.json").read_bytes() == original
    assert int(L.load_wb_remittances_paid().year.max()) == 2026

    monkeypatch.setattr(R, "_get_retry", fake(-1))
    with pytest.raises(RuntimeError):
        R.refresh_worldbank(client=object())


def test_target_trend_from_kosis():
    d = client.get("/target/trend").json()
    years = [r["year"] for r in d["series"]]
    assert years[0] == 2012 and years[-1] >= 2025
    by = {r["year"]: r for r in d["series"]}
    assert by[2012]["total"] == pytest.approx(19_200) and by[2025]["total"] == pytest.approx(55_900)
    assert d["cagr"]["trend"] == pytest.approx(0.094, abs=0.001)
    assert d["cagr"]["endpoint"] == pytest.approx(0.086, abs=0.001)


def test_kosis_matches_press_release_rounding():
    """KOSIS 값을 천 명으로 반올림하면 보도자료에서 옮겨 적은 값과 같아야 한다."""
    import pandas as pd
    from src import load as L
    se = L.load_self_employed_series().set_index("year")
    cur = pd.read_csv(L.CURATED / "survey_immigrants.csv")
    cur = cur[cur.table == "employment_status"].pivot(index="year", columns="category", values="value")
    for y in (2022, 2023, 2024, 2025):
        assert round(se.employer[y] / 1e3) == cur.loc[y, "고용원이 있는 자영업자"]
        assert round(se.solo[y] / 1e3) == cur.loc[y, "고용원이 없는 자영업자"]


def test_kosis_survey_matches_all_hand_copied_values():
    """보도자료에서 손으로 옮긴 110개 값이 KOSIS 통계표와 반올림 범위 안에서 같아야 한다."""
    import pandas as pd
    from src import load as L
    cur = pd.read_csv(L.CURATED / "survey_immigrants.csv")
    m = cur.merge(L.load_survey_kosis(), on=["table", "year", "category"], how="left", suffixes=("_cur", "_kosis"))
    assert m.value_kosis.notna().all()
    tol = m.unit_cur.map(lambda u: 0.5 if u == "천명" else 0.051)
    bad = m[(m.value_kosis - m.value_cur).abs() > tol]
    assert bad.empty, bad[["table", "year", "category", "value_cur", "value_kosis"]]


def test_kosis_by_visa_matches_hand_copied_values():
    import pandas as pd
    from src import load as L
    cur = pd.read_csv(L.CURATED / "survey_by_visa_2025.csv").set_index("visa")
    k = L.load_survey_by_visa_kosis(2025).set_index("visa")
    cols = [c for c in k.columns if c != "visa_label"]
    diff = (k[cols] - cur[cols]).abs()
    assert diff[[c for c in cols if c.endswith("_k")]].max().max() <= 0.5
    assert diff[[c for c in cols if not c.endswith("_k")]].max().max() <= 0.051


def test_survey_default_source_is_kosis_with_long_series():
    from src import load as L
    sv = L.load_survey()
    assert sv[sv.table == "employment_status"].year.min() == 2012
    assert sorted(sv[sv.table == "remit_freq"].year.unique()) == [2017, 2019, 2021, 2023, 2025]


def test_kosis_wide_never_averages_repeated_labels():
    """한 표에 '계'·'기타'가 여러 번 나와도 섞어서 평균 내지 않는다 (2026-11-15 검수에서 고친 문제)."""
    from src import load as L
    pop = L.kosis_wide("DT_2FF001F")["T100"]                       # 성별·체류자격·국적·지역 묶음마다 '계'가 있다
    assert pop[2025] == pytest.approx(1692.3, abs=0.05)
    by_visa = L.kosis_wide("DT_2FF001F", by="visa")["T100"].xs(2025, level="year")
    assert len(by_visa) == 8 and by_visa.sum() == pytest.approx(pop[2025], abs=0.5)
    assert by_visa["ETC"] == pytest.approx(279.9, abs=0.05)        # 국적 묶음의 '기타'(23.8)와 섞이지 않는다
    nat = L.kosis_wide("DT_2FA003F")                               # 분류 이름이 '국적별'인 표
    assert nat["T111"][2025] == pytest.approx(1109.3, abs=0.05)


def test_owner_profile_from_brief():
    """이민정책연구원 통계브리프 No. 2025-14에서 옮긴 표: 합계가 서로 맞고 원문 숫자와 같아야 한다."""
    from src import load as L
    b = L.load_mrtc_brief()
    for t in ("nationality", "industry", "region_industry", "revenue_industry"):
        assert b[b.table == t].value.sum() == 10_323, t
    reg = b[b.table == "registration"]
    assert reg.value.sum() == 10_323 and reg[reg.col == "가맹해지"].value.sum() == 2_288
    d = client.get("/target/profile").json()
    assert d["merchants"] == 10_323 and d["active"] == 8_035
    assert d["nationality"][0]["name"] == "중국" and d["nationality"][0]["share_pct"] == pytest.approx(72.5, abs=0.05)
    assert d["revenue_bands"][0]["share_pct"] == pytest.approx(73.9, abs=0.05)
    assert d["cohorts"][-1]["share_pct"] == pytest.approx(73.5, abs=0.05)
    assert d["cohorts"][-1]["still_active_pct"] == pytest.approx(75.4, abs=0.05)
    assert d["top_regions"][0] == {"name": "경기 안산시", "count": 450}
    assert d["capital_area_share_pct"] == pytest.approx(72.2, abs=0.05)


def test_assumption_evidence():
    """가정값 근거: KOSIS 개인사업자 부채는 평균(코드 11)과 중위(코드 21)를 섞지 않고 읽어야 한다."""
    from src import load as L
    debt = L.load_owner_debt(2024).set_index(["dim", "segment"])
    tot = debt.loc[("전체", "전체")]
    assert tot.avg_loan == pytest.approx(17_912, abs=1) and tot.median_loan == pytest.approx(10_000, abs=1)
    assert tot.avg_loan_biz == pytest.approx(9_976, abs=1)
    assert tot.delinq == pytest.approx(0.97, abs=0.005) and tot.delinq_biz == pytest.approx(1.16, abs=0.005)
    assert tot.delinq_bank < tot.delinq < tot.delinq_nonbank
    small = debt.loc[("매출액별", "3천만원 미만")]
    assert small.avg_loan_biz == pytest.approx(4_407, abs=1) and small.delinq_biz == pytest.approx(2.94, abs=0.005)
    d = client.get("/assumptions/evidence").json()
    assert d["as_of_year"] == 2024 and len(d["owner_debt_like_target"]) == 7
    assert d["current_assumptions"]["기준"]["credit_loss"] == 0.010          # 근거를 모았을 뿐 가정값은 그대로다
    sens = d["sensitivity_base_total_eok"]
    assert sens["마진 2.5%·대손 1.5%"] < sens["현재 기준(마진 3.5%·대손 1.0%)"]
    assert all(r["url"].startswith("http") for r in d["external"])


def test_target_weighted_delinquency():
    """분석 A: 타겟 구성으로 가중한 연체율. 가중값은 구간별 최솟값과 최댓값 사이에 있어야 한다."""
    d = client.get("/assumptions/target-delinquency").json()
    assert d["as_of_year"] == 2024 and set(d["dims"]) == {"매출 구간", "업력", "직원 유무", "업종"}
    for name, v in d["dims"].items():
        assert v["biz"]["low"] <= v["biz"]["mid"] <= v["biz"]["high"], name
        assert v["bank"]["mid"] < v["biz"]["mid"], name                  # 은행 대출 연체율이 훨씬 낮다
        assert sum(v["weights"].values()) == pytest.approx(1.0)
    lo, hi = d["summary_biz"]["mid_range"]
    assert 1.0 < lo < hi < 1.5                                            # 2024 기준 1.09~1.40%
    assert d["current_credit_loss_pct"]["기준"] == pytest.approx(1.0)       # 가정값은 바꾸지 않았다


def test_market_funnel_matches_revenue():
    """분석 C: 퍼널의 수익은 수익 시나리오와 같은 숫자여야 한다 (옛 fig14처럼 어긋나면 안 된다)."""
    for name in ("보수", "기준", "낙관"):
        f = client.get("/market/funnel", params={"scenario": name}).json()
        r = M.revenue(M.SCENARIOS[name])
        assert f["loan"][-1]["revenue_won"] + f["remit"][-1]["revenue_won"] == pytest.approx(r["total_revenue_won"])
        assert f["loan"][-1]["people"] == pytest.approx(r["borrowers"])
        assert [s["people"] for s in f["loan"]] == sorted([s["people"] for s in f["loan"]], reverse=True)
    assert client.get("/market/funnel", params={"scenario": "없음"}).status_code == 422


def test_province_conversion():
    """분석 D: 16개 시도(광주·전남 합침), 가맹점 합계 10,323곳, 지수의 가중 평균 = 100."""
    rows = client.get("/target/province-conversion").json()
    assert len(rows) == 16 and sum(r["merchants"] for r in rows) == 10_323
    nat = sum(r["merchants"] for r in rows) / sum(r["business_eligible"] for r in rows) * 1e4
    assert nat == pytest.approx(95.4, abs=0.1)
    by = {r["sido"]: r for r in rows}
    assert by["광주·전남"]["merchants"] == 233 and by["경기"]["index"] < 95 < by["서울"]["index"]


def test_remittance_savings():
    """분석 E: 1인 연 송금액·횟수와 비교 대상별 절감액. 건당 1만 원 은행보다는 1%가 비싸다."""
    d = client.get("/market/remittance-savings").json()
    assert d["annual_remittance_won"] == pytest.approx(M.mean_annual_remittance())
    assert 9 < d["transfers_per_year"] < 12
    assert d["our_annual_cost_won"] == pytest.approx(d["annual_remittance_won"] * 0.01)
    by = {a["label"][:12]: a for a in d["alternatives"]}
    assert by["시중은행 건당 1만 원"]["saving_won"] < 0 < by["시중은행 건당 3만 원"]["saving_won"]
    lo, hi = d["saving_range_won"]
    assert lo < 0 < hi and hi == pytest.approx(d["annual_remittance_won"] * (0.059 - 0.01))
    assert client.get("/market/remittance-savings", params={"our_fee": 0.5}).status_code == 422


def test_visa_change_trend():
    """분석 F: 체류자격을 바꾼 외국인 비율 13.7%(2012) → 43.8%(2025)."""
    d = client.get("/settlement/visa-change").json()
    assert d["total"]["2012"] == pytest.approx(13.7, abs=0.05) and d["total"]["2025"] == pytest.approx(43.8, abs=0.05)
    assert d["by_visa"]["F5"]["2025"] > 95 and d["by_visa"]["E9"]["2025"] < 1
    assert "F5" not in d["wish_stay_latest"]          # 영주권자에게는 묻지 않는 문항


# ---------------------------------------------------------------- 팀 최종 초안 (비자 스텝업 · 선배 귀화인 P2P)
def test_draft_settlement_pipeline():
    d = client.get("/draft/settlement-pipeline").json()
    assert d["kiip_participants"]["2025"] == pytest.approx(90_180) and d["kiip_participants"]["2015"] == pytest.approx(25_795)
    assert d["naturalized_per_year"]["2025"] == pytest.approx(11_344)
    assert d["f5_plan_naturalize_pct"]["2024"] == pytest.approx(62.1, abs=0.05)
    assert d["kiip_avg_learning_months"]["2025"] == pytest.approx(8.9, abs=0.05)
    rn = d["recent_naturalized"]
    assert rn["self_emp_share_of_employed_pct"] > 2 * rn["foreign_self_emp_share_of_employed_pct"]


def test_draft_repayment_and_rates():
    d = client.get("/draft/repayment").json()
    rates = [s["rate"] for s in d["stages"]]
    assert rates == sorted(rates, reverse=True) and rates[0] == pytest.approx(0.15) and rates[3] == pytest.approx(0.121)
    assert d["stages"][0]["pass_by_group"]["가족정착형"] > 90          # 한도 1,500만 원에선 DSR이 걸림돌이 아니다
    assert d["saving_to_mid_won"] == pytest.approx(10_500_000 * (0.15 - 0.121))
    b = client.get("/draft/rate-benchmarks").json()
    labels = [x["label"] for x in b["benchmarks"]]
    assert not any("연체율" in l for l in labels) and any("카드론" in l for l in labels)
    assert all(x["low"] < x["high"] for x in b["draft"])


def test_draft_mentor_pool():
    d = client.get("/draft/mentor-pool", params={"top": 10}).json()
    assert d["naturalized_total"] == pytest.approx(245_578) and d["naturalized_5y_plus"] == pytest.approx(191_946)
    assert d["top"][0]["label"] == "경기 안산시"
    hubs = {h["label"]: h for h in d["draft_hubs"]}
    assert hubs["서울 용산구"]["naturalized_5y_plus"] < 1_000 < hubs["경기 안산시"]["naturalized_5y_plus"]


def test_draft_unit_economics():
    d = client.get("/draft/unit-economics").json()
    p2 = d["p2"]
    assert p2["platform_fee_won"] == pytest.approx(600_000)
    assert 0.005 < p2["mentor_breakeven_loss_rate"] < 0.007                 # 멘토 손익분기 손실률 약 0.6%
    assert all(x["mentor_net_won"] < 0 for x in p2["by_loss"])              # 1.1% 이상이면 멘토가 손해
    assert all(x["investor_loss_won"] == 0 for x in p2["by_loss"])          # 10% 안 손실은 멘토가 모두 진다
    p1 = d["p1"]
    assert p1[0]["margin_pct"] > p1[-1]["margin_pct"] > 0                   # 마일스톤을 밟을수록 마진이 줄지만 양수
