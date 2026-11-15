"""공모전 제안서에 쓰는 그림만 figures/에 남기고 나머지는 figures/_archive/로 옮긴다.

노트북을 다시 실행하면 모든 그림이 figures/에 다시 생기므로, 실행한 뒤 이 스크립트를 한 번 돌린다.
지우지 않고 옮기기만 한다. 되돌리려면 _archive/의 파일을 figures/로 다시 옮기면 된다.

실행 (F1 폴더에서):  python -m src.figures_select
고른 기준과 쓰는 순서는 figures/README.md에 적었다.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"

KEEP = {
    # 2026-11-15 팀 최종 초안(비자 스텝업 마이너스 통장 · 선배 귀화인 P2P)에 맞춰 다시 골랐다. 송금·옛 수익 시나리오 그림은 보관함으로.
    # 배경·타겟
    "fig01_foreign_residents_trend", "t01_target_count_range", "t08_self_employed_trend", "t12_owner_nationality_industry",
    "t13_owner_revenue_bands", "t02_target_top_regions", "fig07_region_map_top15", "t18_province_conversion",
    # 비자 스텝업: 정착 사다리·금리 경로·상환 부담·위험
    "t21_settlement_ladder", "t22_stepup_interest_path", "t23_rate_benchmarks", "t05_repayment_dsr", "t15_target_weighted_delinquency",
    "t14_owner_cohorts", "t11_stay10y_trend", "t20_visa_change_flow", "t06_settlement_corr_region", "h03_visa_effect_sizes",
    # 커뮤니티 P2P: 멘토 풀·단위경제성
    "t24_mentor_pool", "t25_p2p_mentor_economics",
    # 소득층 존재
    "t03_nts_tax_amount",
}


def select() -> dict:
    archive = FIG / "_archive"
    archive.mkdir(exist_ok=True)
    moved = []
    for p in sorted(FIG.glob("*.png")):
        if p.stem not in KEEP:
            shutil.move(str(p), str(archive / p.name))   # 같은 이름이 있으면 새 그림으로 바꾼다
            moved.append(p.name)
    kept = sorted(p.name for p in FIG.glob("*.png"))
    missing = sorted(k for k in KEEP if not (FIG / f"{k}.png").exists())
    return {"kept": kept, "moved": moved, "missing": missing}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = select()
    print(f"남긴 그림 {len(r['kept'])}장, 옮긴 그림 {len(r['moved'])}장 (figures/_archive/)")
    if r["missing"]:
        print("없는 그림(노트북을 다시 실행해야 한다):", ", ".join(r["missing"]))
