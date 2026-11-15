# 외국인 사장님 금융 — 공모전 데이터 분석

케이뱅크·카카오뱅크 「사장님을 위한 금융 아이디어 공모전」 참가 팀의 데이터 분석 저장소다.
팀 최종 초안은 **① 비자 스텝업 마이너스 통장**(매출·KIIP·영주·귀화 마일스톤에 따라 금리 인하)과 **② 선배 귀화인 커뮤니티 P2P 대출**(귀화 멘토가 10% 후순위 투자)을 제안한다.

## 먼저 볼 것
| 파일 | 내용 |
|---|---|
| `초안_근거_매핑.md` | 초안의 주장마다 붙일 근거, 숫자로 확인된 위험, 고칠 문장 5곳 |
| `figures/` | 제안서용 그림 21장(300dpi PNG). 설명은 `figures/README.md` |
| `가정값_근거.md` | 금리·연체율·대출액 가정값과 공개 자료 비교 |
| `data/DATA_REVIEW.md` | 받은 자료의 품질 검수 결과 |
| `data/SOURCES.md` | 원자료 출처 URL과 수집일 |

## 핵심 숫자
- 외국인 사장님 5.6만 명(2025, 범위 1.0만~25.3만), 2012년 대비 2.9배
- KIIP 참여자 9.0만 명(2025, 10년 동안 3.5배), 영주(F-5) 23.4만 명, 해마다 1만 명 넘게 귀화
- 초안 진입 금리 14~16%는 카드론 평균 13.5%(보도 인용)보다 높다 → 역선택 위험
- 10% 후순위 멘토는 원금 비율만큼만 이자를 받으면 손실률 0.6%를 넘을 때 손해다

## 폴더 구조
```
data/raw/        원자료 (KOSIS·법무부·행안부·국세청·한국은행·World Bank)
data/curated/    보도자료·통계브리프 표를 손으로 옮긴 CSV (행마다 출처)
data/processed/  노트북이 만든 가공 데이터
src/             로더(load.py), 지표 계산(metrics.py), 자료 갱신(refresh.py), 그림 선별(figures_select.py)
notebooks/       분석 노트북 3개 (.py가 원본, .ipynb는 실행 결과)
api/             FastAPI 서버
tests/           pytest
```

## 실행
```bash
pip install -r requirements.txt
pytest -q
uvicorn api.main:app --reload        # http://127.0.0.1:8000/docs
python -m src.refresh                # 원자료 새로 받기 (KOSIS는 .env의 KOSIS_API_KEY 필요)
```
- 인증키는 `.env`에 둔다(`KOSIS_API_KEY=...`). `.env`는 저장소에 올리지 않는다.
- 노트북 실행 순서: `foreign_entrepreneur_finance` → `children_hypothesis` → `target_evidence`. 실행 뒤 `python -m src.figures_select`로 제안서용 그림만 남긴다.

## 한계
- 외국인 사장님을 직접 세는 공개 통계가 없어 여러 자료로 범위를 잡았다.
- 금리·연체율 일부는 보도에서 옮긴 값이라 원문 공시 확인이 필요하다.
- 초안의 금리·한도·수수료는 가정이다. 상관은 인과가 아니다.
- 이민정책연구원 통계브리프 원문 PDF는 재배포 금지 문구가 있어 올리지 않았다(숫자만 `data/curated/`에 출처와 함께 옮겼다).
