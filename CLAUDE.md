# F1 — 외국인 사장님 금융 공모전 데이터 분석

## 프로젝트 맥락
- 케이뱅크·카카오뱅크 「사장님을 위한 금융 아이디어 공모전」(https://2bank.spectory.net/idea/contest?bno=61054) 참가용 분석이다.
- 팀 아이디어(2026-11-15 최종 초안): **① 비자 스텝업 마이너스 통장**(매출·KIIP·영주·귀화 마일스톤에 따라 금리 인하) + **② 선배 귀화인 커뮤니티 P2P 대출**(귀화 멘토가 10% 후순위 투자). 초안 원문은 `data/raw/team/proposal_draft_final.pdf`, 근거 정리는 `초안_근거_매핑.md`. 처음 아이디어였던 해외 정산·송금은 초안에서 빠졌다. 4인 팀이다.
- 이 폴더 담당자(성원)의 역할: 타겟층 인원 조사, 그리고 "돈이 된다·갚을 수 있다·한국에 남는다"를 데이터로 증명하는 일이다.
- 담당자는 통계·코딩 전공자가 아니다. 결과를 설명할 때는 쉬운 말로 쓰고, 숫자에는 출처와 한계를 붙인다.

## 폴더 구조
```
data/raw/         원자료 (법무부·통계청·국세청·한국은행·World Bank·행안부). 출처는 data/SOURCES.md
data/curated/     보도자료 표를 손으로 옮긴 CSV (각 행에 출처 페이지). 지금은 KOSIS 값의 대조용이고, 체류자격별 소득 분포만 분석에 쓴다
data/processed/   노트북이 만든 가공 데이터 (API가 읽는다)
data/DATA_REVIEW.md  데이터 검수 결과 (표별 품질·쓸모 판정, 아직 없는 자료)
src/load.py       원자료 로더          src/viz.py    공통 차트 스타일
src/metrics.py    지표 계산 함수 (API·노트북 공용) ← 계산 로직은 여기에만 둔다
src/refresh.py    원자료 새로 받아오기 (한국은행 환율·World Bank·법무부 월보·통계청 KOSIS)
.env              인증키 (KOSIS_API_KEY, ECOS_API_KEY). 코드·문서·대화에 키 값을 적지 않고, 외부에 올리지 않는다
src/mdis_test.py  MDIS 마이크로데이터 검정 (원자료 대기 중)
notebooks/        foreign_entrepreneur_finance(시장·세분화) / children_hypothesis(가설검정) / target_evidence(제안서 근거 Q1~Q4)
api/main.py       FastAPI 서버          tests/        pytest
figures/          제안서용 그림 21장(최종 초안 기준으로 다시 고름) (PNG, 300dpi). 고른 기준·쓰는 순서는 figures/README.md, 나머지 19장은 figures/_archive/
src/figures_select.py  노트북을 다시 실행한 뒤 제안서용 그림만 남기는 스크립트 (`python -m src.figures_select`)
제안서_근거_요약.md  제안서에 옮길 문장과 한계
가정값_근거.md      가정값(대출액·마진·대손율·금리·수수료)과 공개 자료 비교. 가정값을 바꾸기 전에 본다
분석계획.md        다음 데이터 분석 계획(A~H): 자료·방법·그림·진행 순서
초안_근거_매핑.md   팀 최종 초안의 주장별 근거·위험·고칠 문장 (가장 최신)
ROADMAP.md        향후 계획
```

## 실행
```bash
pip install -r requirements.txt
jupytext --to ipynb notebooks/target_evidence.py   # .py(percent 형식)가 원본, .ipynb는 실행 결과
jupyter nbconvert --to notebook --execute --inplace notebooks/target_evidence.ipynb
uvicorn api.main:app --reload                      # http://127.0.0.1:8000/docs
pytest -q
python -m src.figures_select                       # 노트북 실행 뒤: 제안서용 그림만 figures/에 남긴다
python -m src.refresh                              # 원자료 새로 받아오기 (API에서는 POST /data/refresh)
```
- 새로 받은 환율·송금 파일은 `이름_YYYYMMDD` 꼴로 따로 저장되고 로더(`load.newest`)가 가장 새 파일을 읽는다. 법무부 새 달 파일은 `data/raw/moj/new/`에 받아만 두며, 분석은 2026.9 기준 파일에 고정되어 있다.
- 노트북 실행 순서: foreign_entrepreneur_finance → children_hypothesis → target_evidence (뒤가 앞의 processed CSV를 읽는다).
- Windows 환경이다. 차트 폰트는 Malgun Gothic.

## 작업 규칙
- 노트북은 `.py`(jupytext percent 형식)를 고치고 `.ipynb`는 실행해서 다시 만든다.
- 새 계산은 `src/metrics.py`에 함수로 추가하고, 노트북과 API는 그 함수를 불러 쓴다. 숫자가 바뀌면 `tests/test_api.py`의 기대값도 같이 고친다.
- 가정값(침투율·예대마진·대손율·금리·DSR)은 `ASSUME`(노트북)과 `metrics.SCENARIOS`·`RevenueInput` 기본값에 모아 둔다. 둘을 같은 값으로 유지한다.
- 최종 초안의 상품 가정(한도·단계별 금리·멘토 비중·수수료)은 `metrics.DRAFT_ASSUME` 한 곳에만 둔다. 노트북도 이 값을 읽는다.
- 차트는 `src/viz.py` 스타일을 따른다: 범주색 고정 순서, 크기는 파랑 단일 계열, y축 하나, 제목은 결론 문장, 아래에 출처 한 줄.
- 원자료는 `data/raw/`에 저장하고 `data/SOURCES.md`에 출처 URL·수집일을 한 줄 추가한다. 원자료를 덮어쓰지 않는다.
- 상관을 인과로 쓰지 않는다. 생태학적 상관·구성 효과·시점 차이 같은 한계를 결과 옆에 적는다.
- 한국어 문서는 "~한다" 평서체로 쓴다.

## 현재 핵심 숫자 (target_evidence 기준)
- 사장님 프로필(카드 가맹점 10,323곳, 이민정책연구원 통계브리프 No. 2025-14): 중국 국적 72.5%, 요식·유흥 61%, 연매출 3억 원 이하 74%, 2020년 이후 등록 74%. 한 카드사 기준이다
- 외국인 사장님 5.6만 명(범위 1.0만~25.3만), 2028년 7.3만 명 전망, 예비 사장님 109만 명, 상위 10개 시군구에 38%
- 조사 통계는 KOSIS 통계표(`load.load_survey` 등)에서 읽는다. 손으로 옮긴 값과 반올림 범위 안에서 같다는 것을 테스트가 확인한다
- 외국인 사장님 수는 KOSIS 통계표(2012~2025)에서 읽는다: 2012년 1.9만 → 2025년 5.59만 명, 추세 증가율 연 9.4%(계산법별 8.6~9.4%)
- 수익 시나리오 연 20억 / 76억 / 267억 원 (보수 / 기준 / 낙관)
- 가족정착형 소득자 DSR 40% 통과: 3,000만 원 91%, 5,000만 원 69%
- 시군구 208곳: 사업 가능 외국인 비중 ↔ 10년+ 체류 ρ=0.89
