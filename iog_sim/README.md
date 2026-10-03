# iog_sim — IOG 공급사슬 시뮬레이터

IOG(Business Dashboard of DKU) 라운드 운영을 위한 의사결정 시뮬레이터.

1. **일일 운영** — 오늘 시스템에 입력할 값(자재 발주량, 할인율, 차주 생산계획)을 계산한다.
2. **정책 실험(what-if)** — Lot sizing·할인·안전계수를 갈아끼우며 Balance를 비교한다.

## 설치 / 실행

```bash
pip install numpy scipy pandas pytest xlrd
```

실행은 **패키지 폴더의 한 단계 위**에서 `-m`으로 한다.

```bash
cd <repo>                                        # iog_sim 패키지를 담고 있는 디렉터리
python -m pytest iog_sim/tests -q                # 규칙 구현 검증
python -m iog_sim.examples.current_state_report  # 매일: 입력할 값 + 예상 화면 값
python -m iog_sim.examples.round_plan            # 라운드 종료일까지 전체 기준 계획 (baseline CSV)
python -m iog_sim.examples.demo_run              # 정책 4종 비교 (what-if, 약 1분)
python -m iog_sim.examples.safety_stock_compare  # 안전재고 기준별 Balance 비교 (실수요, 여러 기간)
```

## 매일 하는 일 (`examples/current_state_report.py`)

파일 상단의 `▼▼▼ 입력` 블록에 **어제 20시 틱 이후** 화면 값을 옮겨 적고 실행한다.

| 블록 | 상수 | 화면 | 갱신 |
|---|---|---|---|
| ① | `TODAY` | — | 매일 |
| ① | `MATERIAL_INVENTORY_INOUT` | Material > Material inventory in/out summary (Total in, Total out, Current level) | 매일 |
| ① | `MATERIAL_PLANS` | Material > Material plans (자재, 옵션, 수량, 발주일, 입고일 또는 None) | 매일 |
| ① | `SALES_INVENTORY_INOUT` | Sales > Sales inventory in/out summary (Total in, Total out, Current level) | 매일 |
| ① | `CURRENT_BALANCE` | Ledger > Balance | 매일 |
| ② | `LOCKED_PRODUCTION_PLAN` | Production — 제출해 확정된 이번 주 계획 | 토요일 |
| ③ | `OBSERVED_DEMAND` | 수요 파일보다 먼저 알게 된 실측 수요 (선택) | 필요할 때 |

- Current level이 시뮬레이션 시작 재고가 된다.
- Material plans에서 입고일이 None인 행이 입고예정 주문이 된다(도착일 = 발주일 + 리드타임).
- 입력끼리 어긋나면 `[입력 확인]`에 `!` 경고가 뜬다: in − out ≠ Current level,
  자재 out ≠ BOM × 제품 in, 도착 예정일이 지났는데 미입고.

출력 (CSV는 `examples/reports/{이름}_{TODAY}.csv`)

| 표 | 내용 |
|---|---|
| `this_week_rest` | 이번 주 남은 날 — 확정 계획. `*_short > 0`이면 그날 자재 부족 |
| `next_week_plan` | **차주 제출용** Demand 예측 + Production Job 수 / 시퀀스 |
| `daily_input` | **일일 입력** Material 발주량(당일) / Sales 할인율(`discount_target`일 적용) |
| `material_flow` | 예상 Material inventory in/out — 다음 날 화면과 대조 |
| `sales_flow` | 예상 Sales inventory in/out — 다음 날 화면과 대조 |

## 라운드 전체 계획 (`examples/round_plan.py`)

`current_state_report.py`의 입력 블록을 그대로 읽어 TODAY부터 라운드 종료일까지 같은 정책으로 돌린다.
최근 수요로 만든 SES 예측을 종료일까지 평평하게 깔기 때문에 주간 패턴이 반복되는 기준 계획이 나온다.
나중에 daily report 실측값과 대조할 기준선이다.

| 표 | 내용 |
|---|---|
| `round_plan` | 일별 예측수요·Job 수·자재 발주·익일 할인율·예상 재고·예상 Balance, `submit_by` = 그 주 제출 마감(토) |
| `round_plan_weekly` | 주별(일~토) Job 합계·생산일수·발주 합계·주말 재고·주말 Balance, 확정/제출 예정 구분 |

## 구조

```
iog_sim/
  config.py              게임 규칙 상수 (단가·비용·BOM·리드타임). VERIFY = 시스템에서 재확인 필요
  calendar.py            휴장일·라운드 종료일(하드코딩) / 생산가능일 / 결정일(토) / 계획시계
  costs.py               Makespan -> 인건비, 작업준비비
  state.py               WorldState, PurchaseOrder, ProductionOrder, 일별 기록
  ledger.py              생산비/재고비/자재비 3버킷 집계, Balance, MAE·MAPE·fill rate
  engine.py              일별 이벤트 루프 + 주간 계획 (상태 변경은 전부 여기서만)
  experiment.py          what-if 러너, 주간계획/일일입력/자재·판매 흐름 리포트
  demand/
    forecaster.py        SES(α=0.9, 채택 모델) + Naive(기준선)
    generator.py         수요 파일 재생 + 로더 + 미래 구간 예측 보간
    demand_data_P*.*     수요 파일 (최신 날짜 파일이 자동 선택됨)
  production/
    sequencing.py        makespan, SPT/LTWK/NEH(Taillard 가속), BestOf, CSV 로더
    lotsizing.py         L4L / 고정배치 / Wagner-Whitin형 DP (makespan 오라클 내장)
    safety_stock.py      완제품 안전재고 규칙 + 비용 임계비율(판매기회비 vs 재고유지비)
    t_500_20_{요일}.csv  P1 처리시간표   t2_500_20_{요일}.csv  P2 처리시간표
  material/
    mrp.py               BOM 전개 (생산계획 -> 자재 소요), 계획 밖 구간 추정
    policy.py            EOQ, (s,S), M2 이중조달(일반/긴급)
  sales/
    discount.py          한계이익 최적화 할인
  examples/
    current_state_report.py  매일 운영 리포트 (+ CSV 저장)
    round_plan.py            라운드 종료일까지 전체 기준 계획 (+ CSV 저장)
    demo_run.py              정책 비교 런
    safety_stock_compare.py  안전재고 기준별 Balance 비교
```

## 확정된 게임 규칙 (코드에 반영됨)

| 규칙 | 코드 위치 |
|---|---|
| 계획 마감 = 매주 토요일 자정, 차주 = 일~토 | `GameCalendar.decision_weekday = 5`, `next_week()` |
| 당일 생산분 당일 판매 가능 | `SimConfig.produce_then_sell_same_day = True` (lag 0) |
| 생산가능일: P1 평일만 / P2 매일 | `ProductSpec.weekday_production_only` |
| 개장일 = 평일 − 공휴일 (주말 자동 휴장) | `GameCalendar.is_market_day`, `MARKET_HOLIDAYS_BY_MONTH` |
| 라운드마다 초기재고로 리셋 | `ROUND_ENDS`, `GameCalendar.round_end_for` |
| 하루 한 제품 최대 500 Job | `MAX_LOTS_PER_DAY` |
| Total Expense = 생산비 + 재고비 + 자재비 | `LedgerSummary` |
| 리드타임은 달력일 (주말·휴장일 포함) | M1 3일 / M2 일반 8일·긴급 2일 / M3 5일 |

- **휴장일**: `calendar.MARKET_HOLIDAYS_BY_MONTH`에 **평일 공휴일만** 적는다(현재 09-24·25 추석,
  10-05 개천절 대체공휴일, 10-09 한글날). 확인은 Demand 화면에서 P1 입력칸이 0으로 고정된 평일.
- **라운드 종료일**: `calendar.ROUND_ENDS`. 2회차 종료일(12-04)은 가정이다(VERIFY).
- **500 Job 상한**: 처리시간표가 500 Job까지라 그 이상은 makespan을 알 수 없다. 넘는 요청이 오면
  `processing_times()`가 예외를 던진다.

## 엔진이 하는 일

**주간 계획 (토요일)** — 차주 일~토에만 생산을 배정한다. P1은 차주 토·일에 생산할 수 없으므로
금요일 배치가 '다음 일'까지 커버하도록 계획 시계를 늘린다(`coverage_horizon`). 같은 주를 다시
계획하면 그 주의 기존 배정을 먼저 지운다. 토요일에는 계획을 먼저 세우고 그 계획을 보고 20시 발주를 정한다.

**backfill** (`backfill_current_week=True`) — 라운드 중간에 시작하면 start−1을 예측 원점으로
**이번 주 남은 날(~토)만** 계획한다. 다음 주 날짜에는 오더를 만들지 않는다.

**자재 소요 = 생산계획 × BOM** (`material/mrp.py`) — 자재는 생산일에 생산량만큼 소모되므로 발주 정책에도
그 값을 넘긴다. 확정 계획 밖 구간은 같은 요일·같은 종류(개장일/휴장일)의 최근 계획일을 복사해 추정한다.
안전재고용 σ는 소요 계열의 들쭉날쭉함이 아니라 **제품 1-step 예측오차 × BOM**이다(`_material_sigma`).

**완제품 안전재고** (`PolicySet.safety_stock`, `production/safety_stock.py`) — 규칙이 수요일별 누적 버퍼
목표 ss_h를 내고, 엔진이 그 증분을 그날 수요에 더해 Lot sizing에 넘긴다. 서비스수준을 비우면(None)
뉴스벤더 임계비율 Cu/(Cu+Co)를 쓴다: Cu = 판매기회비 250 + 놓친 마진(단가 − 자재비), Co = 재고유지비 30
(P1 약 0.93, P2 약 0.94). 기본값은 구 방식 `EndOfHorizon(0.95)`이다. 기준별 비교는 `safety_stock_compare`.

**라운드 종료** — 종료일에서 시뮬레이션을 멈추고, 종료일 뒤 날짜는 계획하지 않는다. 계획의 커버 구간이
종료일에서 잘리면 남는 재고는 못 팔므로 Co에 자재비를 더한 임계비율로 안전재고를 낮춘다. 종료일 뒤에 도착하는 발주는 버리고, 종료일이 소요 시계(21일) 안에
들어오면 발주량을 '남은 소요 − 재고포지션'으로 깎는다.

**Job 시퀀스** — 실제 투입 순서는 `BestOf([SPT, LTWK, NEH])`(대개 NEH, SPT보다 makespan 약 10% 짧음).
Lot sizing DP가 수백 번 호출하는 `planning_sequencer`는 빠른 LTWK를 쓴다. 따라서 DP가 보는 인건비는
실제보다 약간 크다.

## 시스템 화면 입력 형식

| 항목 | 형식 | 코드 |
|---|---|---|
| Job sequence | **쉼표 구분** 1-based Job ID (`8,4,5,6,1,3,2,7,10,9`) | `SequenceResult.job_ids()` |
| 1 Job | 1,000개 | `LOT_SIZE` |
| 하루 최대 Job | 500 | `MAX_LOTS_PER_DAY` |
| 수요예측 | 정수만 | |
| 발주 수량 | 실수 허용 | |
| 할인율 | 0.0 ~ 1.0 (0.3 = 30%) | |
| 운영 시각 | 매일 20시 | |

Production/Demand 화면의 **Auto 토글은 OFF**로 두어야 우리 계획이 덮어쓰이지 않는다
(Auto forecasting = 5-period MA, Auto production scheduling = SPT).

## 주의

- **리포트의 미래 수요는 예측치다.** 수요 파일은 어제까지만 있으므로 `extend_with_forecast()`가 같은 SES
  예측으로 채운다. 즉 Balance 궤적은 예측오차 0인 낙관 시나리오다.
- **틱 건너뜀**: 시스템이 하루 틱을 빼고 다음 날 이틀치를 몰아 정산한 적이 있다(09-29 -> 10-01).
  재고 버퍼를 하루치만 두면 그날 통째로 품절난다.
- `config.py`의 `VERIFY` 주석(BOM 비율, 라운드 기초재고, 할인 탄력성 ε)은 라운드 시작 시 재확인할 것.
- 할인 정책은 **필수 버퍼(당일+익일 수요)를 잉여로 오인하지 않도록** 가드가 들어 있다.
- 시스템 대시보드와 대조할 때는 `material_flow` / `sales_flow` CSV를 그대로 맞춰보면 된다.
