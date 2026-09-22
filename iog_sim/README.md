# iog_sim — IOG 공급사슬 시뮬레이터

IOG(Business Dashboard of DKU) 라운드 운영을 위한 의사결정 시뮬레이터.
용도는 두 가지다.

1. **일일 운영** — 오늘 시스템에 입력할 값(자재 발주량, 할인율)을 계산한다.
2. **정책 실험(what-if)** — Lot sizing·할인·안전계수를 갈아끼우며 Balance를 비교한다.

## 설치 / 실행

```bash
pip install numpy scipy pandas pytest
```

실행은 **패키지 폴더의 한 단계 위**에서 `-m`으로 한다 (`PYTHONPATH` 설정 불필요).

```bash
cd <repo>                                        # iog_sim 패키지를 담고 있는 디렉터리
python -m pytest iog_sim/tests -q                # 규칙 구현 검증 (23 tests)
python -m iog_sim.examples.current_state_report  # 오늘 입력할 값 + 일별 Balance 예상
python -m iog_sim.examples.demo_run              # 정책 4종 비교 (what-if)
```

## 구조

```
iog_sim/
  config.py              게임 규칙 상수 (단가·비용·BOM·리드타임). VERIFY 주석 = 시스템에서 재확인 필요
  calendar.py            휴장일(하드코딩) / 생산가능일 / 의사결정일(토) / 계획시계 / 리드타임(달력일)
  costs.py               Makespan -> 인건비, 작업준비비
  state.py               WorldState, PurchaseOrder, ProductionOrder, 일별 기록
  ledger.py              생산비/재고비/자재비 3버킷 집계, Balance, MAE·MAPE·fill rate
  engine.py              일별 이벤트 루프 + 주간 계획 (상태 변경은 전부 여기서만)
  experiment.py          CRN 기반 what-if 러너, 대응표본 비교, 주간계획/일일입력 리포트
  demand/
    forecaster.py        SES(α=0.9, 채택 모델) + Naive(기준선)
    generator.py         실현 수요 3모드 + CSV 로더 + 미래 구간 예측 보간
    demand_data_P*.csv   실수요 CSV (최신 날짜 파일이 자동 선택됨)
  production/
    sequencing.py        makespan, FCFS/SPT/LTWK/MWKR/NEH(Taillard 가속), BestOf, CSV 로더
    lotsizing.py         L4L / 고정배치 / Wagner-Whitin형 DP (makespan 오라클 내장)
    t_500_20_{요일}.csv  P1 처리시간표   t2_500_20_{요일}.csv  P2 처리시간표
  material/
    mrp.py               BOM 전개
    policy.py            EOQ, (s,S), M2 이중조달(일반/긴급)
  sales/
    discount.py          재고비율 규칙, 한계이익 최적화, 탄력성 추정
  examples/
    current_state_report.py  라운드 중간 상태 -> 일별 입력값 리포트 (+ CSV 저장)
    demo_run.py              정책 비교 스모크 런
```

## 확정된 게임 규칙 (코드에 반영됨)

| 규칙 | 코드 위치 |
|---|---|
| 계획 마감 = 매주 토요일 자정, 차주 = 일~토 | `GameCalendar.decision_weekday = 5`, `next_week()` |
| 당일 생산분 당일 판매 불가 | `SimConfig.produce_then_sell_same_day = False`, `availability_lag=1` |
| **생산가능일은 제품마다 다름** — P1 평일만 / P2 매일 | `ProductSpec.weekday_production_only` |
| 하루 한 제품 최대 500 Job | `MAX_LOTS_PER_DAY` |
| Total Expense = 생산비 + 재고비 + 자재비 | `LedgerSummary.production_cost / inventory_cost / material_cost` |
| 리드타임은 달력일 (주말·휴장일 포함) | M1 3일 / M2 일반 8일·긴급 2일 / M3 5일 |

**생산가능일 (Production 화면 관측)**: P1은 토·일에 입력칸이 아예 없고 0으로 고정, P2는 7일 모두 입력 가능하다.
따라서 `planning_horizon()`과 생산일 배정은 제품별로 따로 계산한다.
P1은 금요일 배치가 '토 + 다음 일 + 다음 월'까지 커버해야 하지만(계획 시계 9일), P2는 토요일에도
생산할 수 있어 '다음 일'만 커버하면 된다(8일). P1에서 이 연장을 빼면 매주 일·월에 구조적 품절이 난다.

**500 Job 상한**: `ProcessingTimeTable`이 500 Job까지만 제공하므로 그 이상은 처리시간을 알 수 없다.
상한을 넘겨 계획하면 makespan이 500 Job 기준으로 계산되어 인건비가 크게 과소평가되고, Lot sizing DP가
"거대 배치가 공짜"라고 착각한다. Lot sizing이 상한을 지키며, 넘는 요청이 오면
`processing_times()`가 예외를 던진다(조용히 잘라내지 않는다).

## 입력 채널과 마감 (시스템 화면 대응)

입력은 **결정 시점이 다른 두 묶음**이고, 리포트도 그에 맞춰 둘로 나뉜다.

| 리포트 | 화면 | 입력 내용 | 적용 대상 | 마감 |
|---|---|---|---|---|
| `weekly_plan_report()` | Demand | 수요예측 | 차주 7일 | 결정일(토) 자정 |
| | Production | 요일별 Job 수 + 시퀀스 | 차주 월~금 | 결정일(토) 자정 |
| `daily_input_report()` | Material | 발주량 | **당일** | 매일 20시 |
| | Sales | 할인율 | **익일** (`discount_target` 열) | 매일 20시 |

주간 계획은 한 주치를 결정일에 **한 번에** 제출하므로 일별 표에 섞으면 안 된다.
`GameCalendar.decision_day_for(d)`가 "d가 속한 주를 제출한 토요일"을 돌려주며,
그 날짜가 지났으면 해당 주의 Demand/Production은 더 이상 바꿀 수 없다.

`daily_input_report()`의 `discount_*`는 **그날 입력할 값(= 익일 적용분)**이다.
엔진은 d일에 d+1일 할인을 정하므로(`_decide_discount`), 화면 입력 시점에 맞춰 한 칸 당겨 싣고
적용일을 `discount_target` 열에 함께 표시한다.

## 데이터 입력

- **수요 CSV**: `demand/demand_data_{P1,P2}_{YYYYMMDD}.csv` (`DATE,DEMAND_QTY`). 새 파일을 넣기만 하면
  파일명 날짜순으로 최신본이 자동 선택된다. P1은 개장일만, P2는 365일 매일. 최신순 정렬이며
  P1 파일 끝의 빈 줄과 BOM 유무 차이는 로더가 흡수한다.
- **처리시간 CSV**: `production/t_500_20_{요일}.csv`(P1) / `t2_500_20_{요일}.csv`(P2). `JobID,M1..M20` 형식.
- **휴장일**: `calendar.MARKET_HOLIDAYS_BY_MONTH`에 **월별로 직접 하드코딩**한다(요일 계산 안 함).
  확인 방법은 **Demand 화면**이다 — P1 입력칸이 없고 `0`으로 고정된 날이 휴장일이다.
  현재 2026-09: 24~27 / 2026-10: 주말 전체(3,4,10,11,17,18,24,25,31).
  **VERIFY 10-09(한글날)** 은 아직 화면에서 확인하지 못했다. 그 주 계획 전에 반드시 확인할 것.

## 시스템 화면이 요구하는 입력 형식

| 항목 | 형식 | 코드 |
|---|---|---|
| Job sequence | **쉼표 구분** 1-based Job ID (`8,4,5,6,1,3,2,7,10,9`) | `SequenceResult.job_ids()` |
| 1 Job | 1,000개 | `LOT_SIZE` |
| 하루 최대 Job | **500** ("Max job is 500") | `MAX_LOTS_PER_DAY` |
| 수요예측 | 정수만 | |
| 발주 수량 | 실수 허용 | |
| 할인율 | 0.0 ~ 1.0 (0.3 = 30%) | |
| 운영 시각 | 매일 20시 | |

Production/Demand 화면의 **Auto 토글은 OFF로 두어야** 우리 계획이 덮어쓰이지 않는다
(Auto forecasting = 5-period MA, Auto production scheduling = SPT).

## 운영 리포트 (`current_state_report.py`)

실행 전 스크립트 상단 상수를 시스템 화면 값으로 갱신한다.

| 상수 | 출처 |
|---|---|
| `TODAY` | — (마감일·대상 주는 여기서 자동 계산된다) |
| `CURRENT_FG_INVENTORY` | 완제품 재고 화면 (Current level) |
| `CURRENT_MAT_INVENTORY` | Material inventory in/out summary (Current level) |
| `CURRENT_BALANCE` | Ledger 화면 Balance |
| `LOCKED_PRODUCTION_PLAN` | **Production 탭** — 지난 결정일에 제출되어 확정된 **이번 주** 계획 |

출력은 세 묶음이다.

1. **이번 주 (확정)** — `LOCKED_PRODUCTION_PLAN` 그대로. 변경 불가, 계산의 전제로만 쓴다.
2. **차주 제출용** — 이번 주 토요일 자정까지 넣을 Demand 예측 + Production Job 수/시퀀스. **이번 주의 산출물.**
3. **일일 입력** — Material(당일 발주) / Sales(익일 할인율).

결과는 `examples/reports/next_week_plan_{실행날짜}.csv`, `daily_input_{실행날짜}.csv`로 저장된다.

동작 방식: `Simulator.run()`에 실측 재고를 주입(`initial_fg_inventory` / `initial_mat_inventory`)하고,
확정된 이번 주 계획을 `locked_production_plan`으로 고정한다. 시작일이 결정일(토)이 아니라
`forecast_cache`가 비어 있는 문제는 `backfill_plan_from`으로 메운다(전날을 결정일로 가정해
`_plan_next_week`를 한 번 미리 돌린다). 차주 계획은 시뮬레이션이 마감일(토)을 지나면서
엔진의 정상 경로로 자동 생성된다.

**`LOCKED_PRODUCTION_PLAN`을 꼭 채울 것.** 이번 주 생산계획은 이미 마감되어 바꿀 수 없으므로,
발주량·할인율은 *재최적화한 계획*이 아니라 *실제 확정된 계획*을 전제로 계산해야 맞는 값이 나온다.

## 주의

- **미래 구간 실현 수요**: 수요 CSV는 어제까지만 있으므로, 그대로 두면 리포트 구간의 수요가 전부
  0이 되어 매출·품절이 통째로 빠진다. `extend_with_forecast()`가 같은 SES 예측치로 채운다.
  **즉 리포트의 Balance 궤적은 예측오차 0인 낙관 시나리오다.** 오차 리스크까지 보려면
  `ErrorInjection` / `BlockBootstrapPath`로 여러 경로를 뽑아 비교해야 한다.
- **라운드 종료 효과는 반영되어 있지 않다.** 정책은 운영이 계속된다고 보고 EOQ·배치를 잡으므로,
  종료가 임박하면 팔 수 없는 물량을 만들고 도착하지 않을 자재를 발주한다. 종료 주에는 리포트 값을
  그대로 쓰지 말고 "도착일 > 마지막 유효 생산일"인 발주를 직접 0으로 낮출 것
  (마지막 유효 생산일 = 종료일 - 1, 당일 생산분 당일 판매 불가 때문).
- `config.py`의 `VERIFY` 주석(BOM 비율, 라운드 기초재고, 할인 탄력성 ε)은 라운드 시작 시 재확인할 것.
- 할인 정책에는 **필수 버퍼(당일+익일 수요)를 잉여로 오인하지 않도록** 가드가 들어 있다.
  `InventoryRatioRule`의 임계 비율 기본값이 1.0이 아니라 2.0인 것도 같은 이유다.
- `PolicySet.sequencer`(실제 투입 순서)와 `PolicySet.planning_sequencer`(Lot sizing DP가
  수백 번 호출하는 싼 추정기)는 반드시 분리해서 쓴다. 메타휴리스틱을 planning 쪽에 꽂으면
  계획 1회가 몇 분씩 걸린다.
- 시스템 대시보드와 대조할 때는 `MaterialRecord`의 `received`(Total in) / `consumed`(Total out) /
  `inventory_end`(Current level)를 그대로 맞춰보면 된다.
