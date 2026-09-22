"""라운드 운영 리포트 - 의사결정 시점별로 나눠서 출력한다.

입력 채널이 두 종류이고 결정 시점이 다르므로 표도 나눈다.

  [주간] Demand / Production : 차주 7일치를 결정일(토) 자정에 한 번에 제출.
                               이번 주 계획은 지난 토요일에 이미 마감되어 변경 불가.
  [일일] Material / Sales    : 매일 20시. Material은 당일 발주량, Sales는 익일 할인율.

따라서 이번 주에 실제로 결정하는 것은 **차주 생산계획(요일별 Job 수 + 시퀀스)**과
**매일의 발주·할인**이다. 이번 주 생산계획은 LOCKED_PRODUCTION_PLAN으로 고정해 두고,
그 전제 위에서 나머지를 계산한다.

생산가능일은 제품마다 다르다: P1은 평일만, P2는 토·일 포함 7일.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from iog_sim.calendar import GameCalendar
from iog_sim.config import SimConfig
from iog_sim.demand.generator import extend_with_forecast
from iog_sim.examples.demo_run import POLICIES, build
from iog_sim.experiment import daily_input_report, weekly_plan_report

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

TODAY = dt.date(2026, 9, 22)

# 시스템 "Material inventory in/out summary" 실측값 (Current level)
CURRENT_MAT_INVENTORY = {"M1": 197_000, "M2": 814_000, "M3": 610_000}
# 완제품 재고 실측값 (Current level)
CURRENT_FG_INVENTORY = {"P1": 103_000, "P2": 5_830}
# Ledger 화면의 현재 Balance
CURRENT_BALANCE = -63_708_910

# 지난 결정일에 제출되어 확정된 이번 주 생산계획 (Production 탭에서 읽어 채운다).
# Job 1개 = 1 Lot = 상품 1,000개. 이 값은 바꿀 수 없으므로 계산의 전제로만 쓴다.
LOCKED_PRODUCTION_PLAN = {
    dt.date(2026, 9, 22): {"P1": 82, "P2": 0},
    dt.date(2026, 9, 23): {"P1": 0, "P2": 0},
    dt.date(2026, 9, 24): {"P1": 0, "P2": 40},
    dt.date(2026, 9, 25): {"P1": 86, "P2": 0},
    dt.date(2026, 9, 26): {"P1": 0, "P2": 0},
}

POLICY = next(p for p in POLICIES if p.label == "DP+한계이익할인")


def main() -> None:
    cal, cfg = GameCalendar(), SimConfig()
    deadline = cal.next_decision_day(TODAY)     # 이번 주 제출 마감 (토)
    plan_week = cal.next_week(deadline)         # 그 제출의 대상 = 차주 일~토
    sim_end = plan_week[-1]

    sim = build(POLICY)

    # 수요 CSV는 어제(=TODAY-1)까지만 있다. 그대로 두면 리포트 구간의 실현 수요가 전부
    # 0이 되어 매출·품절이 통째로 빠지므로, 계획에 쓰는 것과 같은 SES 예측치로 채운다.
    window = cal.date_range(TODAY, sim_end)
    extend_with_forecast(
        sim.demand,
        POLICY.forecaster,
        {code: cal.demand_days(window, spec.market_days_only)
         for code, spec in cfg.products.items()},
        window,
        origin=TODAY - dt.timedelta(days=1),
    )

    # 마감일(토)에 엔진의 _plan_next_week가 자동으로 돌아 차주 계획을 세운다.
    res = sim.run(
        start=TODAY,
        end=sim_end,
        seed=0,
        warmup_days=0,
        initial_fg_inventory=CURRENT_FG_INVENTORY,
        initial_mat_inventory=CURRENT_MAT_INVENTORY,
        backfill_plan_from=TODAY - dt.timedelta(days=1),
        locked_production_plan=LOCKED_PRODUCTION_PLAN,
    )
    seq_fn = lambda code, d, lots: sim.sequence_for(code, d, lots).job_ids()  # noqa: E731

    # --- 이번 주: 이미 확정 (참고용) -----------------------------------------
    this_week = [d for d in sorted(LOCKED_PRODUCTION_PLAN) if d >= TODAY]
    locked = weekly_plan_report(res, this_week, sequence_fn=seq_fn)
    print("[이번 주 - 확정됨] 지난 결정일에 제출 완료, 변경 불가 (Job 1개 = 1,000개)")
    print(locked.drop(columns=["P1_sequence", "P2_sequence"]).to_string(index=False))
    print()

    # --- 차주: 이번 주 제출 대상 (핵심 산출물) --------------------------------
    upcoming = weekly_plan_report(res, plan_week, sequence_fn=seq_fn)
    print(f"[차주 제출용] Demand + Production 탭 / 마감 {deadline:%Y-%m-%d}(토) 자정")
    print(f"             대상 {plan_week[0]:%m-%d}~{plan_week[-1]:%m-%d}"
          f" / P1은 평일만, P2는 토·일 포함 매일 생산 가능")
    print(upcoming.drop(columns=["P1_sequence", "P2_sequence"]).to_string(index=False))
    print()

    # --- 일일 입력 -----------------------------------------------------------
    daily = daily_input_report(res, baseline_balance=CURRENT_BALANCE)
    print("[일일 입력] Material/Sales, 매일 20시 마감"
          " (Material은 그날 발주량, Sales는 discount_target일 적용 할인율)")
    print(daily.to_string(index=False))
    print()

    print("[차주 Job 시퀀스] Production 탭 'Job sequence' 칸에 입력")
    for _, row in upcoming.iterrows():
        for code in ("P1", "P2"):
            seq = row[f"{code}_sequence"]
            if seq:
                print(f"  {row['date']} {code} ({len(seq)} Job): {','.join(map(str, seq))}")

    _save(upcoming, daily)


def _save(upcoming, daily) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_date = dt.date.today()          # 실행한 날짜(실제 달력 기준) = 파일명
    upcoming = upcoming.copy()
    for code in ("P1", "P2"):
        # Production 화면은 쉼표 구분 Job ID 열을 요구한다 (예: 8,4,5,6,1,3,2,7,10,9)
        upcoming[f"{code}_sequence"] = upcoming[f"{code}_sequence"].apply(
            lambda seq: ",".join(map(str, seq)))

    for name, frame in (("next_week_plan", upcoming), ("daily_input", daily)):
        path = OUTPUT_DIR / f"{name}_{run_date:%Y%m%d}.csv"
        try:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        except PermissionError:
            # 엑셀 등에서 파일을 열어 둔 채 재실행하면 잠겨 있다. 결과를 버리지 않는다.
            path = path.with_name(f"{path.stem}_{dt.datetime.now():%H%M%S}.csv")
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV 저장: {path}")


if __name__ == "__main__":
    main()
