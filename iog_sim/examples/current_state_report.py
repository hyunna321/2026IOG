"""라운드 운영 리포트 - 매일 시스템 화면 값을 아래 '입력' 블록에 옮겨 적고 실행한다.

    python -m iog_sim.examples.current_state_report

입력 채널은 결정 시점이 다르므로 표도 나눈다.

  [주간] Demand / Production : 차주 7일치를 토요일 자정에 한 번에 제출. 이번 주 계획은 이미 마감.
  [일일] Material / Sales    : 매일 20시. Material은 당일 발주량, Sales는 익일 할인율.

입력 값은 모두 **어제 20시 틱 이후** 화면 값이다. 시뮬레이션은 TODAY 하루(오늘 20시 틱)부터 돈다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd

from iog_sim.calendar import GameCalendar
from iog_sim.config import SimConfig
from iog_sim.demand.generator import extend_with_forecast
from iog_sim.examples.demo_run import POLICIES, build
from iog_sim.experiment import (daily_input_report, material_flow_report, sales_flow_report,
                                weekly_plan_report)
from iog_sim.state import PurchaseOrder

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

# ############################################################################
# ▼▼▼ 입력 ① 매일 갱신 ▼▼▼  (어제 20시 틱 이후 화면 값)
# ############################################################################

TODAY = dt.date(2026, 10, 8)

# [Material 화면] Material inventory in/out summary
#   자재: (Total in, Total out, Current level).  Current level이 시뮬레이션 시작 재고가 된다.
MATERIAL_INVENTORY_INOUT = {
    "M1": (2_059_400, 1_625_000,   434_400),
    "M2": (9_541_377, 4_958_000,   4_583_377),
    "M3": (1_000_000,   854_000,   146_000),
}

# [Material 화면] Material plans
#   (자재, 옵션 "normal"|"urgent", 수량, 발주일, 입고일)  -- 아직 안 들어온 행은 입고일 None.
#   입고일이 None인 행만 입고예정으로 쓴다. 빠뜨리면 같은 물량을 한 번 더 발주한다.
#   도착 예정일은 발주일 + 리드타임(config)으로 계산한다.
MATERIAL_PLANS = [
    ("M3", "normal", 576_760, dt.date(2026, 10, 4), None),
]

# [Sales 화면] Sales inventory in/out summary
#   제품: (Total in, Total out, Current level).  Current level이 시뮬레이션 시작 재고가 된다.
#   모르는 칸은 None (검산만 건너뛴다).
SALES_INVENTORY_INOUT = {
    "P1": (1_625_000, 1_616_500, 8_500),
    "P2": (854_000, 850_306, 3_694),            # VERIFY: Total in/out 화면 값으로 채울 것
}

# [Ledger 화면] Balance. expected_balance 열의 기준선일 뿐 발주·할인·계획에는 영향이 없다.
CURRENT_BALANCE = -90_086_136

# ############################################################################
# ▼▼▼ 입력 ② 토요일마다 갱신 ▼▼▼  (제출해서 확정된 이번 주 Production 계획)
# ############################################################################
# {날짜: {"P1": Job수, "P2": Job수}}. 자재 부족으로 잘린 날은 실제 생산량을 적는다.
# 09-27~10-03은 시스템 AUTO 주(09-29 틱 건너뜀, 10-01 이틀치 정산, 10-02 M2 소진으로 잘림).
LOCKED_PRODUCTION_PLAN = {
    dt.date(2026, 10, 4): {"P1": 0,   "P2": 86},
    dt.date(2026, 10, 5): {"P1": 0,   "P2": 86},
    dt.date(2026, 10, 6): {"P1": 275, "P2": 86},
    dt.date(2026, 10, 7): {"P1": 275, "P2": 86},
    dt.date(2026, 10, 8): {"P1": 294, "P2": 86},
    dt.date(2026, 10, 9): {"P1": 0,   "P2": 175},
    dt.date(2026, 10, 10): {"P1": 0,  "P2": 0}, 
}

# ############################################################################
# ▼▼▼ 입력 ③ 선택 ▼▼▼  수요 파일보다 먼저 알게 된 실측 수요 {제품: {날짜: 수량}}
# ############################################################################
OBSERVED_DEMAND: Dict[str, Dict[dt.date, int]] = {
    "P1": {
        dt.date(2026, 10, 6): 273_000,
        dt.date(2026, 10, 7): 270_000,
    },
    "P2": {
        dt.date(2026, 10, 3): 84_575,
        dt.date(2026, 10, 4): 85_005,
        dt.date(2026, 10, 5): 86_266,
        dt.date(2026, 10, 6): 85_311,
        dt.date(2026, 10, 7): 84_149,
    },
}

# ############################################################################
# ▲▲▲ 입력 끝 ▲▲▲
# ############################################################################

POLICY = next(p for p in POLICIES if p.label == "DP+한계이익할인")


def in_transit_orders(cfg: SimConfig, today: dt.date) -> List[PurchaseOrder]:
    """Material plans에서 아직 입고되지 않은 행 -> 입고예정 주문.

    예정일이 이미 지났는데 입고되지 않은 행(틱 건너뜀 등)은 오늘 도착으로 본다.
    """
    orders = []
    for mat, option, qty, order_date, delivered in MATERIAL_PLANS:
        if delivered is not None:
            continue
        arrival = order_date + dt.timedelta(days=cfg.materials[mat].option(option).lead_time_days)
        orders.append(PurchaseOrder(mat, option, qty, order_date, max(arrival, today)))
    return orders


def check_inputs(cfg: SimConfig, today: dt.date) -> List[str]:
    """화면 값끼리 맞물리는지 검산. 어긋나면 옮겨 적기 실수이거나 시스템 이상이다."""
    warnings = []
    for table, name in ((MATERIAL_INVENTORY_INOUT, "Material"), (SALES_INVENTORY_INOUT, "Sales")):
        for code, (total_in, total_out, level) in table.items():
            if total_in is not None and total_out is not None and total_in - total_out != level:
                warnings.append(f"{name} {code}: in {total_in:,} - out {total_out:,}"
                                f" != Current level {level:,}")

    # 자재 Total out = BOM x 제품 Total in (Sales in은 생산 입고량)
    sales_in = {p: v[0] for p, v in SALES_INVENTORY_INOUT.items()}
    for mat, (_, mat_out, _) in MATERIAL_INVENTORY_INOUT.items():
        users = {p: bom[mat] for p, bom in cfg.bom.items() if mat in bom}
        if mat_out is None or any(sales_in.get(p) is None for p in users):
            continue
        expected = sum(sales_in[p] * k for p, k in users.items())
        if expected != mat_out:
            warnings.append(f"BOM 검산 {mat}: out {mat_out:,} != BOM x 생산 {expected:,}")

    for mat, option, qty, order_date, delivered in MATERIAL_PLANS:
        if delivered is None:
            arrival = order_date + dt.timedelta(
                days=cfg.materials[mat].option(option).lead_time_days)
            if arrival < today:
                warnings.append(f"Material plans {mat} {option} {qty:,} ({order_date}):"
                                f" 예정일 {arrival}이 지났는데 미입고 -> 오늘 도착으로 가정")
    return warnings


def main() -> None:
    cal, cfg = GameCalendar(), SimConfig()
    mat_inventory = {m: v[2] for m, v in MATERIAL_INVENTORY_INOUT.items()}
    fg_inventory = {p: v[2] for p, v in SALES_INVENTORY_INOUT.items()}
    open_orders = in_transit_orders(cfg, TODAY)

    print(f"[입력 확인] 기준일 {TODAY} ({TODAY:%a})")
    print("  자재 재고 :", ", ".join(f"{m} {q:,}" for m, q in mat_inventory.items()))
    print("  제품 재고 :", ", ".join(f"{p} {q:,}" for p, q in fg_inventory.items()))
    for o in open_orders:
        print(f"  입고예정  : {o.material} {o.option:6s} {o.qty:>11,}  발주 {o.order_date}"
              f" -> 도착 {o.arrival_date}")
    for w in check_inputs(cfg, TODAY):
        print(f"  ! {w}")

    deadline = cal.next_decision_day(TODAY)      # 이번 주 제출 마감 (토)
    plan_week = cal.next_week(deadline)          # 그 제출의 대상 = 차주 일~토
    round_end = cal.round_end_for(TODAY)
    sim_end = min(plan_week[-1], round_end) if round_end else plan_week[-1]
    if round_end:
        print(f"  라운드 종료: {round_end} (이후 생산·발주 없음, 종료 주는 안전재고 없이 계획)")
    print()

    sim = build(POLICY)
    for code, points in OBSERVED_DEMAND.items():
        sim.demand.series[code].update(points)
    # 수요 파일은 어제까지만 있으므로, 리포트 구간의 실현 수요를 같은 SES 예측으로 채운다.
    window = cal.date_range(TODAY, sim_end)
    extend_with_forecast(
        sim.demand,
        POLICY.forecaster,
        {code: cal.demand_days(window, spec.market_days_only)
         for code, spec in cfg.products.items()},
        window,
        origin=TODAY - dt.timedelta(days=1),
    )

    res = sim.run(
        start=TODAY,
        end=sim_end,
        initial_fg_inventory=fg_inventory,
        initial_mat_inventory=mat_inventory,
        backfill_current_week=True,
        locked_production_plan=LOCKED_PRODUCTION_PLAN,
        initial_open_orders=open_orders,
    )
    seq_fn = lambda code, d, lots: sim.sequence_for(code, d, lots).job_ids()  # noqa: E731
    seq_cols = ["P1_sequence", "P2_sequence"]

    this_week = cal.date_range(TODAY, min(deadline, sim_end))
    rest = weekly_plan_report(res, this_week, sequence_fn=seq_fn)
    print(f"[이번 주 남은 날 - 확정 계획] {this_week[0]:%m-%d}~{this_week[-1]:%m-%d}"
          f"  (입력 대상 아님. *_short > 0 이면 그날 자재가 모자라 계획이 잘린다)")
    print(rest.drop(columns=seq_cols).to_string(index=False))
    print()

    upcoming_days = [d for d in plan_week if d <= sim_end]
    upcoming = weekly_plan_report(res, upcoming_days, sequence_fn=seq_fn)
    if upcoming_days:
        print(f"[차주 제출용] Demand + Production 탭 / 마감 {deadline:%Y-%m-%d}(토) 자정"
              f" / 대상 {upcoming_days[0]:%m-%d}~{upcoming_days[-1]:%m-%d}")
        print(upcoming.drop(columns=seq_cols).to_string(index=False))
        print()

    daily = daily_input_report(res, baseline_balance=CURRENT_BALANCE)
    print("[일일 입력] Material/Sales, 매일 20시 마감"
          " (Material은 그날 발주량, Sales는 discount_target일 적용 할인율)")
    print(daily.to_string(index=False))
    print()

    mat_flow = material_flow_report(res)
    print("[예상 Material inventory in/out] 다음 날 화면과 대조")
    print(mat_flow.to_string(index=False))
    print()

    sales_flow = sales_flow_report(res)
    print("[예상 Sales inventory in/out] 다음 날 화면과 대조")
    print(sales_flow.to_string(index=False))
    print()

    print("[차주 Job 시퀀스] Production 탭 'Job sequence' 칸에 입력")
    for _, row in upcoming.iterrows():
        for code in ("P1", "P2"):
            seq = row[f"{code}_sequence"]
            if seq:
                print(f"  {row['date']} {code} ({len(seq)} Job): {','.join(map(str, seq))}")

    _save({"this_week_rest": _flatten(rest), "next_week_plan": _flatten(upcoming),
           "daily_input": daily, "material_flow": mat_flow, "sales_flow": sales_flow})


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    """Production 화면은 쉼표 구분 Job ID 열을 요구한다 (예: 8,4,5,6,1,3,2,7,10,9)."""
    frame = frame.copy()
    for code in ("P1", "P2"):
        frame[f"{code}_sequence"] = frame[f"{code}_sequence"].apply(
            lambda seq: ",".join(map(str, seq)))
    return frame


def _save(frames: Dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = OUTPUT_DIR / f"{name}_{TODAY:%Y%m%d}.csv"
        try:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        except PermissionError:
            # 엑셀에서 열어 둔 채 재실행하면 잠겨 있다. 결과를 버리지 않는다.
            path = path.with_name(f"{path.stem}_{dt.datetime.now():%H%M%S}.csv")
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"CSV 저장: {path}")


if __name__ == "__main__":
    main()
