"""라운드 전체 계획 - TODAY부터 라운드 종료일까지의 일별·주별 기준 계획(baseline).

    python -m iog_sim.examples.round_plan

입력은 `current_state_report.py`의 '입력' 블록을 그대로 읽는다(두 번 적지 않는다).
가장 최근 수요 파일로 SES 예측을 만들어 라운드 끝까지 실현 수요 자리에 깔고, 매일 쓰는 정책
그대로(토요일마다 차주 계획, 매일 발주·할인) 종료일까지 돌린다.

수요가 평평하다는 가정이므로 결과는 '규칙적인 간격·물량의 주문/생산' 패턴이 된다.
이 CSV를 기준선으로 남겨 두고, 이후 daily report의 실측값이 여기서 얼마나 벗어났는지 대조한다.

출력 (CSV는 `examples/reports/{이름}_{TODAY}.csv`)
  round_plan        일별: 예측수요·Job 수·자재 발주·할인율·예상 재고·예상 Balance
  round_plan_weekly 주별(일~토): 제출 마감일, 주간 합계, 주말 재고, 주말 Balance
"""

from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd

from iog_sim.calendar import GameCalendar
from iog_sim.config import SimConfig
from iog_sim.demand.generator import extend_with_forecast
from iog_sim.examples import current_state_report as inputs
from iog_sim.examples.current_state_report import POLICY, _save, check_inputs, in_transit_orders
from iog_sim.examples.demo_run import build
from iog_sim.experiment import (daily_input_report, material_flow_report, sales_flow_report,
                                weekly_plan_report)

ORDER_COLS = ["order_M1", "order_M2_normal", "order_M2_urgent", "order_M3"]


def week_start(d: dt.date) -> dt.date:
    """d가 속한 계획 주(일~토)의 일요일."""
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def build_daily(res, dates: List[dt.date], baseline_balance: float) -> pd.DataFrame:
    plan = weekly_plan_report(res, dates)            # 시퀀스는 제출 주에만 필요하다
    plan = plan.drop(columns=["P1_sequence", "P2_sequence"])
    daily = daily_input_report(res, baseline_balance=baseline_balance)
    daily = daily.rename(columns={"discount_P1": "next_discount_P1",
                                  "discount_P2": "next_discount_P2"}).drop(columns="discount_target")
    frame = (plan.merge(daily, on="date", how="left")
                 .merge(sales_flow_report(res), on="date", how="left")
                 .merge(material_flow_report(res), on="date", how="left"))
    frame.insert(2, "submit_by", [week_start(d) - dt.timedelta(days=1) for d in frame["date"]])
    return frame


def build_weekly(daily: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    rows = []
    for start, g in daily.groupby(daily["date"].map(week_start), sort=True):
        deadline = start - dt.timedelta(days=1)
        last = g.iloc[-1]
        rows.append({
            "week": f"{g['date'].iloc[0]:%m-%d}~{g['date'].iloc[-1]:%m-%d}",
            "submit_by": deadline,
            "status": "확정" if deadline < today else "제출 예정",
            "P1_jobs": int(g["P1_jobs"].sum()),
            "P1_prod_days": int((g["P1_jobs"] > 0).sum()),
            "P2_jobs": int(g["P2_jobs"].sum()),
            "P2_prod_days": int((g["P2_jobs"] > 0).sum()),
            "P1_demand": int(g["P1_out"].sum() + g["P1_lost"].sum()),
            "P2_demand": int(g["P2_out"].sum() + g["P2_lost"].sum()),
            "P1_lost": int(g["P1_lost"].sum()),
            "P2_lost": int(g["P2_lost"].sum()),
            **{c: int(g[c].sum()) for c in ORDER_COLS},
            "P1_level_end": int(last["P1_level"]),
            "P2_level_end": int(last["P2_level"]),
            "balance_end": int(last["expected_balance"]),
        })
    return pd.DataFrame(rows)


def order_cadence(daily: pd.DataFrame) -> List[str]:
    """자재별 발주일·수량과 직전 발주와의 간격(일). 규칙성 확인용."""
    lines = []
    for col in ORDER_COLS:
        hits = daily.loc[daily[col] > 0, ["date", col]]
        if hits.empty:
            lines.append(f"  {col:16s} 발주 없음")
            continue
        parts, prev = [], None
        for d, q in zip(hits["date"], hits[col]):
            gap = f" (+{(d - prev).days}d)" if prev else ""
            parts.append(f"{d:%m-%d} {int(q):,}{gap}")
            prev = d
        lines.append(f"  {col:16s} " + " | ".join(parts))
    return lines


def main() -> None:
    cal, cfg = GameCalendar(), SimConfig()
    today = inputs.TODAY
    round_end = cal.round_end_for(today)
    if round_end is None:
        raise SystemExit(f"{today}가 속한 라운드 종료일이 calendar.ROUND_ENDS에 없다.")

    mat_inventory = {m: v[2] for m, v in inputs.MATERIAL_INVENTORY_INOUT.items()}
    fg_inventory = {p: v[2] for p, v in inputs.SALES_INVENTORY_INOUT.items()}
    open_orders = in_transit_orders(cfg, today)

    print(f"[라운드 계획] 기준일 {today} ({today:%a}) ~ 종료일 {round_end} ({round_end:%a}),"
          f" {(round_end - today).days + 1}일")
    print("  자재 재고 :", ", ".join(f"{m} {q:,}" for m, q in mat_inventory.items()))
    print("  제품 재고 :", ", ".join(f"{p} {q:,}" for p, q in fg_inventory.items()))
    for w in check_inputs(cfg, today):
        print(f"  ! {w}")

    sim = build(POLICY)
    for code, points in inputs.OBSERVED_DEMAND.items():
        sim.demand.series[code].update(points)
    window = cal.date_range(today, round_end)
    origin = today - dt.timedelta(days=1)
    extend_with_forecast(
        sim.demand, POLICY.forecaster,
        {code: cal.demand_days(window, spec.market_days_only)
         for code, spec in cfg.products.items()},
        window, origin=origin,
    )
    level = {code: sim.demand.series[code][next(d for d in window
                                                 if sim.demand.series[code][d] > 0)]
             for code in cfg.products}
    print(f"  가정 수요 : SES 원점 {origin}, 개장일당 "
          + ", ".join(f"{c} {q:,}" for c, q in level.items())
          + "  (라운드 끝까지 평평 / P1은 휴장일 0)")
    print()

    res = sim.run(
        start=today, end=round_end,
        initial_fg_inventory=fg_inventory,
        initial_mat_inventory=mat_inventory,
        backfill_current_week=True,
        locked_production_plan=inputs.LOCKED_PRODUCTION_PLAN,
        initial_open_orders=open_orders,
    )

    daily = build_daily(res, window, inputs.CURRENT_BALANCE)
    weekly = build_weekly(daily, today)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("[주별 요약] 계획 주 = 일~토, submit_by(토) 자정까지 Demand/Production 제출")
        print(weekly.to_string(index=False))
        print()

        cols = ["date", "weekday", "submit_by", "forecast_P1", "P1_jobs", "forecast_P2", "P2_jobs",
                *ORDER_COLS, "next_discount_P1", "next_discount_P2",
                "P1_level", "P2_level", "P1_lost", "P2_lost",
                "M1_level", "M2_level", "M3_level", "expected_balance"]
        print("[일별 계획] next_discount_* = 그날 20시에 입력하는 익일 할인율")
        print(daily[cols].to_string(index=False))
        print()

    print("[자재 발주 리듬] 발주일 수량 (직전 발주 대비 간격)")
    print("\n".join(order_cadence(daily)))
    shorts = daily[(daily["P1_short"] > 0) | (daily["P2_short"] > 0)]
    if not shorts.empty:
        print()
        print("  ! 자재 부족으로 계획 Job이 잘리는 날:",
              ", ".join(f"{d:%m-%d}(P1 {a}, P2 {b})" for d, a, b in
                        zip(shorts["date"], shorts["P1_short"], shorts["P2_short"])))
    print()

    s = res.summary
    print(f"[라운드 합계] 예상 Balance {inputs.CURRENT_BALANCE + s.balance:,.0f}"
          f" (현재 {inputs.CURRENT_BALANCE:,} + 증분 {s.balance:,.0f})")
    print(f"  매출 {s.revenue:,.0f} / 생산비 {s.production_cost:,.0f}"
          f" / 재고비 {s.inventory_cost:,.0f} / 자재비 {s.material_cost:,.0f}")
    print()
    print("주의: 수요를 예측치로 고정한 낙관 시나리오(예측오차 0)다. 실측과의 차이는 이 CSV와 대조할 것.")

    _save({"round_plan": daily, "round_plan_weekly": weekly})


if __name__ == "__main__":
    main()
