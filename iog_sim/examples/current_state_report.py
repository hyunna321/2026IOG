"""라운드 운영 리포트 - 매일 시스템 화면 값을 아래 '입력' 블록에 옮겨 적고 실행한다.

    python -m iog_sim.examples.current_state_report

입력 채널은 결정 시점이 다르므로 표도 나눈다.

  [주간] Demand / Production : 차주 7일치를 토요일 자정에 한 번에 제출. 이번 주 계획은 이미 마감.
  [일일] Material / Sales    : 매일 20시. Material은 당일 발주량, Sales는 익일 할인율.

입력 값은 모두 **어제 20시 틱 이후** 화면 값이다. 시뮬레이션은 TODAY 하루(오늘 20시 틱)부터 돈다.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from iog_sim.calendar import GameCalendar
from iog_sim.config import SimConfig
from iog_sim.demand.generator import extend_with_forecast
from iog_sim.examples.demo_run import POLICIES, build, make_demand
from iog_sim.experiment import (daily_input_report, material_flow_report, sales_flow_report,
                                weekly_plan_report)
from iog_sim.state import PurchaseOrder

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
# 전날 대조용 기록. 매일 이 파일과 함께 커밋해야 다음 담당자가 git pull로 받아 대조한다.
SNAPSHOT_PATH = Path(__file__).resolve().parent / "daily_snapshot.json"

# ############################################################################
# ▼▼▼ 입력 ① 매일 갱신 ▼▼▼  (어제 20시 틱 이후 화면 값)
# ############################################################################

TODAY = dt.date(2026, 10, 16)

# [Material 화면] Material inventory in/out summary
#   자재: (Total in, Total out, Current level).  Current level이 시뮬레이션 시작 재고가 된다.
MATERIAL_INVENTORY_INOUT = {
    "M1": (3_014_479, 2_940_000, 74_479),
    "M2": (9_626_000, 8_906_000, 720_000),
    "M3": (1_576_760, 1_513_000, 63_760),
}

# [Material 화면] Material plans
#   (자재, 옵션 "normal"|"urgent", 수량, 발주일, 입고일)  -- 아직 안 들어온 행은 입고일 None.
#   입고일이 None인 행만 입고예정으로 쓴다. 빠뜨리면 같은 물량을 한 번 더 발주한다.
#   도착 예정일은 발주일 + 리드타임(config)으로 계산한다.
MATERIAL_PLANS = [
    ("M1", "normal", 73_521, dt.date(2026, 10, 13), None),
    ("M3", "normal", 22_240, dt.date(2026, 10, 11), None),
]

# [Sales 화면] Sales inventory in/out summary
#   제품: (Total in, Total out, Current level).  Current level이 시뮬레이션 시작 재고가 된다.
#   모르는 칸은 None (검산만 건너뛴다).
SALES_INVENTORY_INOUT = {
    "P1": (2_940_000, 2_805_500, 134_500),
    "P2": (1_513_000, 1_510_787, 2_213),         # VERIFY: Total in/out 화면 값으로 채울 것
}

# [Ledger 화면] Balance. expected_balance 열의 기준선일 뿐 발주·할인·계획에는 영향이 없다.
CURRENT_BALANCE = 72_131_718

# ############################################################################
# ▼▼▼ 입력 ② 토요일마다 갱신 ▼▼▼  (제출해서 확정된 이번 주 Production 계획)
# ############################################################################
# {날짜: {"P1": Job수, "P2": Job수}}. 자재 부족으로 잘린 날은 실제 생산량을 적는다.
# 09-27~10-03은 시스템 AUTO 주(09-29 틱 건너뜀, 10-01 이틀치 정산, 10-02 M2 소진으로 잘림).
LOCKED_PRODUCTION_PLAN = {
    dt.date(2026, 10, 11): {"P1": 0,   "P2": 66},
    dt.date(2026, 10, 12): {"P1": 226,   "P2": 83},
    dt.date(2026, 10, 13): {"P1": 265, "P2": 83},
    dt.date(2026, 10, 14): {"P1": 265, "P2": 83},
    dt.date(2026, 10, 15): {"P1": 265, "P2": 83},
    dt.date(2026, 10, 16): {"P1": 274,   "P2": 86},
}

# ############################################################################
# ▼▼▼ 입력 ③ 선택 ▼▼▼  수요 파일보다 먼저 알게 된 실측 수요 {제품: {날짜: 수량}}
# ############################################################################
OBSERVED_DEMAND: Dict[str, Dict[dt.date, int]] = {
    "P1": {
        dt.date(2026, 10, 6): 273_000,
        dt.date(2026, 10, 7): 270_000,
        dt.date(2026, 10, 8): 263_500,
        dt.date(2026, 10, 9): 0,
        dt.date(2026, 10, 10): 0,
        dt.date(2026, 10, 11): 0,
        dt.date(2026, 10, 12): 251000,
        dt.date(2026, 10, 13): 234_000,
        dt.date(2026, 10, 14): 233_000,
        dt.date(2026, 10, 15): 209_000,
        dt.date(2026, 10, 16): 216_500,
    },
    "P2": {
        dt.date(2026, 10, 3): 84_575,
        dt.date(2026, 10, 4): 85_005,
        dt.date(2026, 10, 5): 86_266,
        dt.date(2026, 10, 6): 85_311,
        dt.date(2026, 10, 7): 84_149,
        dt.date(2026, 10, 8): 82_938,
        dt.date(2026, 10, 9): 82_455,
        dt.date(2026, 10, 10): 82_654,
        dt.date(2026, 10, 11): 83_079,
        dt.date(2026, 10, 12): 84131,
        dt.date(2026, 10, 13): 83_627,
        dt.date(2026, 10, 14): 83_505,
        dt.date(2026, 10, 15): 80_787,
        dt.date(2026, 10, 16): 82_090,
    },
}

# ############################################################################
# ▲▲▲ 입력 끝 ▲▲▲
# ############################################################################

# 화면 값끼리 BOM 검산이 어긋나면 실행을 멈춘다. 시스템 화면 자체가 어긋난 것을 확인한 날만 True.
SKIP_BOM_CHECK = False

# '확인 후 진행' 경고는 터미널에서 y를 입력해야 계속된다. 자동 실행(테스트·실험 스크립트)에서만 True.
AUTO_CONFIRM = False

# 하루 수요가 직전 값보다 이 비율 넘게 변하면 OBSERVED_DEMAND 오타를 의심한다.
DEMAND_JUMP = 0.15

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


def _week_of(d: dt.date) -> List[dt.date]:
    """d가 속한 주(일~토)."""
    sunday = d - dt.timedelta(days=(d.weekday() + 1) % 7)
    return [sunday + dt.timedelta(days=i) for i in range(7)]


def check_hard(today: dt.date, history: Dict[str, dict],
               round_end: Optional[dt.date] = None) -> List[str]:
    """틀리면 반드시 입력 실수인 것들 -> 리포트를 멈춘다 (BOM 검산은 check_bom).

    - in - out != Current level : 시스템 화면에서는 항상 맞으므로 옮겨 적기 실수
    - 이번 주(일~토, 라운드 종료일까지) 날짜가 LOCKED_PRODUCTION_PLAN에 없음 : 지난주 계획이 남았거나
      차주 계획으로 이미 바꿈
    - TODAY가 지난 실행 기록보다 과거 : TODAY 오타
    """
    errors = []
    for table, name in ((MATERIAL_INVENTORY_INOUT, "Material"), (SALES_INVENTORY_INOUT, "Sales")):
        for code, (total_in, total_out, level) in table.items():
            if total_in is not None and total_out is not None and total_in - total_out != level:
                errors.append(f"{name} {code}: in {total_in:,} - out {total_out:,}"
                              f" != Current level {level:,} -> 화면 값을 다시 옮겨 적으세요")
    missing = [d for d in _week_of(today)
               if (round_end is None or d <= round_end) and d not in LOCKED_PRODUCTION_PLAN]
    if missing:
        errors.append(f"LOCKED_PRODUCTION_PLAN에 이번 주 날짜가 없음: "
                      f"{', '.join(f'{d:%m-%d}' for d in missing)} -> 이번 주(일~토)에 제출된 계획을 넣으세요")
    if history:
        latest = max(dt.date.fromisoformat(k) for k in history)
        if today < latest:
            errors.append(f"TODAY({today})가 지난 실행 기록({latest})보다 과거 -> TODAY를 확인하세요")
    return errors


def check_inputs(cfg: SimConfig, today: dt.date) -> List[str]:
    """입고 예정일이 지났는데 미입고로 남은 발주. 실제 입고 지연일 수도 있어 '확인 후 진행'."""
    warnings = []
    for mat, option, qty, order_date, delivered in MATERIAL_PLANS:
        if delivered is None:
            arrival = order_date + dt.timedelta(
                days=cfg.materials[mat].option(option).lead_time_days)
            if arrival < today:
                warnings.append(f"Material plans {mat} {option} {qty:,} ({order_date}):"
                                f" 예정일 {arrival}이 지났는데 미입고 -> 오늘 도착으로 가정")
    return warnings


def check_bom(cfg: SimConfig) -> List[str]:
    """자재 Total out = BOM x 제품 Total in (Sales in은 생산 입고량).

    어긋나면 한쪽 표를 다른 날 화면에서 옮겨 적은 것이다. 그대로 돌리면 자재 재고가 틀려
    발주량이 틀린다 (10/13: M1만 전날 값 -> 126,000개 과대 -> 10/16 P1 126 Job 부족).
    """
    errors = []
    sales_in = {p: v[0] for p, v in SALES_INVENTORY_INOUT.items()}
    for mat, (_, mat_out, _) in MATERIAL_INVENTORY_INOUT.items():
        users = {p: bom[mat] for p, bom in cfg.bom.items() if mat in bom}
        if mat_out is None or any(sales_in.get(p) is None for p in users):
            continue
        expected = sum(sales_in[p] * k for p, k in users.items())
        if expected != mat_out:
            errors.append(f"BOM 검산 {mat}: out {mat_out:,} != BOM x 생산 {expected:,}")
    return errors


def _screen() -> dict:
    return {"material": {m: list(v) for m, v in MATERIAL_INVENTORY_INOUT.items()},
            "sales": {p: list(v) for p, v in SALES_INVENTORY_INOUT.items()}}


def check_against_snapshot(history: Dict[str, dict], today: dt.date) -> List[str]:
    """어제 리포트가 예상한 '오늘 화면'과 실제 오늘 화면을 대조한다 (경고만, 실행은 계속).

    판매(Total out)는 수요가 예측과 달라 원래 어긋나므로 보지 않는다. 시스템이 정상이면
    정확히 맞아야 하는 값만 본다: 자재 Total in(입고), 자재 Total out(소모), 제품 Total in(생산).
    정산이 빠진 날(09-29), 생산·입고가 덜 반영된 날, 화면을 다른 날 값으로 옮겨 적은 날,
    TODAY를 안 바꾼 날을 잡는다. 어제 기록이 없거나 어제 입력이 BOM 검산을 통과하지 못했으면
    대조하지 않는다 (틀린 기록과 비교하면 오늘 정상 입력이 경고로 뜬다).
    """
    prev = history.get((today - dt.timedelta(days=1)).isoformat())
    if not prev or not prev.get("bom_ok", True):
        return []
    actual = {
        "material_in": {m: v[0] for m, v in MATERIAL_INVENTORY_INOUT.items()},
        "material_out": {m: v[1] for m, v in MATERIAL_INVENTORY_INOUT.items()},
        "sales_in": {p: v[0] for p, v in SALES_INVENTORY_INOUT.items()},
    }
    label = {"material_in": "자재 Total in(입고)", "material_out": "자재 Total out(소모)",
             "sales_in": "제품 Total in(생산)"}
    hint = {"material_in": "입고예정분이 안 들어왔거나 MATERIAL_PLANS가 틀림",
            "material_out": "어제 생산이 계획과 다르게 반영됨",
            "sales_in": "어제 생산이 계획과 다르게 반영됨 (정산 누락? 자재 부족?)"}
    warnings = []
    for key, by_code in prev["expected"].items():
        for code, exp in by_code.items():
            got = actual[key].get(code)
            if exp is None or got is None or got == exp:
                continue
            warnings.append(f"[전날 대조] {code} {label[key]}: 화면 {got:,} / 어제 리포트 예상 {exp:,}"
                            f" (차이 {got - exp:+,}) -> {hint[key]}")
    if warnings:
        warnings.append("[전날 대조] TODAY가 오늘인지, 화면 값이 어제 20시 정산 이후 값인지 확인하세요. "
                        "실제 생산량이 계획과 달랐으면 LOCKED_PRODUCTION_PLAN의 그날 값을 실제 Job 수로 고치세요")
    return warnings


def check_demand_jumps(series: Dict[str, Dict[dt.date, int]], today: dt.date) -> List[str]:
    """최근 7일 OBSERVED_DEMAND가 직전 수요보다 DEMAND_JUMP 넘게 변했으면 오타 의심 (실제 급변일 수도 있음)."""
    warnings = []
    for code, points in OBSERVED_DEMAND.items():
        full = series.get(code, {})
        for d, v in sorted(points.items()):
            if not v or not (today - dt.timedelta(days=7) <= d < today):
                continue
            prev = [x for k, x in sorted(full.items()) if k < d and x > 0]
            if prev and abs(v / prev[-1] - 1) > DEMAND_JUMP:
                warnings.append(f"OBSERVED_DEMAND {code} {d:%m-%d} = {v:,} (직전 {prev[-1]:,}, "
                                f"{v / prev[-1] - 1:+.0%}) -> 오타가 아닌지 Demand 화면과 대조하세요")
    return warnings


def check_missing_orders(history: Dict[str, dict], today: dt.date) -> List[str]:
    """어제 리포트가 권고한 발주가 MATERIAL_PLANS에 없으면 경고 (기록을 빼먹으면 같은 물량을 또 발주한다)."""
    yesterday = today - dt.timedelta(days=1)
    prev = history.get(yesterday.isoformat())
    if not prev:
        return []
    recorded = {(m, o) for m, o, _, d, _ in MATERIAL_PLANS if d == yesterday}
    return [f"어제({yesterday:%m-%d}) 권고 발주 {m} {o} {q:,}가 MATERIAL_PLANS에 없음 -> "
            f"실제로 발주했으면 행을 추가, 일부러 안 했으면 y로 진행"
            for m, o, q in prev.get("orders", []) if (m, o) not in recorded]


def _confirm(lines: List[str]) -> None:
    """'확인 후 진행' 경고: 터미널에서 y를 입력해야 계속한다. 그 외 입력·자동 실행은 중단."""
    if not lines:
        return
    _print_warnings(lines, f"확인 필요 {len(lines)}건 - 읽고 y를 입력해야 계속합니다")
    if AUTO_CONFIRM:
        print("  (AUTO_CONFIRM = True: 자동으로 진행)")
        return
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("[중단] 확인이 필요한 경고가 있습니다. 터미널에서 직접 실행해 y로 확인하세요.")
    answer = input("  위 경고를 모두 확인했고 그대로 진행하려면 y 입력: ").strip().lower()
    if answer != "y":
        raise SystemExit("[중단] 경고를 확인한 뒤 입력값을 고치고 다시 실행하세요.")


def make_snapshot(res, today: dt.date, bom_ok: bool, acknowledged: Optional[List[str]] = None) -> dict:
    """오늘 입력 화면 + 오늘 20시 정산 후 예상 누계 + 오늘 권고 발주 (내일 리포트가 대조한다)."""
    received = {r.material: r.received for r in res.state.material_records if r.date == today}
    consumed = {r.material: r.consumed for r in res.state.material_records if r.date == today}
    produced = {r.product: r.produced for r in res.state.daily_records if r.date == today}

    def plus(total, add):
        return None if total is None else int(total + add)

    return {
        "bom_ok": bom_ok,
        "acknowledged": list(acknowledged or []),
        "orders": [[o.material, o.option, int(o.qty)] for o in res.state.open_orders
                   if o.order_date == today],
        "screen": _screen(),
        "expected": {
            "material_in": {m: plus(v[0], received.get(m, 0)) for m, v in MATERIAL_INVENTORY_INOUT.items()},
            "material_out": {m: plus(v[1], consumed.get(m, 0)) for m, v in MATERIAL_INVENTORY_INOUT.items()},
            "sales_in": {p: plus(v[0], produced.get(p, 0)) for p, v in SALES_INVENTORY_INOUT.items()},
        },
    }


def _load_snapshots() -> Dict[str, dict]:
    try:
        data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_snapshot(history: Dict[str, dict], today: dt.date, snapshot: dict) -> None:
    """날짜별로 남긴다 (같은 날 다시 돌리면 그날 기록만 바뀐다). 최근 14일만 보관."""
    history = dict(history)
    history[today.isoformat()] = snapshot
    keep = sorted(history)[-14:]
    SNAPSHOT_PATH.write_text(json.dumps({k: history[k] for k in keep}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"전날 대조 기록 저장: {SNAPSHOT_PATH.name} (current_state_report.py와 함께 커밋)")


def _print_warnings(lines: List[str], title: str) -> None:
    """터미널에서 놓치지 않도록 느낌표 테두리로 감싼다."""
    if not lines:
        return
    bar = "!" * 78
    print()
    print(bar)
    print(f"!!!  {title}")
    print(bar)
    for line in lines:
        print(f"!!!  {line}")
    print(bar)
    print()


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
    bom_errors = check_bom(cfg)
    if bom_errors and not SKIP_BOM_CHECK:
        _print_warnings(bom_errors, "[중단] BOM 검산 실패 - 리포트를 멈춥니다")
        raise SystemExit(
            "\n[중단] 자재 Total out과 제품 Total in이 BOM으로 맞지 않습니다.\n"
            "  Material·Sales 화면 값을 같은 날(어제 20시 정산 이후) 화면에서 다시 옮겨 적으세요.\n"
            "  화면 값 자체가 어긋난 것을 확인했을 때만 SKIP_BOM_CHECK = True로 두고 실행하세요.")
    history = _load_snapshots()
    hard = check_hard(TODAY, history, cal.round_end_for(TODAY))
    if hard:
        _print_warnings(hard, "[중단] 입력 오류 - 고치기 전에는 실행할 수 없습니다")
        raise SystemExit("\n[중단] 위 입력 오류를 고친 뒤 다시 실행하세요.")
    yesterday = history.get((TODAY - dt.timedelta(days=1)).isoformat())
    if yesterday is None:
        print("  (어제 대조 기록 없음 -> 전날 대조·발주 기록 확인 생략)")
    elif yesterday.get("acknowledged"):
        print("  (참고) 어제 담당자가 확인 후 진행한 경고:")
        for w in yesterday["acknowledged"]:
            print(f"    - {w}")
    full_demand = make_demand().series
    for code, points in OBSERVED_DEMAND.items():
        full_demand[code].update(points)
    warnings = (bom_errors + check_inputs(cfg, TODAY) + check_against_snapshot(history, TODAY)
                + check_missing_orders(history, TODAY) + check_demand_jumps(full_demand, TODAY))
    _confirm(warnings)

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
    _write_snapshot(history, TODAY, make_snapshot(res, TODAY, bom_ok=not bom_errors,
                                                  acknowledged=warnings))
    if warnings:
        # 출력이 길어 맨 위 경고가 화면 밖으로 밀려나므로 끝에서 한 번 더 보여 준다
        _print_warnings(warnings, f"다시 알림: 확인 후 진행한 경고 {len(warnings)}건 (단톡방에 공유)")


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
