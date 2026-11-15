"""시뮬레이션 최적화 - 정책 파라미터를 여러 라운드 시나리오에서 단계별로 비교·채택한다.

    python -m iog_sim.examples.optimize table          # 1회: makespan 표 생성 (24코어 약 3분)
    python -m iog_sim.examples.optimize <stage> ...    # discount / safety / material / forecast / lookahead

시나리오: 실제 수요 파일의 겹치지 않는 5주 구간 각각을 라운드 하나로 본다(`round_scenarios`).
지표: 라운드 Balance (초기재고로 시작, 종료일에 끝 = 게임 점수와 같은 정의).
채택 규칙: 현행(incumbent) 대비 짝비교 평균 차이 > 0, t >= 2, 그리고 최근 절반 시나리오에서도
평균 차이 > 0일 때만 바꾼다. 채택된 값은 reports/optimize_state.json에 쌓여 다음 단계의 현행이 된다.

실험에서는 P1·P2 모두 BestOf makespan 표를 쓴다(실제 P1 순서인 IG는 평균 0.9% 더 짧다 --
모든 정책에 비슷하게 작용하므로 비교에는 영향이 작다).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

from iog_sim.calendar import GameCalendar
from iog_sim.config import MAX_LOTS_PER_DAY, SimConfig
from iog_sim.demand.forecaster import SimpleExponentialSmoothingForecaster
from iog_sim.engine import PolicySet, Simulator
from iog_sim.examples.demo_run import make_demand, processing_times
from iog_sim.experiment import Scenario, cost_detail, round_scenarios, run_experiment
from iog_sim.material.policy import DualSourcingPolicy, SsPolicy
from iog_sim.production.lotsizing import DynamicLotSizing, FixedBatchDays
from iog_sim.production.safety_stock import (CumulativeEmpiricalSigma, CumulativeModelSigma,
                                             EndOfHorizon)
from iog_sim.production.sequencing import LTWK, NEH, SPT, BestOf, TabulatedMakespan
from iog_sim.sales.discount import MarginalProfitOptimizer, NoDiscount

REPORTS = Path(__file__).resolve().parent / "reports"
TABLE_PATH = REPORTS / "makespan_table.json"
STATE_PATH = REPORTS / "optimize_state.json"
# 프로세스 1개가 약 150MB. 16GB PC에서 22개를 띄우면 메모리가 모자라 작업자가 죽는다.
WORKERS = int(os.environ.get("IOG_WORKERS", min(8, os.cpu_count() or 1)))

# P2 수요 기록이 2025-06-22부터라 6주치 이력을 두고 첫 라운드를 시작한다.
FIRST_START = dt.date(2025, 8, 3)          # 일요일
SCENARIO_WEEKS = 5                          # 대회 라운드(11/1~12/4)와 같은 길이

# 현재 운영 정책 (current_state_report의 DP+한계이익할인)
BASE: Dict[str, object] = dict(
    lot="dp", lookahead=7, alpha=0.9, ss="end", service=0.95,
    z=1.65, cover=None, z_urgent=2.33, slack=1, discount="marginal", elasticity=1.0,
)

_SS = {"end": EndOfHorizon, "cum_model": CumulativeModelSigma, "cum_emp": CumulativeEmpiricalSigma}
_WORLD: dict = {}


def _world() -> dict:
    """프로세스마다 한 번: 수요 파일과 makespan 표를 읽는다."""
    if not _WORLD:
        table = json.loads(TABLE_PATH.read_text()) if TABLE_PATH.exists() else {}
        _WORLD["demand"] = make_demand()
        _WORLD["seq"] = TabulatedMakespan(BestOf([SPT(), LTWK(), NEH()]), table)
    return _WORLD


def build(params: dict, sc: Scenario) -> Simulator:
    p = {**BASE, **params}
    w = _world()
    cfg = SimConfig()
    cfg.discount_elasticity = p["elasticity"]
    ses = lambda: SimpleExponentialSmoothingForecaster(alpha=p["alpha"])
    discount = {"none": NoDiscount(),
                "marginal": MarginalProfitOptimizer(),
                "protect": MarginalProfitOptimizer(protect_until_receipt=True)}[p["discount"]]
    lot = DynamicLotSizing() if p["lot"] == "dp" else FixedBatchDays(int(p["lot"]))
    policy = PolicySet(
        forecaster={"P1": ses(), "P2": ses()},
        lot_sizing=lot,
        sequencer=w["seq"],
        planning_sequencer=None,
        material={
            "M1": SsPolicy(z=p["z"], cover_days=p["cover"]),
            "M2": DualSourcingPolicy(z_normal=p["z"], z_urgent=p["z_urgent"],
                                     slack_days=p["slack"], cover_days=p["cover"]),
            "M3": SsPolicy(z=p["z"], cover_days=p["cover"]),
        },
        discount=discount,
        safety_stock=_SS[p["ss"]](p["service"]),
        lookahead_days=p["lookahead"],
        label=str(params),
    )
    return Simulator(cfg, GameCalendar(round_ends=[sc.end]), w["demand"], policy, processing_times)


def scenarios():
    last = max(max(s) for s in _world()["demand"].series.values())
    return round_scenarios(FIRST_START, last, SCENARIO_WEEKS)


# ---------------------------------------------------------------------------
# makespan 표
# ---------------------------------------------------------------------------

def _table_entry(task):
    product, day, n = task
    p = processing_times(product, day, n)
    return TabulatedMakespan.key(p), BestOf([SPT(), LTWK(), NEH()]).solve(p).makespan


def build_table() -> None:
    week = [dt.date(2026, 11, 16) + dt.timedelta(days=i) for i in range(7)]   # 월~일
    tasks = [(prod, day, n) for prod in ("P1", "P2") for day in week
             for n in range(MAX_LOTS_PER_DAY, 0, -1)]                         # 큰 것부터
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        table = dict(ex.map(_table_entry, tasks, chunksize=8))
    TABLE_PATH.write_text(json.dumps(table))
    print(f"makespan 표 {len(table)}개 저장: {TABLE_PATH} ({time.time() - t0:.0f}초)")


# ---------------------------------------------------------------------------
# 단계 실행
# ---------------------------------------------------------------------------

def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def run_stage(name: str, candidates: Dict[str, dict]) -> dict:
    """candidates: {라벨: 현행에 덮어쓸 파라미터}. '현행'은 자동 포함."""
    state = load_state()
    incumbent = state.get("adopted", {})
    policies = {"현행": dict(incumbent)}
    policies.update({label: {**incumbent, **c} for label, c in candidates.items()})
    scs = scenarios()
    t0 = time.time()
    res = run_experiment(build, policies, scs, workers=WORKERS)
    df = res.to_frame()
    df.to_csv(REPORTS / f"optimize_{name}.csv", index=False, encoding="utf-8-sig")

    cmp = res.compare("현행")
    recent = {s.name for s in scs[len(scs) // 2:]}
    rec = df[df["scenario"].isin(recent)].pivot_table(index="scenario", columns="label",
                                                      values="balance")
    cmp["recent_diff"] = [float((rec[l] - rec["현행"]).mean()) for l in cmp["label"]]
    cmp["adopt"] = (cmp["mean_diff"] > 0) & (cmp["t"] >= 2) & (cmp["recent_diff"] > 0)

    mean_bal = df.groupby("label")["balance"].mean()
    print(f"\n=== {name}: 시나리오 {len(scs)}개 x 정책 {len(policies)}개, "
          f"{time.time() - t0:.0f}초 / 현행 평균 Balance {mean_bal['현행'] / 1e6:,.1f}M ===")
    show = cmp.copy()
    for c in ("mean_diff", "se", "recent_diff"):
        show[c] = (show[c] / 1e6).round(2)
    show["t"] = show["t"].round(2)
    print(show.to_string(index=False))

    winners = cmp[cmp["adopt"]]
    if len(winners):
        best = winners.iloc[0]["label"]
        state["adopted"] = policies[best]
        state.setdefault("history", []).append(
            {"stage": name, "adopted": best, "params": policies[best],
             "mean_diff": float(winners.iloc[0]["mean_diff"]), "t": float(winners.iloc[0]["t"])})
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> 채택: {best} {policies[best]}")
    else:
        state.setdefault("history", []).append({"stage": name, "adopted": None})
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print("-> 유의한 개선 없음: 현행 유지")
    detail = cost_detail(res)
    print((detail / 1e6).round(1).to_string())
    return state


STAGES = {
    # DP가 고정 n일 배치보다 나은지 (작업준비비 vs 잔업·재고비 균형 확인)
    "lotsizing": lambda: {f"{n}일배치": {"lot": n} for n in (1, 2, 3, 5)},
    "discount": lambda: {
        "할인없음": {"discount": "none"},
        "보호형할인": {"discount": "protect"},
    },
    "safety": lambda: {
        f"{ss}/{sl if sl else 'cost'}": {"ss": ss, "service": sl}
        for ss in ("end", "cum_model", "cum_emp") for sl in (None, 0.85, 0.90, 0.95, 0.98)
        if not (ss == "end" and sl == 0.95)
    },
    "material": lambda: {
        f"z{z}/{('EOQ' if c is None else f'{c}일')}": {"z": z, "cover": c}
        for z in (0.5, 1.0, 1.3, 1.65, 2.0) for c in (None, 3, 5, 7, 10)
        if not (z == 1.65 and c is None)
    },
    "urgent": lambda: {
        f"zu{zu}/slack{s}": {"z_urgent": zu, "slack": s}
        for zu in (1.65, 2.0, 2.33, 2.6) for s in (0, 1, 2)
        if not (zu == 2.33 and s == 1)
    },
    "forecast": lambda: {f"alpha{a}": {"alpha": a} for a in (0.3, 0.5, 0.7, 0.8, 1.0)},
    "lookahead": lambda: {f"look{d}": {"lookahead": d} for d in (0, 3, 14)},
}


def main(argv):
    if not argv:
        print(__doc__)
        return
    for name in argv:
        if name == "table":
            build_table()
        else:
            run_stage(name, STAGES[name]())


if __name__ == "__main__":
    main(sys.argv[1:])
