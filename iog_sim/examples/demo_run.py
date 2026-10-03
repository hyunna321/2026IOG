"""엔드투엔드 스모크 런.

제공된 실데이터(수요 CSV, ProcessingTimeTable CSV)로 엔진을 한 바퀴 돌려,
배선(이벤트 순서, 비용 회계, 정책 인터페이스)이 맞물리는지 확인한다.

수요예측은 alpha=0.9 SES(SimpleExponentialSmoothingForecaster)로 고정한다.
휴장일은 `calendar.MARKET_HOLIDAYS_BY_MONTH`에 월별로 하드코딩한다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import numpy as np

from iog_sim.calendar import GameCalendar
from iog_sim.config import SimConfig
from iog_sim.demand.forecaster import SimpleExponentialSmoothingForecaster
from iog_sim.demand.generator import HistoricalReplay, load_historical_replay
from iog_sim.engine import PolicySet, Simulator
from iog_sim.experiment import cost_decomposition, cost_detail, run_experiment
from iog_sim.material.policy import DualSourcingPolicy, SsPolicy
from iog_sim.production.lotsizing import DynamicLotSizing, FixedBatchDays, LotForLot
from iog_sim.production.safety_stock import EndOfHorizon, SafetyStockRule
from iog_sim.production.sequencing import BestOf, LTWK, NEH, SPT, Memoized, load_processing_times
from iog_sim.sales.discount import MarginalProfitOptimizer, NoDiscount

DATA_DIR = Path(__file__).resolve().parent.parent    # iog_sim/iog_sim
_CAL = GameCalendar()

START = dt.date(2026, 1, 5)
END = dt.date(2026, 3, 14)      # 10주 (앞 2주는 워밍업으로 집계 제외)


# --- 1. 실수요 (제공된 CSV 재생) ---------------------------------------------

_DEMAND_SUFFIXES = (".csv", ".xls", ".xlsx")


def _latest_demand_file(product: str) -> str:
    """demand_data_{product}_{YYYYMMDD}.{csv,xls,xlsx} 중 가장 최근 추출본.

    Demand 화면 다운로드는 라운드에 따라 CSV로도 구형 `.xls`로도 떨어지므로 확장자를
    가리지 않는다. 정렬 기준은 파일명의 YYYYMMDD(=추출일)이고, 같은 날짜가 여러 확장자로
    있으면 `_DEMAND_SUFFIXES` 순서(CSV 우선)로 고른다. 매주 새 파일을 넣기만 하면 된다."""
    folder = DATA_DIR / "demand"
    matches = [p for suffix in _DEMAND_SUFFIXES
               for p in folder.glob(f"demand_data_{product}_*{suffix}")]
    if not matches:
        raise FileNotFoundError(
            f"demand_data_{product}_*{{{','.join(_DEMAND_SUFFIXES)}}} 를 찾을 수 없음")
    return str(max(matches, key=lambda p: (p.stem, -_DEMAND_SUFFIXES.index(p.suffix.lower()))))


def make_demand() -> HistoricalReplay:
    return load_historical_replay({
        "P1": _latest_demand_file("P1"),
        "P2": _latest_demand_file("P2"),
    })


# --- 2. 실처리시간표 (요일별 500-job, 20-machine 테이블; t_=P1, t2_=P2) ------

_PT_PREFIX = {"P1": "t", "P2": "t2"}
_PT_CACHE: Dict[tuple, np.ndarray] = {}


def processing_times(product: str, date: dt.date, n_lots: int) -> np.ndarray:
    weekday = _CAL.weekday_key(date)
    key = (product, weekday)
    if key not in _PT_CACHE:
        path = DATA_DIR / "production" / f"{_PT_PREFIX[product]}_500_20_{weekday}.csv"
        _PT_CACHE[key] = load_processing_times(str(path))
    table = _PT_CACHE[key]
    if n_lots > len(table):
        # 조용히 잘라 내면 인건비가 과소평가되고 DP가 '거대 배치가 공짜'라고 착각한다.
        raise ValueError(
            f"{product} {date}: {n_lots} Job 요청, 처리시간표는 {len(table)} Job까지. "
            f"config.MAX_LOTS_PER_DAY를 넘는 계획이 만들어졌다.")
    return table[:n_lots]


# --- 3. 정책 세트 -----------------------------------------------------------

# 실제 투입 순서. 정책끼리 같은 처리시간표를 반복해 풀므로 결과를 공유한다.
SEQUENCER = Memoized(BestOf([SPT(), LTWK(), NEH()]))


def base_policy(label: str, lot_sizing, discount, z: float = 1.65,
                safety_stock: SafetyStockRule = EndOfHorizon(0.95)) -> PolicySet:
    return PolicySet(
        forecaster={"P1": SimpleExponentialSmoothingForecaster(),
                    "P2": SimpleExponentialSmoothingForecaster()},
        lot_sizing=lot_sizing,
        sequencer=SEQUENCER,
        planning_sequencer=LTWK(),       # DP가 수백 번 호출하므로 빠른 규칙 유지
        material={
            "M1": SsPolicy(z=z),
            "M2": DualSourcingPolicy(z_normal=z),
            "M3": SsPolicy(z=z),
        },
        discount=discount,
        safety_stock=safety_stock,
        label=label,
    )


POLICIES = [
    base_policy("L4L+할인없음", LotForLot(), NoDiscount()),
    base_policy("2일배치+할인없음", FixedBatchDays(2), NoDiscount()),
    base_policy("DP+할인없음", DynamicLotSizing(), NoDiscount()),
    base_policy("DP+한계이익할인", DynamicLotSizing(), MarginalProfitOptimizer()),
]


def build(policy: PolicySet) -> Simulator:
    return Simulator(
        cfg=SimConfig(),
        calendar=GameCalendar(),
        demand=make_demand(),
        policies=policy,
        processing_times=processing_times,
    )


if __name__ == "__main__":
    result = run_experiment(build, POLICIES, START, END, n_replications=1)
    print(cost_decomposition(result).to_string())
    print()
    print(cost_detail(result).to_string())
    print()
    print(result.compare(baseline="L4L+할인없음").to_string(index=False))
