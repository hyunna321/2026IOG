"""공용 부품: 실수요 로더, 처리시간표, 운영 정책(POLICIES), 시뮬레이터 빌더.

current_state_report / round_plan이 여기서 정책을 가져간다. 정책끼리의 비교·튜닝은
`examples/optimize.py`(여러 라운드 시나리오 짝비교)에서 한다.

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
from iog_sim.material.policy import DualSourcingPolicy, SsPolicy
from iog_sim.production.lotsizing import DynamicLotSizing, FixedBatchDays
from iog_sim.production.safety_stock import CumulativeModelSigma, EndOfHorizon, SafetyStockRule
from iog_sim.production.sequencing import (BestOf, DiskCached, IteratedGreedy, LTWK, NEH, SPT,
                                          Memoized, load_processing_times)
from iog_sim.sales.discount import MarginalProfitOptimizer, NoDiscount

DATA_DIR = Path(__file__).resolve().parent.parent    # iog_sim/iog_sim
_CAL = GameCalendar()

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
# P1은 BestOf 해에서 출발하는 Iterated Greedy(800회, 시드 고정)로 다듬는다. 배치당 수 초~십수 초라
# 결과를 reports/sequence_cache.json에 남겨 같은 (요일, Job 수)는 다음 실행부터 바로 쓴다.
# P2는 순서 영향이 작아 BestOf 그대로.
BASE_SEQUENCER = Memoized(BestOf([SPT(), LTWK(), NEH()]))
SEQUENCE_CACHE = Path(__file__).resolve().parent / "reports" / "sequence_cache.json"
SEQUENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
_P1_IG800 = DiskCached(IteratedGreedy(iterations=800, start=BASE_SEQUENCER), SEQUENCE_CACHE,
                       label="IG800(BestOf)")
# 2026-11-14: 상위팀 P1 makespan이 같은 처리시간표에서 1~3% 짧았다. IG800 해에서 시드 1·2로
# 8,000회씩 이어 돌리면 70~180 줄어든다 (하루 잔업 최대 4시간). 배치당 약 3분이라 첫 실행만 느리고,
# 같은 (요일, Job 수)는 캐시에서 바로 읽는다. IG는 시작 해보다 나빠지지 않는다.
SEQUENCER = {
    "P1": DiskCached(
        IteratedGreedy(iterations=8000, seed=2,
                       start=IteratedGreedy(iterations=8000, seed=1, start=_P1_IG800)),
        SEQUENCE_CACHE, label="IG800+8000s1+8000s2(BestOf)"),
    "P2": BASE_SEQUENCER,
}
# Lot sizing DP용 makespan. 실제 순서와 같은 BestOf로 구해야 잔업 경계 판단이 맞다
# (LTWK는 P1 makespan을 약 15% 크게 본다). 배치 크기 하나에 0.1~0.6초라 같은 캐시 파일에 남긴다.
# P1 실제 순서(IG)는 BestOf보다 makespan이 평균 0.9% 더 짧다 -- DP는 이만큼 보수적으로 본다.
PLANNING_SEQUENCER = DiskCached(BASE_SEQUENCER, SEQUENCE_CACHE, label="BestOf")


# M1·M3 발주 크기 = EOQ x 0.75 (2026-11-13 재검증: 0.7~0.8배 모두 +8~10M/라운드, M2는 줄이면 손해)
M13_LOT_SCALE = 0.75
# M2 긴급 판단용 안전계수 (2026-11-13 재검증: 2.33 -> 1.65 +0.7M/라운드, 31라운드 중 진 적 없음)
M2_URGENT_Z = 1.65


def base_policy(label: str, lot_sizing, discount, z: float = 1.65,
                safety_stock: SafetyStockRule = EndOfHorizon(0.95)) -> PolicySet:
    return PolicySet(
        forecaster={"P1": SimpleExponentialSmoothingForecaster(),
                    "P2": SimpleExponentialSmoothingForecaster()},
        lot_sizing=lot_sizing,
        sequencer=SEQUENCER,
        planning_sequencer=PLANNING_SEQUENCER,
        material={
            "M1": SsPolicy(z=z, lot_scale=M13_LOT_SCALE),
            "M2": DualSourcingPolicy(z_normal=z, z_urgent=M2_URGENT_Z),   # 주문비 3M이라 EOQ 그대로가 최적
            "M3": SsPolicy(z=z, lot_scale=M13_LOT_SCALE),
        },
        discount=discount,
        safety_stock=safety_stock,
        label=label,
    )


POLICIES = [
    # 고정 n일 배치 (n=1 = L4L). DP가 이들보다 나은지로 작업준비비 vs 잔업·재고비 균형을 확인한다.
    *(base_policy(f"{n}일배치+할인없음", FixedBatchDays(n), NoDiscount()) for n in (1, 2, 3, 5)),
    base_policy("DP+할인없음", DynamicLotSizing(), NoDiscount()),
    base_policy("DP+한계이익할인", DynamicLotSizing(), MarginalProfitOptimizer()),
    # 운영 정책 (2026-11-14~). 11-12 P1 8,500개 품절 뒤, 주 마지막 날에만 버퍼를 두던 EndOfHorizon 대신
    # 누적 예측오차 버퍼(서비스수준 0.75)를 쓴다. 12라운드 백테스트 평균 +2.3M/라운드(유의하지 않음),
    # 11-15 주 몬테카를로에서 품절확률 73% -> 45%, 순비용 -3M.
    base_policy("DP+한계이익할인+누적버퍼0.75", DynamicLotSizing(), MarginalProfitOptimizer(),
                safety_stock=CumulativeModelSigma(0.75)),
]


def build(policy: PolicySet) -> Simulator:
    return Simulator(
        cfg=SimConfig(),
        calendar=GameCalendar(),
        demand=make_demand(),
        policies=policy,
        processing_times=processing_times,
    )

