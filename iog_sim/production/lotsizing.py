"""Production 모듈 - Lot sizing (언제 얼마나 만들 것인가).

IOG의 비용 구조가 만드는 3중 trade-off
  (1) 작업준비비   P1 5,000,000 / P2 4,000,000  — 배치를 적게 열수록 유리
  (2) 완제품 재고비 30원/개·일 = 30,000원/Lot·일 — 미리 만들수록 불리
  (3) 잔업비        floor(Makespan/100) > 70시간 초과분 2배 요율 — 한 배치가 커질수록 불리

P1 setup 5,000,000원은 Lot 하루 보관비 30,000원의 약 167 Lot·일에 해당한다. 그래서 2일치 통합은
대개 이득이고, 3일치부터는 (3)의 잔업비가 급격히 붙는다. 경계가 makespan 100시간 단위 계단이라
"감으로 n일 묶기"가 아니라 실제 makespan을 평가하는 DP(`DynamicLotSizing`)로 푼다.

비용은 장부(엔진 `_close_day`)와 같은 규칙으로 센다.
  - 재고비: 매 달력일(주말·휴장일 포함) 마감 재고 x 재고유지비
  - 품절비: 못 판 수량 x 판매기회비
  - Lot 올림으로 생긴 잔량은 다음 배치가 그만큼 덜 만든다

P1은 평일만 생산하므로 금요일 배치가 토·일(공휴일 포함) 수요까지 포함해야 한다. P2는 매일 생산 가능하다.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..config import LOT_SIZE, MAX_LOTS_PER_DAY, ProductSpec
from ..costs import batch_production_cost

# (생산일, Lot 수) -> 예상 Makespan
MakespanOracle = Callable[[dt.date, int], float]


def lots_for(qty: int) -> int:
    """제품 개수 -> Lot 수 (올림). 1 Lot 미만도 배치를 열면 1 Lot."""
    return int(np.ceil(qty / LOT_SIZE)) if qty > 0 else 0


@dataclass
class LotPlan:
    """제품 1개에 대한 한 주 생산계획."""

    product: str
    lots_by_date: Dict[dt.date, int] = field(default_factory=dict)
    expected_cost: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)

    def total_lots(self) -> int:
        return sum(self.lots_by_date.values())


class LotSizingPolicy(ABC):
    name = "base"

    @abstractmethod
    def plan(
        self,
        product: ProductSpec,
        demand_by_date: Dict[dt.date, int],
        production_days: Sequence[dt.date],
        opening_inventory: int,
        makespan_oracle: MakespanOracle,
        availability_lag: int = 0,
    ) -> LotPlan:
        """demand_by_date에는 안전재고 증분이 이미 더해져 있다 (엔진 `_plan_week`).

        availability_lag: 생산일 D의 산출물이 D+lag부터 판매 가능 (현행 규칙 0 = 당일 판매).
        """


def _producer_for(day: dt.date, prod_days: Sequence[dt.date], lag: int) -> Optional[dt.date]:
    """day 수요를 댈 수 있는 가장 늦은 생산가능일. 없으면 None."""
    deadline = day - dt.timedelta(days=lag)
    c = [pd for pd in prod_days if pd <= deadline]
    return c[-1] if c else None


# ---------------------------------------------------------------------------
# 기준선: n일 고정 배치 (n=1 = Lot-for-Lot)
# ---------------------------------------------------------------------------

class FixedBatchDays(LotSizingPolicy):
    """수요일 n개를 한 배치로 묶어 첫 수요일(또는 직전 생산가능일)에 생산한다.

    n=1은 매 수요일 생산하는 Lot-for-Lot이다. what-if 비교에서 n=1,2,3,5를 DP와 나란히 돌린다.
    """

    def __init__(self, n_days: int = 1):
        self.n_days = n_days
        self.name = "L4L" if n_days == 1 else f"Fixed{n_days}"

    def plan(self, product, demand_by_date, production_days, opening_inventory,
             makespan_oracle, availability_lag=0) -> LotPlan:
        plan = LotPlan(product=product.code)
        prod_days = sorted(production_days)
        demand_days = sorted(demand_by_date)
        inv = opening_inventory
        for i in range(0, len(demand_days), self.n_days):
            window = demand_days[i:i + self.n_days]
            need = sum(demand_by_date[d] for d in window)
            pd_ = _producer_for(window[0], prod_days, availability_lag)
            if pd_ is not None:
                already = plan.lots_by_date.get(pd_, 0)
                lots = min(lots_for(max(0, need - inv)), MAX_LOTS_PER_DAY - already)
                if lots > 0:
                    plan.lots_by_date[pd_] = already + lots
                    inv += lots * LOT_SIZE
            # 생산할 수 없는 구간(예: 주 초 일·월)도 기초재고는 소모된다
            inv = max(0, inv - need)
        return _evaluate(plan, product, demand_by_date, opening_inventory, makespan_oracle,
                         availability_lag)


# ---------------------------------------------------------------------------
# 동적 최적화 — 운영 본안
# ---------------------------------------------------------------------------

class DynamicLotSizing(LotSizingPolicy):
    """Wagner-Whitin형 전진 DP. 배치 비용이 선형이 아니라 makespan 함수라는 점을 반영한다.

    상태: (다음 배치를 여는 수요일 i, 앞 배치에서 넘어온 잔량 carry)
      한 배치가 수요일 i..j-1을 맡으면 생산량 = ceil((수요합 - carry) / 1000) Lot이고,
      Lot 올림으로 남은 잔량이 다음 상태의 carry가 된다. carry는 1,000개 미만이라 상태 수가 작다.

    목적함수 (장부와 상수항만 다르다)
      작업준비비 + 인건비(makespan 오라클)
      + 생산량 x 재고유지비 x (생산일 ~ 계획 마지막 날)        <- 재고비
      + 품절량 x (판매기회비 + 재고유지비 x 그날 ~ 마지막 날)   <- 하루 상한(500 Job)에 걸릴 때만
    재고비 식: 마감 재고 합 = Σ(팔린 날 - 만든 날)이므로, 수요 쪽(상수)을 빼면 '만든 수량 x 남은 일수'만
    계획에 따라 달라진다. 품절된 수량은 팔리지 않으므로 그 상수에서 빠진 만큼을 되돌려 더한다.
    """

    name = "DP"

    def plan(self, product, demand_by_date, production_days, opening_inventory,
             makespan_oracle, availability_lag=0) -> LotPlan:
        demand_days = sorted(demand_by_date)
        n = len(demand_days)
        if n == 0:
            return LotPlan(product=product.code)

        prod_days = sorted(production_days)
        need = [demand_by_date[d] for d in demand_days]
        inv = opening_inventory                      # 기초재고는 앞에서부터 차감 (재고비는 상수)
        for k in range(n):
            used = min(inv, need[k])
            need[k] -= used
            inv -= used
        prefix = np.concatenate([[0], np.cumsum(need)]).astype(int).tolist()

        h = product.holding_cost
        after_last = demand_days[-1] + dt.timedelta(days=1)

        cost_cache: Dict[tuple, float] = {}

        def batch_cost(prod_day: dt.date, lots: int) -> float:
            if lots <= 0:
                return 0.0
            key = (prod_day, lots)
            if key not in cost_cache:
                ms = makespan_oracle(prod_day, lots)
                cost_cache[key] = float(batch_production_cost(ms, product, lots).total)
            return cost_cache[key]

        # states[i] = {carry: (누적비용, (이전 i, 이전 carry, 생산일, Lot))}
        states = [dict() for _ in range(n + 1)]
        states[0][0] = (0.0, None)

        def relax(j: int, carry: int, cost: float, back: tuple) -> None:
            cur = states[j].get(carry)
            if cur is None or cost < cur[0]:
                states[j][carry] = (cost, back)

        for i in range(n):
            pd_ = _producer_for(demand_days[i], prod_days, availability_lag)
            lost_unit = product.stockout_cost + h * (after_last - demand_days[i]).days
            for carry, (cost, _) in list(states[i].items()):
                if pd_ is None:                      # 생산 불가: 잔량으로만 댄다
                    lost = max(0, need[i] - carry)
                    relax(i + 1, max(0, carry - need[i]), cost + lost * lost_unit,
                          (i, carry, None, 0))
                    continue
                for j in range(i + 1, n + 1):
                    total = prefix[j] - prefix[i]
                    lots = lots_for(total - carry)
                    if lots > MAX_LOTS_PER_DAY:
                        # 구간을 넓히면 더 커지기만 한다. 단일 구간조차 넘으면 상한까지 만들고 품절 감수
                        if j > i + 1:
                            break
                        lots = MAX_LOTS_PER_DAY
                    produced = lots * LOT_SIZE
                    lost = max(0, total - carry - produced)
                    c = (cost + batch_cost(pd_, lots)
                         + produced * h * (after_last - pd_).days
                         + lost * lost_unit)
                    relax(j, carry + produced + lost - total, c, (i, carry, pd_, lots))

        carry = min(states[n], key=lambda k: states[n][k][0])
        plan = LotPlan(product=product.code)
        i = n
        while states[i][carry][1] is not None:
            prev_i, prev_carry, pd_, lots = states[i][carry][1]
            if pd_ is not None and lots > 0:
                plan.lots_by_date[pd_] = min(MAX_LOTS_PER_DAY,
                                             plan.lots_by_date.get(pd_, 0) + lots)
            i, carry = prev_i, prev_carry
        return _evaluate(plan, product, demand_by_date, opening_inventory, makespan_oracle,
                         availability_lag)


# ---------------------------------------------------------------------------
# 공통 평가기
# ---------------------------------------------------------------------------

def _evaluate(plan: LotPlan, product: ProductSpec, demand_by_date, opening_inventory,
              makespan_oracle, availability_lag: int = 0) -> LotPlan:
    """계획의 예상 비용 내역 (작업준비·인건비·잔업·재고비·예상품절).

    장부와 같이 첫 날부터 마지막 수요일까지 **매 달력일** 마감 재고에 재고유지비를 매긴다.
    생산일 D의 산출물은 D+lag부터 판매 가능하지만 생산 당일부터 재고로 잡힌다.
    """
    setup = labor = ot_hours = 0
    for d, lots in sorted(plan.lots_by_date.items()):
        bc = batch_production_cost(makespan_oracle(d, lots), product, lots)
        setup += bc.setup
        labor += bc.labor
        ot_hours += bc.overtime_hours

    holding = shortage = 0
    dates = list(plan.lots_by_date) + list(demand_by_date)
    if dates:
        lag = dt.timedelta(days=availability_lag)
        inv = opening_inventory
        day, last = min(dates), max(dates)
        while day <= last:
            inv += plan.lots_by_date.get(day - lag, 0) * LOT_SIZE
            dem = demand_by_date.get(day, 0)
            sold = min(inv, dem)
            shortage += dem - sold
            inv -= sold
            waiting = plan.lots_by_date.get(day, 0) * LOT_SIZE if availability_lag > 0 else 0
            holding += (inv + waiting) * product.holding_cost
            day += dt.timedelta(days=1)

    plan.detail = {
        "setup_cost": setup,
        "labor_cost": labor,
        "overtime_hours": ot_hours,
        "fg_holding_cost": holding,
        "expected_shortage_units": shortage,
        "expected_stockout_cost": shortage * product.stockout_cost,
    }
    plan.expected_cost = setup + labor + holding + shortage * product.stockout_cost
    return plan
