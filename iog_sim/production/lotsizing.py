"""Production 모듈 - Lot sizing (언제 얼마나 만들 것인가).

IOG의 비용 구조가 만드는 3중 trade-off
  (1) 작업준비비   P1 5,000,000 / P2 2,000,000  — 배치를 적게 열수록 유리
  (2) 완제품 재고비 30원/개·일 = 30,000원/Lot·일 — 미리 만들수록 불리
  (3) 잔업비        floor(Makespan/100) > 70시간 초과분 2배 요율 — 한 배치가 커질수록 불리

(2)의 크기가 결정적이다. P1 setup 5,000,000원은 Lot 하루 보관비 30,000원의 약 167 Lot·일에
해당하므로, "하루 앞서 만드는 물량이 167 Lot(=167,000개)을 넘으면 배치 통합은 손해"다.
P1 일 수요가 7만 개(70 Lot) 수준이면 2일치 통합(=70 Lot을 1일 선행 보관, 2,100,000원)은
setup 1회(5,000,000원)를 아끼므로 유리하지만, 3일치부터는 (3)의 잔업비가 급격히 붙는다.
따라서 이 문제는 "감으로 2~3일 묶기"가 아니라 makespan을 실제로 평가하는 DP로 풀어야 한다.

또 하나 구조적 제약: P2 수요는 365일이지만 생산은 평일만 가능하다.
=> 금요일 배치는 반드시 토·일 수요까지 포함해야 하며, 이때 P2의 주말 재고비는 불가피 비용이다.
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
        safety_stock: int = 0,
        availability_lag: int = 1,
    ) -> LotPlan:
        """availability_lag: 생산일 D의 산출물이 D+lag부터 판매 가능.

        IOG는 당일 생산분을 당일 팔 수 없으므로 lag=1이 기본이다. 이 한 칸 때문에
        '수요일에 맞춰 생산'이라는 L4L의 정의 자체가 '수요 전날까지 생산'으로 바뀐다.
        """


# ---------------------------------------------------------------------------
# 기준선 정책
# ---------------------------------------------------------------------------

class LotForLot(LotSizingPolicy):
    """수요일에 맞춰 당일(또는 직전 생산가능일) 생산. setup을 최대한 많이 지불하는 안."""

    name = "L4L"

    def plan(self, product, demand_by_date, production_days, opening_inventory,
             makespan_oracle, safety_stock=0, availability_lag=1) -> LotPlan:
        plan = LotPlan(product=product.code)
        inv = opening_inventory
        prod_days = sorted(production_days)
        for d in sorted(demand_by_date):
            need = demand_by_date[d] + (safety_stock if d == max(demand_by_date) else 0)
            # 이 수요를 감당할 가장 늦은 생산가능일 (당일 생산분은 당일 판매 불가)
            deadline = d - dt.timedelta(days=availability_lag)
            candidates = [pd for pd in prod_days if pd <= deadline]
            if not candidates:
                # 이 주 안에서는 생산 불가(예: 일·월). 기초재고로만 충당되므로
                # 그만큼 재고를 반드시 차감해야 이후 날짜의 소요량이 맞는다.
                inv = max(0, inv - demand_by_date[d])
                continue
            pd_ = candidates[-1]
            short = max(0, need - inv)
            already = plan.lots_by_date.get(pd_, 0)
            lots = min(lots_for(short), MAX_LOTS_PER_DAY - already)
            if lots > 0:
                plan.lots_by_date[pd_] = already + lots
                inv += lots * LOT_SIZE
            inv -= demand_by_date[d]
            inv = max(inv, 0)
        return _evaluate(plan, product, demand_by_date, opening_inventory, makespan_oracle,
                         availability_lag)


class FixedBatchDays(LotSizingPolicy):
    """n일치를 한 배치로 묶는 고정 규칙. what-if 스윕용(n=1,2,3,5)."""

    name = "FixedDays"

    def __init__(self, n_days: int = 2):
        self.n_days = n_days

    def plan(self, product, demand_by_date, production_days, opening_inventory,
             makespan_oracle, safety_stock=0, availability_lag=1) -> LotPlan:
        plan = LotPlan(product=product.code)
        prod_days = sorted(production_days)
        demand_days = sorted(demand_by_date)
        inv = opening_inventory
        i = 0
        while i < len(demand_days):
            window = demand_days[i:i + self.n_days]
            need = sum(demand_by_date[d] for d in window)
            if window[-1] == demand_days[-1]:
                need += safety_stock
            deadline = window[0] - dt.timedelta(days=availability_lag)
            candidates = [pd for pd in prod_days if pd <= deadline]
            if not candidates:
                inv = max(0, inv - sum(demand_by_date[d] for d in window))
                i += self.n_days
                continue
            pd_ = candidates[-1]
            already = plan.lots_by_date.get(pd_, 0)
            lots = min(lots_for(max(0, need - inv)), MAX_LOTS_PER_DAY - already)
            if lots > 0:
                plan.lots_by_date[pd_] = already + lots
                inv += lots * LOT_SIZE
            inv = max(0, inv - sum(demand_by_date[d] for d in window))
            i += self.n_days
        return _evaluate(plan, product, demand_by_date, opening_inventory, makespan_oracle,
                         availability_lag)


# ---------------------------------------------------------------------------
# 동적 최적화 — 이 문제에 맞는 본안
# ---------------------------------------------------------------------------

class DynamicLotSizing(LotSizingPolicy):
    """Wagner-Whitin 형 DP. 단, 배치 비용이 선형이 아니라 makespan 함수라는 점을 반영.

    상태: '다음 배치를 여는 수요일 인덱스'
    f(i) = min_{j>i} [ batch_cost(생산일(i), 수요 i..j-1 합) + holding(i..j-1) + f(j) ]

    batch_cost 안에서 makespan_oracle을 호출하므로, 잔업 경계(70시간)를 넘는 순간의
    비용 점프가 DP에 그대로 반영된다. 이것이 단순 EOQ/WW와 결정적으로 다른 부분이다.

    계산량: 수요일 7일 * 생산일 5일 수준이라 완전탐색 DP가 그대로 가능하다.
    makespan_oracle 호출이 비싸므로 (날짜, Lot수) 캐시를 반드시 쓸 것.
    """

    name = "DP"

    def plan(self, product, demand_by_date, production_days, opening_inventory,
             makespan_oracle, safety_stock=0, availability_lag=1) -> LotPlan:
        demand_days = sorted(demand_by_date)
        n = len(demand_days)
        if n == 0:
            return LotPlan(product=product.code)

        prod_days = sorted(production_days)
        need = [demand_by_date[d] for d in demand_days]
        need[-1] += safety_stock
        # 기초재고는 앞에서부터 차감
        inv = opening_inventory
        for k in range(n):
            used = min(inv, need[k])
            need[k] -= used
            inv -= used

        cache: Dict[tuple, float] = {}

        def batch_cost(prod_day: dt.date, lots: int) -> float:
            if lots <= 0:
                return 0.0
            key = (prod_day, lots)
            if key not in cache:
                ms = makespan_oracle(prod_day, lots)
                cache[key] = float(batch_production_cost(ms, product, lots).total)
            return cache[key]

        def producer_for(day: dt.date) -> Optional[dt.date]:
            deadline = day - dt.timedelta(days=availability_lag)
            c = [pd for pd in prod_days if pd <= deadline]
            return c[-1] if c else None

        INF = float("inf")
        f = [INF] * (n + 1)
        choice = [0] * (n + 1)
        f[n] = 0.0
        for i in range(n - 1, -1, -1):
            pd_ = producer_for(demand_days[i])
            if pd_ is None:                      # 생산 불가 -> 다음 구간으로 넘김
                f[i] = f[i + 1]
                choice[i] = i + 1
                continue
            for j in range(i + 1, n + 1):
                qty = sum(need[i:j])
                lots = lots_for(qty)
                if lots > MAX_LOTS_PER_DAY:
                    # 하루 상한 초과. 구간을 더 넓히면 더 커지기만 하므로 j 탐색을 끊는다.
                    # 단 단일 구간(j=i+1)조차 넘으면 그날 상한까지만 만들고 부족은 감수한다.
                    if j > i + 1:
                        break
                    lots = MAX_LOTS_PER_DAY
                hold = sum(
                    need[k] * max(0, (demand_days[k] - pd_).days) * product.holding_cost
                    for k in range(i, j)
                )
                # 과생산분(Lot 반올림 잔량)도 다음 날로 이월되며 재고비가 붙는다
                leftover = lots * LOT_SIZE - qty
                hold += leftover * product.holding_cost
                c = batch_cost(pd_, lots) + hold + f[j]
                if c < f[i]:
                    f[i] = c
                    choice[i] = j

        plan = LotPlan(product=product.code)
        i = 0
        while i < n:
            j = choice[i]
            pd_ = producer_for(demand_days[i])
            qty = sum(need[i:j])
            if pd_ is not None:
                already = plan.lots_by_date.get(pd_, 0)
                lots = min(lots_for(qty), MAX_LOTS_PER_DAY - already)
                if lots > 0:
                    plan.lots_by_date[pd_] = already + lots
            i = j if j > i else i + 1
        plan.expected_cost = f[0]
        return _evaluate(plan, product, demand_by_date, opening_inventory, makespan_oracle,
                         availability_lag)


# ---------------------------------------------------------------------------
# 공통 평가기
# ---------------------------------------------------------------------------

def _evaluate(plan: LotPlan, product: ProductSpec, demand_by_date, opening_inventory,
              makespan_oracle, availability_lag: int = 1) -> LotPlan:
    """계획의 예상 비용 내역을 채운다 (설정·인건비·잔업·재고비·예상품절).

    생산일 D의 산출물은 D+lag부터 판매 가능하므로, 가용시점 기준으로 재고를 굴린다.
    """
    setup = labor = 0
    ot_hours = 0
    for d, lots in sorted(plan.lots_by_date.items()):
        ms = makespan_oracle(d, lots)
        bc = batch_production_cost(ms, product, lots)
        setup += bc.setup
        labor += bc.labor
        ot_hours += bc.overtime_hours

    lag = dt.timedelta(days=availability_lag)
    available_on = {d + lag: lots for d, lots in plan.lots_by_date.items()}

    inv = opening_inventory
    holding = shortage = 0
    all_days = sorted(set(list(plan.lots_by_date) + list(available_on) + list(demand_by_date)))
    for d in all_days:
        # 생산 당일에는 재고로만 잡히고(재고유지비 부과) 판매 가능해지는 것은 d+lag
        inv += available_on.get(d, 0) * LOT_SIZE
        dem = demand_by_date.get(d, 0)
        sold = min(inv, dem)
        shortage += dem - sold
        inv -= sold
        holding += (inv + plan.lots_by_date.get(d, 0) * LOT_SIZE
                    * (1 if availability_lag > 0 else 0)) * product.holding_cost

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


def safety_stock_units(sigma_cum: float, service_level: float = 0.95) -> int:
    """완제품 안전재고. 품절비 250원 vs 재고비 30원/일이므로 임계비율이 매우 높다.

    뉴스벤더 임계비율 = Cu / (Cu + Co) = 250 / (250 + 30) ≈ 0.893
    => 하루짜리 관점이면 서비스수준 약 89%가 최적. 다만 잉여재고는 다음 날에도
       팔 수 있으므로(폐기 없음) 실제 과잉비용은 30원/일보다 작다.
       따라서 0.90~0.97 구간을 what-if로 스윕해 확정할 것.
    """
    from scipy.stats import norm

    return int(max(0, round(norm.ppf(service_level) * sigma_cum)))
