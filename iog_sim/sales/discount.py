"""Sales 모듈 - 가격할인 정책.

수요모형 (1차 가정, 실측으로 교체할 것)
    D(d) = D0 * (1 + ε * d)      d = 할인율, ε = 탄력성
    문제소개 예시(10% 할인 -> 100개→110개)는 ε = 1.0,
    Sales 화면 관측(6% 증가)은 그때의 할인율에 따라 ε가 다름을 뜻한다. => 실측 회귀 필요.

핵심 산식 (재고가 충분한 경우, ε=1, p=250 기준)
    Δ매출  = p·D·(εd - d - εd²)  = -p·D·d²        (ε=1이면 항상 음수)
    Δ재고비 = +30 · D·εd · T                       (T = 회피한 보관일수)
    합계   = D·d·(30T - 250d) > 0  ⟺  d < 0.12·T

  즉 "할인은 그 자체로는 손해이고, 남는 재고를 하루라도 일찍 털 때만 이득"이다.
  ε=1 기준 손익분기 할인율은 재고 1일 이월 회피당 약 12%.
  반대로 재고가 수요보다 적으면 할인은 품절비(250원/개·일)만 키우므로 반드시 0이어야 한다.

따라서 할인은 '재고 상태 의존 정책'이다: d* = f(재고 / 예상수요).

**필수 버퍼**: 당일 생산분을 당일 팔 수 있어도 재고 하루치는 잉여가 아니다. Lot sizing이 배치를
몰아 잡아 무생산일이 흔하고, 자재가 모자라면 계획 Job이 잘리기 때문이다(`MarginalProfitOptimizer`).
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np

from ..config import ProductSpec


def effective_demand(base_demand: float, discount: float, elasticity: float) -> int:
    return int(max(0, round(base_demand * (1.0 + elasticity * discount))))


def unit_price(product: ProductSpec, discount: float) -> float:
    return product.price * (1.0 - discount)


class DiscountPolicy(ABC):
    name = "base"

    @abstractmethod
    def decide(
        self,
        date: dt.date,
        product: ProductSpec,
        inventory: int,
        demand_forecast: Dict[dt.date, float],
        elasticity: float,
        max_discount: float = 0.30,
        scheduled_receipts: Optional[Dict[dt.date, float]] = None,
    ) -> float:
        """scheduled_receipts: {날짜 -> 그날 판매 가능해지는 생산 입고량}.

        '언제 물건이 더 들어오는가'를 모르면 잉여재고가 며칠 남을지 추정할 수 없다.
        모르면 None으로 두되, 그 경우 정책은 보수적으로(할인을 덜 하는 쪽으로) 동작해야 한다.
        엔진은 생산일 + `availability_lag`를 키로 이 표를 만든다(`_decide_discount`).
        """


class NoDiscount(DiscountPolicy):
    name = "none"

    def decide(self, date, product, inventory, demand_forecast, elasticity, max_discount=0.30,
               scheduled_receipts=None) -> float:
        return 0.0


class MarginalProfitOptimizer(DiscountPolicy):
    """할인율 그리드에서 (매출 - 재고비 - 품절비)를 최대화.

    **필수 버퍼 보호 (required_cover)**
    당일 생산분을 당일 팔 수 있어도 하루치 재고는 여전히 필요하다. Lot sizing이 배치를 몰아
    잡아 생산이 0인 날이 흔하고, 자재가 모자라면 그날 Job이 잘리기 때문이다.
    이 하루치를 잉여로 오인해 할인으로 털면, 다음 날 팔 물건이 사라져 품절비 250원/개·일을 문다.
    (스모크 런에서 실제로 관측: 할인 적용 시 품절비 7M -> 88M, Balance 206M -> 25M)

    따라서 할인 대상은 `재고 - required_cover × 익일수요`를 넘는 부분뿐이며,
    그 이하에서는 어떤 할인율도 검토하지 않는다.

    lookahead일 동안의 예상 수요와 **예정 생산 입고**를 함께 보고 잉여가 며칠 남을지 추정한다.
    자연 소진이 빠르면 할인의 재고비 절감 효과가 작아 d*=0이 된다.
    """

    name = "marginal"

    def __init__(self, grid: Sequence[float] = tuple(np.arange(0.0, 0.31, 0.01)),
                 lookahead_days: int = 3, required_cover: float = 1.0):
        self.grid = list(grid)
        self.lookahead_days = lookahead_days
        self.required_cover = required_cover

    def _carry_days(self, excess: float, date: dt.date,
                    demand_forecast: Dict[dt.date, float],
                    scheduled_receipts: Optional[Dict[dt.date, float]] = None) -> float:
        """잉여분이 며칠 동안 재고로 남아 있을지 추정.

        예정 생산 입고를 빼먹으면 '곧 자연 소진된다'고 착각해 carry를 과소평가하고,
        반대로 입고를 더하면 잉여가 더 오래 남으므로 carry가 커진다. 둘 다 반영한다.
        """
        remaining = excess
        days = 0.0
        for i in range(1, self.lookahead_days + 1):
            d = date + dt.timedelta(days=i)
            if scheduled_receipts:
                remaining += scheduled_receipts.get(d, 0.0)
            remaining -= demand_forecast.get(d, 0.0)
            if remaining <= 0:
                return days
            days += 1
        return days

    def decide(self, date, product, inventory, demand_forecast, elasticity, max_discount=0.30,
               scheduled_receipts: Optional[Dict[dt.date, float]] = None) -> float:
        base = demand_forecast.get(date, 0.0)
        if base <= 0 or inventory <= 0:
            return 0.0

        # 익일 수요분은 규칙상 반드시 재고로 들고 있어야 한다 -> 할인 대상에서 제외
        next_day = date + dt.timedelta(days=1)
        buffer_need = demand_forecast.get(next_day, base) * self.required_cover
        excess = inventory - base - buffer_need
        if excess <= 0:
            return 0.0

        carry = self._carry_days(excess, date, demand_forecast, scheduled_receipts)
        if carry <= 0:
            return 0.0                      # 잉여가 곧 자연 소진 -> 할인할 이유 없음

        best_d, best_net = 0.0, -float("inf")
        for d in self.grid:
            if d > max_discount:
                continue
            dem = base * (1.0 + elasticity * d)
            extra = dem - base
            if extra > excess:              # 필수 버퍼를 깎아먹는 할인율은 후보에서 제외
                continue
            sold = min(inventory, dem)
            revenue = unit_price(product, d) * sold
            leftover = max(0.0, excess - extra)
            holding = leftover * product.holding_cost * carry
            net = revenue - holding
            if net > best_net:
                best_d, best_net = d, net
        return best_d

