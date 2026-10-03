"""Material 모듈 - 발주 정책.

정책 계층을 3단으로 나눈다.

  1. 기준 주문량   : EOQ (주문비 vs 재고유지비 균형)
  2. 재주문점      : s = 리드타임 기간 소요(생산계획 x BOM) + 안전재고(z * σ * sqrt(LT))
  3. 이중 조달     : M2만 해당. 일반(LT8, 25원)으로 기저를 깔고, 긴급(LT2, 50원)으로 예측오차를 메움

이 문제에서 긴급 조달이 특히 중요한 이유
  원두 프리미엄 = (50 - 25)원/개 * 2개/제품 = 50원/제품
  완제품 품절비 = 250원/개·일
  => 품절이 예상되면 긴급 조달은 손실의 1/5 수준. "긴급은 비싸니까 쓰지 말자"는 오판이다.
  단, 긴급 주문에도 주문비 3,000,000원이 그대로 붙으므로 '자주 조금씩'은 금물.
  손익분기 수량 = 3,000,000 / (250 - 50) = 15,000 제품분 이상 구제할 때만 발주.
"""

from __future__ import annotations

import datetime as dt
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import MaterialSpec
from ..state import PurchaseOrder, WorldState


# ---------------------------------------------------------------------------
# 공식
# ---------------------------------------------------------------------------

def eoq(daily_demand: float, order_cost: float, holding_cost_per_unit_day: float) -> float:
    """일 단위 EOQ. H를 '원/개·일'로 쓰면 D도 '개/일'이어야 한다."""
    if daily_demand <= 0 or holding_cost_per_unit_day <= 0:
        return 0.0
    return math.sqrt(2 * daily_demand * order_cost / holding_cost_per_unit_day)


def safety_stock(sigma_daily: float, lead_time_days: int, z: float = 1.65) -> float:
    """σ_LT = σ_일 * sqrt(LT). σ_일은 자재 소요의 '예측오차'이지 소요 계열의 변동이 아니다
    (엔진 `_material_sigma`)."""
    return z * sigma_daily * math.sqrt(max(lead_time_days, 0))


# ---------------------------------------------------------------------------
# 정책 인터페이스
# ---------------------------------------------------------------------------

@dataclass
class OrderDecision:
    material: str
    option: str
    qty: int
    reason: str = ""


class MaterialPolicy(ABC):
    name = "base"

    @abstractmethod
    def decide(
        self,
        date: dt.date,
        state: WorldState,
        spec: MaterialSpec,
        requirements: Dict[dt.date, float],   # 그 자재의 일별 소요 (생산계획 x BOM)
        sigma_daily: float,                   # 일 소요 예측오차 표준편차
    ) -> List[OrderDecision]:
        ...


class SsPolicy(MaterialPolicy):
    """(s, S) 연속 점검 정책. 재고 포지션이 s 아래로 떨어지면 S까지 채운다."""

    name = "sS"

    def __init__(self, z: float = 1.65, cover_days: Optional[int] = None):
        self.z = z
        self.cover_days = cover_days   # None이면 EOQ 사용

    def decide(self, date, state, spec, requirements, sigma_daily) -> List[OrderDecision]:
        opt = spec.option("normal")
        lt = opt.lead_time_days
        horizon = [date + dt.timedelta(days=i) for i in range(1, lt + 1)]
        d_lt = sum(requirements.get(d, 0.0) for d in horizon)
        daily = d_lt / max(lt, 1)

        ss = safety_stock(sigma_daily, lt, self.z)
        s = d_lt + ss
        q = (daily * self.cover_days) if self.cover_days else eoq(daily, spec.order_cost, spec.holding_cost)
        S = s + q

        pos = state.inventory_position(spec.code)
        if pos < s:
            qty = int(round(S - pos))
            if qty > 0:
                return [OrderDecision(spec.code, "normal", qty,
                                      f"IP {pos:,} < s {s:,.0f} (LT{lt}일 소요 {d_lt:,.0f} + SS {ss:,.0f})")]
        return []


class DualSourcingPolicy(MaterialPolicy):
    """M2 전용. 일반 경로로 기저 보충 + 긴급 경로로 결품 방지.

    긴급 발동 조건 (둘 다 만족할 때만)
      1) 긴급 리드타임(2일) 내에 자재가 바닥날 것으로 예상됨
      2) 구제되는 제품 수량 * (품절비 - 긴급프리미엄) > 주문비
    """

    name = "dual"

    def __init__(self, z_normal: float = 1.65, z_urgent: float = 2.33,
                 stockout_cost: int = 250, bom_qty: int = 2):
        self.base = SsPolicy(z=z_normal)
        self.z_urgent = z_urgent
        self.stockout_cost = stockout_cost
        self.bom_qty = bom_qty

    def decide(self, date, state, spec, requirements, sigma_daily) -> List[OrderDecision]:
        decisions = list(self.base.decide(date, state, spec, requirements, sigma_daily))

        urgent = spec.option("urgent")
        normal = spec.option("normal")
        lt_u = urgent.lead_time_days
        window = [date + dt.timedelta(days=i) for i in range(1, lt_u + 1)]
        need = sum(requirements.get(d, 0.0) for d in window)
        ss_u = safety_stock(sigma_daily, lt_u, self.z_urgent)

        available = state.on_hand(spec.code) + state.in_transit(spec.code, before=window[-1])
        gap = (need + ss_u) - available
        if gap <= 0:
            return decisions

        premium = urgent.unit_cost - normal.unit_cost
        saved_units = gap / max(self.bom_qty, 1)                     # 구제되는 제품 개수
        benefit = saved_units * (self.stockout_cost - premium * self.bom_qty)
        if benefit > spec.order_cost:
            decisions.append(OrderDecision(
                spec.code, "urgent", int(round(gap)),
                f"긴급: 2일내 부족 {gap:,.0f} / 기대이익 {benefit:,.0f} > 주문비 {spec.order_cost:,}"))
        return decisions


def to_purchase_orders(decisions: List[OrderDecision], date: dt.date,
                       specs: Dict[str, MaterialSpec]) -> List[PurchaseOrder]:
    orders = []
    for d in decisions:
        opt = specs[d.material].option(d.option)
        orders.append(PurchaseOrder(
            material=d.material,
            option=d.option,
            qty=d.qty,
            order_date=date,
            arrival_date=date + dt.timedelta(days=opt.lead_time_days),
        ))
    return orders
