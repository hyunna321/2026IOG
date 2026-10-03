"""시뮬레이션 상태 객체.

WorldState는 '어느 시점의 공장 전체 스냅샷'이다. 정책 함수는 이 객체를 읽기만 하고,
상태 변경은 engine.py의 이벤트 핸들러만 수행한다 (변경 지점을 한 곳으로 몰아 재현성 확보).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PurchaseOrder:
    material: str
    option: str             # "normal" | "urgent"
    qty: int
    order_date: dt.date
    arrival_date: dt.date
    received: bool = False


@dataclass
class ProductionOrder:
    """하루 단위 생산 배치 계획."""

    product: str
    date: dt.date
    n_lots: int
    job_sequence: Optional[List[int]] = None   # 1-based Job ID 순열


@dataclass
class DailyRecord:
    date: dt.date
    product: str
    forecast_demand: int = 0
    actual_demand: int = 0
    effective_demand: int = 0      # 할인 반영 후 수요
    discount: float = 0.0
    sold: int = 0
    lost_sales: int = 0
    produced: int = 0
    planned_lots: int = 0
    makespan: float = 0.0
    overtime_hours: int = 0
    revenue: int = 0
    setup_cost: int = 0
    labor_cost: int = 0
    fg_holding_cost: int = 0
    stockout_cost: int = 0
    fg_inventory_end: int = 0


@dataclass
class MaterialRecord:
    date: dt.date
    material: str
    received: int = 0
    consumed: int = 0
    inventory_end: int = 0
    order_cost: int = 0
    purchase_cost: int = 0
    holding_cost: int = 0


@dataclass
class WorldState:
    fg_inventory: Dict[str, int] = field(default_factory=dict)      # 완제품 재고
    mat_inventory: Dict[str, int] = field(default_factory=dict)     # 자재 재고
    open_orders: List[PurchaseOrder] = field(default_factory=list)
    production_plan: Dict[dt.date, Dict[str, ProductionOrder]] = field(default_factory=dict)
    discount_plan: Dict[dt.date, Dict[str, float]] = field(default_factory=dict)
    forecast_cache: Dict[str, Dict[dt.date, int]] = field(default_factory=dict)
    # 제품별 1-step 예측오차 표준편차 (자재 안전재고의 근거). 주간계획 때 갱신.
    forecast_sigma: Dict[str, float] = field(default_factory=dict)
    # 생산계획이 확정된 마지막 날. 이후 날짜의 자재 소요는 직전 주 패턴으로 추정한다.
    planned_through: Optional[dt.date] = None
    daily_records: List[DailyRecord] = field(default_factory=list)
    material_records: List[MaterialRecord] = field(default_factory=list)

    # --- 조회 헬퍼 (정책 함수가 쓰는 read-only API) -----------------------
    def on_hand(self, material: str) -> int:
        return self.mat_inventory.get(material, 0)

    def in_transit(self, material: str, before: Optional[dt.date] = None) -> int:
        """미입고 발주 잔량. before가 주어지면 그 날짜까지 도착하는 물량만."""
        return sum(
            o.qty
            for o in self.open_orders
            if o.material == material
            and not o.received
            and (before is None or o.arrival_date <= before)
        )

    def inventory_position(self, material: str, horizon_end: Optional[dt.date] = None) -> int:
        """재고 포지션 = 현재고 + 입고예정. 재주문점 비교의 기준값."""
        return self.on_hand(material) + self.in_transit(material, horizon_end)

    def planned_lots(self, date: dt.date, product: str) -> int:
        order = self.production_plan.get(date, {}).get(product)
        return order.n_lots if order else 0

    def discount_for(self, date: dt.date, product: str) -> float:
        return self.discount_plan.get(date, {}).get(product, 0.0)
