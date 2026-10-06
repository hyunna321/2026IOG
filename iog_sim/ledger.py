"""Ledger - 비용 집계와 KPI.

IOG Ledger 화면 대응 (확정 규칙)
  Total Revenue = 판매수익
  Total Expense = 생산비 + 재고비 + 자재비
      생산비 = 작업준비비 + 인건비(정규+잔업)
      재고비 = 완제품 재고유지비 + 자재 재고유지비 + 재고고갈비(품절)
      자재비 = 자재 주문비 + 자재 구매비
  Balance       = Revenue - Expense          <- 최종 목적함수

  재고고갈비를 '재고비'로 분류한 것은 3개 버킷이 전 비용을 덮어야 하기 때문이다.
  Balance 총액은 분류와 무관하게 동일하고, 달라지는 것은 부문 랭킹 해석뿐이다.

랭킹 부문 대응
  수요예측 : MAE / MAPE
  생산관리 : 작업준비비 + 인건비
  조달계획 : 주문비 + 구매비 + 자재재고유지비 + 재고고갈비
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .state import DailyRecord, MaterialRecord


@dataclass
class LedgerSummary:
    revenue: float = 0.0
    setup_cost: float = 0.0
    labor_cost: float = 0.0
    fg_holding_cost: float = 0.0
    stockout_cost: float = 0.0
    material_order_cost: float = 0.0
    material_purchase_cost: float = 0.0
    material_holding_cost: float = 0.0
    overtime_hours: int = 0
    units_sold: int = 0
    units_lost: int = 0
    mae: Dict[str, float] = field(default_factory=dict)
    mape: Dict[str, float] = field(default_factory=dict)
    fill_rate: Dict[str, float] = field(default_factory=dict)

    # --- Ledger 화면의 3개 버킷 -----------------------------------------
    @property
    def production_cost(self) -> float:
        """생산비 = 작업준비비 + 인건비."""
        return self.setup_cost + self.labor_cost

    @property
    def inventory_cost(self) -> float:
        """재고비 = 완제품 재고유지비 + 자재 재고유지비 + 재고고갈비."""
        return self.fg_holding_cost + self.material_holding_cost + self.stockout_cost

    @property
    def material_cost(self) -> float:
        """자재비 = 주문비 + 구매비."""
        return self.material_order_cost + self.material_purchase_cost

    # --- 부문 랭킹용 -----------------------------------------------------
    @property
    def procurement_cost(self) -> float:
        """조달계획 랭킹 = 발주 + 구매 + 자재재고유지 + 재고고갈."""
        return (self.material_order_cost + self.material_purchase_cost
                + self.material_holding_cost + self.stockout_cost)

    @property
    def total_expense(self) -> float:
        return self.production_cost + self.inventory_cost + self.material_cost

    @property
    def balance(self) -> float:
        return self.revenue - self.total_expense

    def as_row(self) -> Dict[str, float]:
        row = {
            "balance": self.balance,
            "revenue": self.revenue,
            "total_expense": self.total_expense,
            "production_cost": self.production_cost,
            "inventory_cost": self.inventory_cost,
            "material_cost": self.material_cost,
            "setup_cost": self.setup_cost,
            "labor_cost": self.labor_cost,
            "overtime_hours": self.overtime_hours,
            "fg_holding_cost": self.fg_holding_cost,
            "stockout_cost": self.stockout_cost,
            "material_order_cost": self.material_order_cost,
            "material_purchase_cost": self.material_purchase_cost,
            "material_holding_cost": self.material_holding_cost,
            "procurement_cost": self.procurement_cost,
            "units_sold": self.units_sold,
            "units_lost": self.units_lost,
        }
        for p, v in self.mae.items():
            row[f"mae_{p}"] = v
        for p, v in self.mape.items():
            row[f"mape_{p}"] = v
        for p, v in self.fill_rate.items():
            row[f"fill_rate_{p}"] = v
        return row


def summarize(daily: List[DailyRecord], materials: List[MaterialRecord]) -> LedgerSummary:
    s = LedgerSummary()
    by_product: Dict[str, List[DailyRecord]] = {}
    for r in daily:
        s.revenue += r.revenue
        s.setup_cost += r.setup_cost
        s.labor_cost += r.labor_cost
        s.fg_holding_cost += r.fg_holding_cost
        s.stockout_cost += r.stockout_cost
        s.overtime_hours += r.overtime_hours
        s.units_sold += r.sold
        s.units_lost += r.lost_sales
        by_product.setdefault(r.product, []).append(r)

    for r in materials:
        s.material_order_cost += r.order_cost
        s.material_purchase_cost += r.purchase_cost
        s.material_holding_cost += r.holding_cost

    for p, recs in by_product.items():
        active = [r for r in recs if r.actual_demand > 0]
        if active:
            err = np.array([abs(r.forecast_demand - r.actual_demand) for r in active], dtype=float)
            act = np.array([r.actual_demand for r in active], dtype=float)
            s.mae[p] = float(err.mean())
            s.mape[p] = float((err / act).mean() * 100)
        dem = sum(r.actual_demand for r in recs)
        s.fill_rate[p] = (sum(r.sold for r in recs) / dem) if dem else 1.0
    return s
