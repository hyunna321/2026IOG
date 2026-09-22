"""비용 함수. IOG 규칙을 그대로 코드화한 계층 — 여기서 틀리면 모든 실험이 무의미하다.

검증 기준(매뉴얼 원문 예제):
  Makespan 6,680 -> 66시간 -> 6,600,000원
  Makespan 8,850 -> 88시간 -> 70*100,000 + 18*200,000 = 10,600,000원
  Makespan 9,466 -> 94시간 -> 70*100,000 + 24*200,000 = 11,800,000원
  Makespan 9,363 -> 93시간 -> 93*100,000 + 23*100,000 = 11,600,000원
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    MAKESPAN_PER_LABOR_HOUR,
    OVERTIME_WAGE_PER_HOUR,
    REGULAR_WAGE_PER_HOUR,
    ProductSpec,
)


def labor_hours(makespan: float) -> int:
    """Makespan(시간) -> 인건비 청구 시간. 소숫점 버림."""
    return int(makespan // MAKESPAN_PER_LABOR_HOUR)


def labor_cost(makespan: float, product: ProductSpec) -> int:
    """정규/잔업 구분 인건비."""
    hours = labor_hours(makespan)
    regular = min(hours, product.regular_labor_hours)
    overtime = max(0, hours - product.regular_labor_hours)
    return regular * REGULAR_WAGE_PER_HOUR + overtime * OVERTIME_WAGE_PER_HOUR


def overtime_hours(makespan: float, product: ProductSpec) -> int:
    return max(0, labor_hours(makespan) - product.regular_labor_hours)


@dataclass(frozen=True)
class BatchCost:
    setup: int
    labor: int
    overtime_hours: int
    makespan: float

    @property
    def total(self) -> int:
        return self.setup + self.labor


def batch_production_cost(makespan: float, product: ProductSpec, n_lots: int) -> BatchCost:
    """생산비 = 작업준비비(배치 1회 고정) + 인건비(Makespan 환산).

    n_lots == 0 이면 배치를 열지 않은 것이므로 비용 0.
    """
    if n_lots <= 0:
        return BatchCost(setup=0, labor=0, overtime_hours=0, makespan=0.0)
    return BatchCost(
        setup=product.setup_cost,
        labor=labor_cost(makespan, product),
        overtime_hours=overtime_hours(makespan, product),
        makespan=makespan,
    )


# --- 한계 비용: lot sizing / 할인 정책의 의사결정 근거 -----------------------

def marginal_holding_cost(qty: int, days: int, unit_holding: int) -> int:
    """qty개를 days일 보유할 때의 재고유지비."""
    return qty * days * unit_holding


def breakeven_lots_for_setup(product: ProductSpec, carry_days: int = 1) -> float:
    """작업준비비 1회를 아끼려고 며칠치를 미리 만들 때, 재고비로 상쇄되는 Lot 수.

    예) P1: setup 5,000,000 / (1,000개 * 1일 * 30원) = 166.7 Lot·일
        즉 하루 앞서 만드는 물량이 166 Lot(=166,000개)을 넘으면 배치 통합이 손해.
    """
    per_lot_day = 1_000 * product.holding_cost
    return product.setup_cost / (per_lot_day * max(carry_days, 1))


def urgent_sourcing_premium_vs_stockout(
    unit_premium: int, bom_qty: int, stockout_cost: int
) -> float:
    """긴급 조달 프리미엄 대비 품절비 비율.

    예) M2 원두: 프리미엄 25원/개 * 2개/제품 = 50원 vs 품절비 250원/개·일
        -> 비율 0.2. 품절이 예상되면 긴급 조달이 압도적으로 유리하다.
    """
    return (unit_premium * bom_qty) / stockout_cost
