"""IOG 게임 규칙 상수 정의.

출처: IOG_문제소개.pdf / IOG_매뉴얼.pdf
여기 있는 값은 '문서에서 읽은 값'이므로, 라운드 시작 시 반드시 시스템 화면에서
재확인한 뒤 VERIFY 주석이 붙은 항목을 갱신할 것.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

# ---------------------------------------------------------------------------
# 생산 인건비 환산 규칙 (매뉴얼 Ledger 예시 기준)
#   Makespan(시간) -> floor(Makespan / 100) = 인건비 청구 시간
#   청구 시간 중 정규시간분은 100,000원/시간, 초과분은 200,000원/시간
# ---------------------------------------------------------------------------
MAKESPAN_PER_LABOR_HOUR = 100          # Makespan 100시간 = 인건비 1시간
REGULAR_WAGE_PER_HOUR = 100_000        # 원/시간
OVERTIME_WAGE_PER_HOUR = 200_000       # 원/시간
LOT_SIZE = 1_000                       # 1 Lot = 1 Job = 제품 1,000개
N_MACHINES = 20                        # 20-stage permutation flow shop
# 하루 한 제품의 최대 Job 수. ProcessingTimeTable이 t_{500}_20_{요일}.csv 즉 500 Job까지만
# 제공하므로 그 이상은 처리시간 자체를 알 수 없다. 이 상한을 넘겨 계획하면 makespan이
# 500 Job 기준으로만 계산되어 인건비가 크게 과소평가된다. (VERIFY: 시스템 입력 상한 확인)
MAX_LOTS_PER_DAY = 500


@dataclass(frozen=True)
class ProductSpec:
    code: str
    name: str
    channel: str
    price: int                  # 판매단가 (원/개)
    holding_cost: int           # 완제품 재고유지비 (원/개·일)
    stockout_cost: int          # 판매기회비 (원/개·일)
    setup_cost: int             # 작업준비비 (원/배치)
    regular_makespan: int       # 정규시간 기준 Makespan (시간)
    market_days_only: bool      # True면 주식시장 개장일에만 수요 발생
    # True면 생산도 평일에만 가능. Production 화면에서 P1은 토·일 입력칸이 아예 없고(0 고정),
    # P2는 7일 모두 입력 가능하다 -> 생산가능일은 제품마다 다르다.
    weekday_production_only: bool

    @property
    def regular_labor_hours(self) -> int:
        return self.regular_makespan // MAKESPAN_PER_LABOR_HOUR


@dataclass(frozen=True)
class SourcingOption:
    """자재 조달 경로 (일반 / 긴급)."""

    name: str
    unit_cost: int              # 구매비 (원/개)
    lead_time_days: int         # 리드타임 (달력일; 주말·공휴일 포함)


@dataclass(frozen=True)
class MaterialSpec:
    code: str
    name: str
    initial_inventory: int      # 라운드 기초재고 (VERIFY: 라운드마다 바뀜)
    order_cost: int             # 주문비 (원/회)
    holding_cost: int           # 재고유지비 (원/개·일)
    options: Mapping[str, SourcingOption]

    def option(self, name: str = "normal") -> SourcingOption:
        return self.options[name]


PRODUCTS: Dict[str, ProductSpec] = {
    "P1": ProductSpec(
        code="P1",
        name="아메리카노",
        channel="할인마트",
        price=250,
        holding_cost=30,
        stockout_cost=250,
        setup_cost=5_000_000,
        regular_makespan=7_000,
        market_days_only=True,
        weekday_production_only=True,
    ),
    "P2": ProductSpec(
        code="P2",
        name="카푸치노",
        channel="편의점",
        price=350,
        holding_cost=30,
        stockout_cost=250,
        setup_cost=2_000_000,
        regular_makespan=7_000,
        market_days_only=False,
        weekday_production_only=False,     # 토·일에도 생산 가능
    ),
}

MATERIALS: Dict[str, MaterialSpec] = {
    "M1": MaterialSpec(
        code="M1",
        name="캔용기",
        initial_inventory=300_000,
        order_cost=1_000_000,
        holding_cost=1,
        options={"normal": SourcingOption("normal", 30, 3)},
    ),
    "M2": MaterialSpec(
        code="M2",
        name="원두",
        initial_inventory=3_000_000,
        order_cost=3_000_000,
        holding_cost=1,
        options={
            "normal": SourcingOption("normal", 25, 8),
            "urgent": SourcingOption("urgent", 50, 2),
        },
    ),
    "M3": MaterialSpec(
        code="M3",
        name="병용기",
        initial_inventory=500_000,
        order_cost=2_000_000,
        holding_cost=2,
        options={"normal": SourcingOption("normal", 50, 5)},
    ),
}

# BOM: 제품 1개당 자재 소요량
# P2는 실측 확인됨: 2026-09-27 P2 85 Job(=85,000개) 생산에 M2 170,000 / M3 85,000이 빠졌다.
# VERIFY: P1({M1:1, M2:2})은 아직 문제소개 PDF 도식 판독값이다. P1 첫 생산 후 Material
#         inventory의 out qty로 반드시 재확인할 것.
BOM: Dict[str, Dict[str, int]] = {
    "P1": {"M1": 1, "M2": 2},
    "P2": {"M2": 2, "M3": 1},
}


@dataclass
class SimConfig:
    """시뮬레이션 실행 옵션 (게임 규칙이 아니라 '가정'에 해당하는 것들)."""

    # 당일 생산분 당일 판매 **가능** (2026-09-28 사용자 확정).
    # 근거: 09-27에 P2 85 Job(85,000개)을 생산한 그날 84,856개(=09-27 실수요)가 팔려
    # Current level이 144만 남았다. 따라서 엔진은 '생산 -> 판매' 순서로 하루를 돈다.
    # False로 되돌리면 '판매 -> 생산' 순서가 되고 availability_lag가 1로 올라간다.
    produce_then_sell_same_day: bool = True
    # 할인 수요 탄력성: 수요배수 = 1 + elasticity * 할인율
    #   문제소개 예시(10% 할인 -> 수요 100->110)는 elasticity = 1.0
    #   Sales 화면 관측치(6% 증가)와 다르므로 실측 로그로 재추정할 것
    discount_elasticity: float = 1.0
    max_discount: float = 0.30
    products: Dict[str, ProductSpec] = field(default_factory=lambda: dict(PRODUCTS))
    materials: Dict[str, MaterialSpec] = field(default_factory=lambda: dict(MATERIALS))
    bom: Dict[str, Dict[str, int]] = field(default_factory=lambda: {k: dict(v) for k, v in BOM.items()})
