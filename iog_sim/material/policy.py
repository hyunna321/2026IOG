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

    def __init__(self, z: float = 1.65, cover_days: Optional[int] = None, lot_scale: float = 1.0):
        self.z = z
        self.cover_days = cover_days   # None이면 EOQ 사용
        # 발주 크기 배수 (EOQ x lot_scale). M1은 주말에 안 쓰여도 매일 보관비가 붙고 M3는 보관비가 2배라
        # EOQ보다 작게 받는 편이 낫다 (report_draft/material_lot_recheck.py: 0.75배 31라운드 전승 +10.2M).
        self.lot_scale = lot_scale

    def decide(self, date, state, spec, requirements, sigma_daily) -> List[OrderDecision]:
        opt = spec.option("normal")
        lt = opt.lead_time_days
        horizon = [date + dt.timedelta(days=i) for i in range(1, lt + 1)]
        d_lt = sum(requirements.get(d, 0.0) for d in horizon)
        daily = d_lt / max(lt, 1)

        ss = safety_stock(sigma_daily, lt, self.z)
        s = d_lt + ss
        q = (daily * self.cover_days) if self.cover_days else eoq(daily, spec.order_cost, spec.holding_cost)
        S = s + q * self.lot_scale

        pos = state.inventory_position(spec.code)
        if pos < s:
            qty = int(round(S - pos))
            if qty > 0:
                return [OrderDecision(spec.code, "normal", qty,
                                      f"IP {pos:,} < s {s:,.0f} (LT{lt}일 소요 {d_lt:,.0f} + SS {ss:,.0f})")]
        return []


def locked_buffer(day: dt.date, planned_through: Optional[dt.date], ss: float) -> float:
    """그날 지켜야 할 자재 버퍼. 확정 생산계획 안의 날은 소요가 '우리가 제출한 계획 x BOM'으로
    정해져 예측오차가 없으므로 0, 그 뒤(직전 주 패턴으로 추정한 날)만 안전재고를 둔다."""
    return 0.0 if planned_through is not None and day <= planned_through else ss


def _ww_first_lot(days: List[dt.date], net: Dict[dt.date, float],
                  order_cost: float, holding: float) -> float:
    """Wagner-Whitin: days[0]에 도착하는 첫 주문이 덮어야 할 양.

    일별 순소요 net을 그대로 풀어 '주문비 vs 재고유지비'를 맞춘다. EOQ처럼 일 평균을 쓰지 않으므로
    주말(소요 0)·금요일 큰 배치 같은 주간 패턴에 맞춰 덮는 구간이 끝난다.
    보관비는 장부와 같이 도착일 ~ 사용 전날 마감마다 1일씩 붙는다.
    """
    n = len(days)
    need = [net.get(d, 0.0) for d in days]
    first = next((k for k, q in enumerate(need) if q > 0), None)
    if first is None:
        return 0.0
    best = [0.0] * (n + 1)          # best[i] = days[i:]를 덮는 최소비용 (days[i]에 주문 도착)
    nxt = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        hold, best[i] = 0.0, float("inf")
        # 첫 주문(i=0)은 slack으로 일찍 도착해도 첫 순소요일까지는 반드시 덮는다
        lo = first + 1 if i == 0 else i + 1
        for j in range(i + 1, n + 1):            # days[i] 주문이 days[i:j]를 덮는다
            hold += holding * (days[j - 1] - days[i]).days * need[j - 1]
            # days[j]에 다음 주문이 도착 -> 그날 순소요가 있어야 의미가 있다 (끝이면 그대로)
            if j < lo or (j < n and need[j] <= 0):
                continue
            cost = order_cost + hold + best[j]
            if cost < best[i]:
                best[i], nxt[i] = cost, j
    return sum(need[:nxt[0]])


class WeeklyAlignedPolicy(MaterialPolicy):
    """확정 구간 안전재고 0 + 일별 소요 기반 Wagner-Whitin 발주 (주간 주기 정렬).

    (s, S)+EOQ와 다른 점
      1) 안전재고는 확정 계획 밖(추정 소요) 날에만 둔다. 확정 주의 자재 소요는 우리가 제출한
         생산계획 x BOM이라 예측오차가 없다. 추정 구간의 버퍼 크기(z·σ·√LT)는 기존과 같다.
      2) 발주 시점: 오늘 발주가 닿는 날(오늘+LT)부터 날짜별로 굴린 예상 재고가 버퍼 아래로
         내려가는 첫날이 그 도착일(+slack_days) 안이면 오늘 발주한다. 재고포지션 합계가 아니라
         날짜별 입고예정을 보므로, 늦게 도착하는 주문을 오늘 쓸 수 있는 재고로 착각하지 않는다.
      3) 발주량: EOQ(일 평균) 대신 그 뒤 일별 순소요를 Wagner-Whitin으로 풀어 첫 주문분만 낸다.
         주말에 쓰지 않는 자재를 넘기는 비용과 주문비를 날짜 단위로 비교하므로 주 경계에서 끊긴다.

    오늘+LT 전에 생기는 부족은 일반 경로로 막을 수 없다(그 구간은 생산이 잘려 소모되지 않는다고 보고
    0에서 멈춘다). M2는 그 구간을 DualSourcingPolicy의 긴급 판단이 본다.
    """

    name = "weekly"

    def __init__(self, z: float = 1.65, slack_days: int = 0, plan_error: float = 0.0,
                 round_end_plan_error: Optional[float] = None, trigger_frac: float = 1.0):
        self.z = z
        self.slack_days = slack_days
        # 발주 발동 기준: 예상 재고가 버퍼 x trigger_frac 아래로 내려갈 때 (발주량은 버퍼 전체를 다시 채움).
        # 1.0이면 버퍼를 조금만 파고들어도 발주해, 가계획이 조금 커질 때마다 주문비(M2 3M)를 내고
        # 몇만 개짜리 보충 주문을 낸다. 확정 구간(버퍼 0)의 실제 부족은 이 값과 관계없이 항상 발동한다.
        self.trigger_frac = trigger_frac
        # 라운드 종료 직전 발주를 '남은 소요'로 자를 때 남길 여유 비율 (None = plan_error와 같음).
        # 종료 뒤 남는 자재는 구매비 전액 손실이라 평소 버퍼보다 작게 둘 수 있다.
        self.round_end_plan_error = plan_error if round_end_plan_error is None else round_end_plan_error
        # 계획 수정 버퍼: 확정 전 날의 누적 소요 x plan_error를 추가로 둔다. 가계획(또는 복사 추정)과
        # 토요일 확정 계획의 누적 자재 소요 차이는 한 주 기준 p95 약 10%, p99 약 15~20%였다
        # (예측 갱신 + 안전재고 증분 + DP 배치 이동). z·σ·√LT(1-step 예측오차)만으로는 이걸 못 덮는다.
        self.plan_error = plan_error

    def decide(self, date, state, spec, requirements, sigma_daily) -> List[OrderDecision]:
        lt = spec.option("normal").lead_time_days
        arrive = date + dt.timedelta(days=lt)
        days = sorted(d for d in requirements if d > date)
        if not days or arrive > days[-1]:
            return []
        ss = safety_stock(sigma_daily, lt, self.z)
        pt = state.planned_through

        arrivals: Dict[dt.date, float] = {}
        for o in state.open_orders:
            if o.material == spec.code and not o.received and o.arrival_date > date:
                arrivals[o.arrival_date] = arrivals.get(o.arrival_date, 0.0) + o.qty

        inv = float(state.on_hand(spec.code))
        covered = 0.0                     # 지금까지 순소요로 잡은 누적 추가 공급량
        unlocked = 0.0                    # 확정 계획 밖 날의 누적 소요
        net: Dict[dt.date, float] = {}
        buffers: Dict[dt.date, float] = {}
        for day in days:
            inv += arrivals.get(day, 0.0) - requirements.get(day, 0.0)
            if pt is None or day > pt:
                unlocked += requirements.get(day, 0.0)
            if day < arrive:
                inv = max(inv, 0.0)
                continue
            buffers[day] = locked_buffer(day, pt, ss + self.plan_error * unlocked)
            short = buffers[day] - inv
            if short > covered:
                net[day] = short - covered
                covered = short
        if not net:
            return []
        # 발동일: 예상 재고(이미 낸 주문 포함, 오늘 주문 없음)가 버퍼 x trigger_frac 아래로 내려가는 첫날
        trigger = None
        level = float(state.on_hand(spec.code))
        for day in days:
            level += arrivals.get(day, 0.0) - requirements.get(day, 0.0)
            if day < arrive:
                level = max(level, 0.0)
                continue
            if level < buffers[day] * self.trigger_frac:      # 버퍼 0인 확정일은 level < 0 = 실제 부족
                trigger = day
                break
        if trigger is None or trigger > arrive + dt.timedelta(days=self.slack_days):
            return []                      # 아직 버퍼 안쪽이거나, 내일 이후에 발주해도 늦지 않다
        first = min(net)

        window = [d for d in days if d >= arrive]
        qty = int(math.ceil(_ww_first_lot(window, net, spec.order_cost, spec.holding_cost)))
        if qty <= 0:
            return []
        locked = "확정" if pt is not None and first <= pt else "추정"
        return [OrderDecision(spec.code, "normal", qty,
                              f"{first:%m-%d}({locked}) 버퍼 {buffers[first]:,.0f} 미달 "
                              f"-> {arrive:%m-%d} 도착, WW 덮는 양 {qty:,}")]


class DualSourcingPolicy(MaterialPolicy):
    """M2 전용. 일반 경로로 기저 보충 + 긴급 경로로 결품 방지.

    긴급 발동 조건 (둘 다 만족할 때만)
      1) 긴급발주가 닿는 첫날(오늘 + 긴급 LT)부터 slack_days일 안에 예상 재고가 안전재고 아래로 떨어짐
      2) 구제되는 제품 수량 * (품절비 - 긴급프리미엄) > 주문비

    예상 재고는 날짜별로 굴린다: 현재고 + 그날 도착 예정 - 그날 소요. 긴급으로도 닿지 않는 날
    (오늘 ~ 긴급 도착 전날)의 부족분은 생산이 잘려 소모되지 않으므로 0에서 멈추고, 긴급발주
    수량에도 넣지 않는다. slack_days만큼 미리 보는 것은 '마지막 가능일'에야 발주해 하루만 놓쳐도
    되돌릴 수 없게 되는 것을 막기 위해서다 (2026-10-14 M2 긴급 사례).
    """

    name = "dual"

    def __init__(self, z_normal: float = 1.65, z_urgent: float = 2.33,
                 stockout_cost: int = 250, bom_qty: int = 2, slack_days: int = 1,
                 cover_days: Optional[int] = None, base: Optional[MaterialPolicy] = None,
                 lock_aware: bool = False):
        # base: 일반 경로 정책. 없으면 기존 (s,S)+EOQ.
        # lock_aware: 긴급 판단 버퍼도 확정 계획 안의 날은 0으로 본다. WeeklyAlignedPolicy는 확정 주에
        #   버퍼를 두지 않으므로, 이걸 끄면 확정 주 재고가 0 근처일 때마다 쓸데없이 긴급이 나간다.
        self.base = base or SsPolicy(z=z_normal, cover_days=cover_days)
        self.lock_aware = lock_aware
        self.z_urgent = z_urgent
        self.stockout_cost = stockout_cost
        self.bom_qty = bom_qty
        self.slack_days = slack_days

    def decide(self, date, state, spec, requirements, sigma_daily) -> List[OrderDecision]:
        decisions = list(self.base.decide(date, state, spec, requirements, sigma_daily))

        urgent = spec.option("urgent")
        normal = spec.option("normal")
        lt_u = urgent.lead_time_days
        first = date + dt.timedelta(days=lt_u)                 # 오늘 긴급발주가 처음 닿는 날
        last = first + dt.timedelta(days=self.slack_days)
        ss_u = safety_stock(sigma_daily, lt_u, self.z_urgent)

        arrivals: Dict[dt.date, float] = {}
        for o in state.open_orders:
            if o.material == spec.code and not o.received and date < o.arrival_date <= last:
                arrivals[o.arrival_date] = arrivals.get(o.arrival_date, 0.0) + o.qty
        inv = float(state.on_hand(spec.code))
        gap = 0.0                          # 긴급 도달 구간에서 버퍼 대비 가장 큰 부족
        day = date + dt.timedelta(days=1)
        while day <= last:
            inv += arrivals.get(day, 0.0) - requirements.get(day, 0.0)
            if day < first:
                inv = max(inv, 0.0)        # 긴급으로도 못 막는 부족: 생산이 잘려 소모되지 않는다
            else:
                buf = locked_buffer(day, state.planned_through, ss_u) if self.lock_aware else ss_u
                gap = max(gap, buf - inv)
            day += dt.timedelta(days=1)
        if gap <= 0:
            return decisions

        premium = urgent.unit_cost - normal.unit_cost
        saved_units = gap / max(self.bom_qty, 1)                     # 구제되는 제품 개수
        benefit = saved_units * (self.stockout_cost - premium * self.bom_qty)
        if benefit > spec.order_cost:
            decisions.append(OrderDecision(
                spec.code, "urgent", int(round(gap)),
                f"긴급: {first:%m-%d}~{last:%m-%d} 부족 {gap:,.0f} / 기대이익 {benefit:,.0f} > 주문비 {spec.order_cost:,}"))
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
