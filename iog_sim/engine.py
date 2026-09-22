"""시뮬레이션 엔진 - 일별 이벤트 루프.

하루 안에서의 이벤트 순서가 결과를 좌우한다. IOG 규칙에 맞춘 순서는 다음과 같다.

    1) 자재 입고        발주일 + LT == 오늘 -> 입고. "입고 당일 생산 투입 가능"
    2) 생산 실행        평일만. 자재 차감 -> 완제품 입고. Makespan -> 생산비
    3) 수요 실현·판매   할인 반영 수요 -> 판매 = min(수요, 재고) -> 매출 / 미충족분 -> 판매기회비
    4) 익일 할인 결정   Sales 화면 "다음날 할인율 결정"
    5) 자재 발주 결정   (s,S) 연속점검. 발주일 기록 -> 미래 입고 예약
    6) 일 마감          완제품·자재 재고유지비 부과, 기록 적재
    7) (금요일) 차주 계획  수요예측 -> Lot sizing -> Job sequencing -> 생산계획 확정

상태 변경은 이 파일에서만 일어난다. 정책 객체는 상태를 읽고 '결정'만 반환한다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .calendar import GameCalendar
from .config import LOT_SIZE, SimConfig
from .costs import batch_production_cost
from .demand.forecaster import ForecastResult, Forecaster
from .demand.generator import DemandGenerator
from .ledger import LedgerSummary, summarize
from .material.policy import MaterialPolicy, to_purchase_orders
from .production.lotsizing import LotSizingPolicy, safety_stock_units
from .production.sequencing import Sequencer, SequenceResult
from .sales.discount import DiscountPolicy, effective_demand, unit_price
from .state import DailyRecord, MaterialRecord, ProductionOrder, WorldState

# (제품, 생산일, Lot수) -> 처리시간 행렬
ProcessingTimeProvider = Callable[[str, dt.date, int], np.ndarray]


@dataclass
class PolicySet:
    """what-if 실험에서 갈아끼우는 단위. 이 객체 하나가 곧 '하나의 전략'이다."""

    forecaster: Dict[str, Forecaster]
    lot_sizing: LotSizingPolicy
    sequencer: Sequencer                      # 실제 투입 순서 결정 (마감 전 시간 여유 사용)
    material: Dict[str, MaterialPolicy]
    discount: DiscountPolicy
    service_level: float = 0.95
    forecast_bias: float = 1.0
    label: str = "policy"
    # Lot sizing DP가 후보 배치마다 호출하는 '싼' Makespan 추정기.
    # 정확한 메타휴리스틱을 여기에 꽂으면 계획 1회에 수백 번 호출되어 감당이 안 된다.
    planning_sequencer: Optional[Sequencer] = None


@dataclass
class SimResult:
    summary: LedgerSummary
    state: WorldState
    label: str = ""
    seed: int = 0
    warmup_days: int = 0


class Simulator:
    def __init__(
        self,
        cfg: SimConfig,
        calendar: GameCalendar,
        demand: DemandGenerator,
        policies: PolicySet,
        processing_times: ProcessingTimeProvider,
    ):
        self.cfg = cfg
        self.cal = calendar
        self.demand = demand
        self.pol = policies
        self.pt = processing_times
        self._ms_cache: Dict[tuple, float] = {}
        self._seq_cache: Dict[tuple, SequenceResult] = {}
        if policies.planning_sequencer is None:
            from .production.sequencing import LTWK
            policies.planning_sequencer = LTWK()

    # ------------------------------------------------------------------
    # Makespan 오라클 (Lot sizing DP가 반복 호출 -> 싼 추정기 + 캐시)
    # ------------------------------------------------------------------
    def makespan_for(self, product: str, date: dt.date, n_lots: int) -> float:
        if n_lots <= 0:
            return 0.0
        key = (product, date, n_lots)
        if key not in self._ms_cache:
            p = self.pt(product, date, n_lots)
            self._ms_cache[key] = self.pol.planning_sequencer.solve(p).makespan
        return self._ms_cache[key]

    def sequence_for(self, product: str, date: dt.date, n_lots: int) -> SequenceResult:
        key = (product, date, n_lots)
        if key not in self._seq_cache:
            p = self.pt(product, date, n_lots)
            self._seq_cache[key] = self.pol.sequencer.solve(p)
        return self._seq_cache[key]

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def run(self, start: dt.date, end: dt.date, seed: int = 0,
            warmup_days: int = 0,
            initial_fg_inventory: Optional[Dict[str, int]] = None,
            initial_mat_inventory: Optional[Dict[str, int]] = None,
            backfill_plan_from: Optional[dt.date] = None,
            locked_production_plan: Optional[Dict[dt.date, Dict[str, int]]] = None) -> SimResult:
        """warmup_days: 초기 과도구간(initial transient) 제거.

        시뮬레이션 시작 시점에는 완제품 재고가 0이고 첫 주간계획도 아직 서지 않았으므로,
        첫 1~2주의 품절은 정책의 결과가 아니라 시작 조건의 결과다. 이것을 그대로 집계하면
        모든 정책에 동일한 상수 손실이 얹혀 정책 간 차이가 묻힌다.
        기록은 전부 남기되(state.daily_records), 집계(summary)에서만 제외한다.

        initial_fg_inventory / initial_mat_inventory: 시작 시점 재고를 실측값으로
        덮어쓴다(라운드 중간 상태에서 리포트를 뽑을 때 사용, 기본은 0 / 라운드 기초재고).

        backfill_plan_from: start가 결정일(토요일)이 아닌 라운드 중간이라 아직 차주
        계획이 없을 때, 이 날짜를 결정일로 가정해 한 번 `_plan_next_week`를 미리 돌려
        forecast_cache/production_plan을 채운다 (예: start 하루 전날을 넘기면
        start부터의 한 주치 계획이 채워진다).

        locked_production_plan: {날짜: {제품: Lot수}}. 이미 지난 결정일에 제출되어
        확정된 주간 생산계획. 주어진 날짜는 재최적화 결과를 덮어쓴다(그날 생산이 없으면
        빈 dict를 넣는다). 남은 기간의 자재 발주·할인은 이 확정 계획을 전제로 계산된다.
        """
        self.demand.reset(seed)
        state = WorldState(
            today=start,
            fg_inventory=dict(initial_fg_inventory) if initial_fg_inventory is not None
                         else {p: 0 for p in self.cfg.products},
            mat_inventory=dict(initial_mat_inventory) if initial_mat_inventory is not None
                          else {m: s.initial_inventory for m, s in self.cfg.materials.items()},
        )
        state.demand_history = {p: {} for p in self.cfg.products}
        state.forecast_cache = {p: {} for p in self.cfg.products}
        if backfill_plan_from is not None:
            self._plan_next_week(state, backfill_plan_from)
        if locked_production_plan is not None:
            for day, by_product in locked_production_plan.items():
                state.production_plan[day] = {
                    code: ProductionOrder(product=code, date=day, n_lots=lots)
                    for code, lots in by_product.items() if lots > 0
                }

        for d in self.cal.date_range(start, end):
            state.today = d
            recv = self._receive_materials(state, d)
            if self.cfg.produce_then_sell_same_day:
                produced, prod_costs, consumed = self._produce(state, d)
                sales = self._sell(state, d)
            else:
                # 당일 생산분은 당일 판매 불가 -> 판매를 먼저 처리해 오늘 생산분이
                # 오늘 매출에 잡히지 않게 한다. 기말재고(= 재고유지비 부과 대상)에는 포함된다.
                sales = self._sell(state, d)
                produced, prod_costs, consumed = self._produce(state, d)
            self._decide_discount(state, d)
            order_costs = self._order_materials(state, d)
            self._close_day(state, d, recv, produced, prod_costs, sales, order_costs, consumed)
            if self.cal.is_decision_day(d):
                self._plan_next_week(state, d)

        cutoff = start + dt.timedelta(days=warmup_days)
        daily = [r for r in state.daily_records if r.date >= cutoff]
        mats = [r for r in state.material_records if r.date >= cutoff]
        return SimResult(summary=summarize(daily, mats), state=state,
                         label=self.pol.label, seed=seed, warmup_days=warmup_days)

    # ------------------------------------------------------------------
    # 1) 자재 입고
    # ------------------------------------------------------------------
    def _receive_materials(self, state: WorldState, d: dt.date) -> Dict[str, int]:
        recv: Dict[str, int] = {}
        for o in state.open_orders:
            if not o.received and o.arrival_date == d:
                state.mat_inventory[o.material] = state.mat_inventory.get(o.material, 0) + o.qty
                recv[o.material] = recv.get(o.material, 0) + o.qty
                o.received = True
        return recv

    # ------------------------------------------------------------------
    # 2) 생산
    # ------------------------------------------------------------------
    def _produce(self, state: WorldState, d: dt.date):
        produced: Dict[str, int] = {}
        costs: Dict[str, dict] = {}
        consumed: Dict[str, int] = {}

        for code, order in sorted(state.production_plan.get(d, {}).items()):
            spec = self.cfg.products[code]
            # 생산가능일은 제품마다 다르다 (P1 평일만 / P2 매일).
            if not self.cal.is_production_day(d, spec.weekday_production_only):
                continue
            lots = order.n_lots
            if lots <= 0:
                continue
            # 자재 제약 -> 실제 생산 가능 Lot 수
            feasible = lots
            for mat, per_unit in self.cfg.bom[code].items():
                have = state.mat_inventory.get(mat, 0)
                feasible = min(feasible, have // (per_unit * LOT_SIZE))
            if self.cfg.material_shortage_mode == "fail" and feasible < lots:
                feasible = 0
            feasible = max(0, int(feasible))
            if feasible == 0:
                costs[code] = {"setup": 0, "labor": 0, "ot": 0, "makespan": 0.0}
                continue

            for mat, per_unit in self.cfg.bom[code].items():
                used = feasible * LOT_SIZE * per_unit
                state.mat_inventory[mat] -= used
                consumed[mat] = consumed.get(mat, 0) + used

            res = self.sequence_for(code, d, feasible)
            bc = batch_production_cost(res.makespan, spec, feasible)
            state.fg_inventory[code] = state.fg_inventory.get(code, 0) + feasible * LOT_SIZE
            produced[code] = feasible * LOT_SIZE
            costs[code] = {"setup": bc.setup, "labor": bc.labor,
                           "ot": bc.overtime_hours, "makespan": bc.makespan}
            order.job_sequence = res.job_ids()
            order.planned_makespan = res.makespan
        return produced, costs, consumed

    # ------------------------------------------------------------------
    # 3) 수요 실현 및 판매
    # ------------------------------------------------------------------
    def _sell(self, state: WorldState, d: dt.date) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for code, spec in self.cfg.products.items():
            if spec.market_days_only and not self.cal.is_market_day(d):
                out[code] = dict(actual=0, effective=0, sold=0, lost=0, revenue=0, discount=0.0)
                continue
            actual = self.demand.realized(d, code)
            state.demand_history[code][d] = actual
            disc = state.discount_for(d, code)
            eff = effective_demand(actual, disc, self.cfg.discount_elasticity)
            inv = state.fg_inventory.get(code, 0)
            sold = min(inv, eff)
            state.fg_inventory[code] = inv - sold
            out[code] = dict(
                actual=actual,
                effective=eff,
                sold=sold,
                lost=eff - sold,
                revenue=int(round(unit_price(spec, disc) * sold)),
                discount=disc,
            )
        return out

    # ------------------------------------------------------------------
    # 4) 익일 할인
    # ------------------------------------------------------------------
    def _decide_discount(self, state: WorldState, d: dt.date) -> None:
        tomorrow = d + dt.timedelta(days=1)
        lag = dt.timedelta(days=0 if self.cfg.produce_then_sell_same_day else 1)
        for code, spec in self.cfg.products.items():
            fc = {k: float(v) for k, v in state.forecast_cache.get(code, {}).items()}
            # 판매 가능해지는 시점 기준의 생산 입고 스케줄 (생산일 + lag)
            receipts = {
                day + lag: float(order.n_lots * LOT_SIZE)
                for day, by_product in state.production_plan.items()
                if day > d
                for pcode, order in by_product.items() if pcode == code
            }
            rate = self.pol.discount.decide(
                tomorrow, spec, state.fg_inventory.get(code, 0), fc,
                self.cfg.discount_elasticity, self.cfg.max_discount,
                scheduled_receipts=receipts)
            state.discount_plan.setdefault(tomorrow, {})[code] = rate

    # ------------------------------------------------------------------
    # 5) 자재 발주
    # ------------------------------------------------------------------
    def _material_demand_forecast(self, state: WorldState, d: dt.date, horizon: int = 21):
        """자재 소요 예측: 제품 수요예측 * BOM. 예측 시계를 넘어가면 최근값으로 연장."""
        out: Dict[str, Dict[dt.date, float]] = {m: {} for m in self.cfg.materials}
        for code, spec in self.cfg.products.items():
            fc = state.forecast_cache.get(code, {})
            fallback = float(np.mean(list(fc.values()))) if fc else 0.0
            for i in range(1, horizon + 1):
                day = d + dt.timedelta(days=i)
                if spec.market_days_only and not self.cal.is_market_day(day):
                    val = 0.0
                else:
                    val = float(fc.get(day, fallback))
                for mat, per_unit in self.cfg.bom[code].items():
                    out[mat][day] = out[mat].get(day, 0.0) + val * per_unit
        return out

    def _order_materials(self, state: WorldState, d: dt.date) -> Dict[str, dict]:
        mat_fc = self._material_demand_forecast(state, d)
        result: Dict[str, dict] = {}
        for mat, spec in self.cfg.materials.items():
            pol = self.pol.material.get(mat)
            if pol is None:
                continue
            series = np.array(list(mat_fc[mat].values()), dtype=float)
            sigma_daily = float(np.std(np.diff(series))) if len(series) > 2 else 0.0
            decisions = pol.decide(d, state, spec, mat_fc[mat], sigma_daily)
            if not decisions:
                result[mat] = dict(order_cost=0, purchase_cost=0)
                continue
            orders = to_purchase_orders(decisions, d, self.cfg.materials)
            state.open_orders.extend(orders)
            order_cost = spec.order_cost * len(orders)     # 주문 1회당 주문비
            purchase = sum(spec.option(o.option).unit_cost * o.qty for o in orders)
            result[mat] = dict(order_cost=order_cost, purchase_cost=purchase)
        return result

    # ------------------------------------------------------------------
    # 6) 일 마감
    # ------------------------------------------------------------------
    def _close_day(self, state, d, recv, produced, prod_costs, sales, order_costs, consumed) -> None:
        for code, spec in self.cfg.products.items():
            s = sales.get(code, {})
            c = prod_costs.get(code, {})
            inv = state.fg_inventory.get(code, 0)
            state.daily_records.append(DailyRecord(
                date=d,
                product=code,
                forecast_demand=int(state.forecast_cache.get(code, {}).get(d, 0)),
                actual_demand=s.get("actual", 0),
                effective_demand=s.get("effective", 0),
                discount=s.get("discount", 0.0),
                sold=s.get("sold", 0),
                lost_sales=s.get("lost", 0),
                produced=produced.get(code, 0),
                planned_lots=state.planned_lots(d, code),
                makespan=c.get("makespan", 0.0),
                overtime_hours=c.get("ot", 0),
                revenue=s.get("revenue", 0),
                setup_cost=c.get("setup", 0),
                labor_cost=c.get("labor", 0),
                fg_holding_cost=inv * spec.holding_cost,
                stockout_cost=s.get("lost", 0) * spec.stockout_cost,
                fg_inventory_end=inv,
            ))
        for mat, spec in self.cfg.materials.items():
            inv = state.mat_inventory.get(mat, 0)
            oc = order_costs.get(mat, {})
            state.material_records.append(MaterialRecord(
                date=d,
                material=mat,
                received=recv.get(mat, 0),
                consumed=consumed.get(mat, 0),
                inventory_end=inv,
                order_cost=oc.get("order_cost", 0),
                purchase_cost=oc.get("purchase_cost", 0),
                holding_cost=inv * spec.holding_cost,
            ))

    # ------------------------------------------------------------------
    # 7) 주간 계획 (토요일 자정 마감, 대상 = 차주 일~토)
    # ------------------------------------------------------------------
    def _plan_next_week(self, state: WorldState, d: dt.date) -> None:
        week = self.cal.next_week(d)                      # 일~토
        lag = 0 if self.cfg.produce_then_sell_same_day else 1
        # 계획 시점(토 자정)과 차주 시작(일) 사이의 '틈새일'. 토요일 마감에서는 비지만,
        # 마감 요일이 바뀌어도 안전하도록 일반형으로 남겨 둔다.
        gap_days = [x for x in self.cal.date_range(d + dt.timedelta(days=1),
                                                   week[0] - dt.timedelta(days=1))]

        for code, spec in self.cfg.products.items():
            # 생산가능일과 계획 시계는 제품마다 다르다.
            #   P1(평일 생산): 금요일 배치가 '토 + 다음 일 + 다음 월'까지 커버해야 한다.
            #   P2(매일 생산): 토요일 배치가 '다음 일'만 커버하면 된다.
            weekdays_only = spec.weekday_production_only
            horizon = self.cal.planning_horizon(d, weekdays_only, availability_lag=lag)
            # 생산계획을 제출하는 것은 차주(week)뿐이다. horizon의 꼬리는
            # '마지막 배치가 커버해야 할 수요'로만 쓰이고, 그 날짜에 생산을 배정하지는 않는다.
            production_days = [x for x in week if self.cal.is_production_day(x, weekdays_only)]
            target_dates = self.cal.demand_days(horizon, spec.market_days_only)
            if not target_dates:
                continue
            history = self.demand.history_until(d, code)
            if len(history) < 10:
                continue
            fc: ForecastResult = self.pol.forecaster[code].fit_predict(history, target_dates, d)
            point = {k: int(round(v * self.pol.forecast_bias))
                     for k, v in zip(fc.dates, fc.mu)}
            state.forecast_cache[code].update(point)

            ss = safety_stock_units(fc.cumulative_sigma(1), self.pol.service_level)
            gap_demand = sum(
                state.forecast_cache[code].get(g, 0)
                for g in gap_days
                if not (spec.market_days_only and not self.cal.is_market_day(g))
            )
            opening = max(0, state.fg_inventory.get(code, 0) - int(gap_demand))
            plan = self.pol.lot_sizing.plan(
                product=spec,
                demand_by_date={k: max(0, v) for k, v in point.items()},
                production_days=production_days,
                opening_inventory=opening,
                makespan_oracle=lambda day, lots, c=code: self.makespan_for(c, day, lots),
                safety_stock=ss,
                availability_lag=lag,
            )
            for day, lots in plan.lots_by_date.items():
                state.production_plan.setdefault(day, {})[code] = ProductionOrder(
                    product=code, date=day, n_lots=lots)
