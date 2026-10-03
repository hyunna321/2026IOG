"""시뮬레이션 엔진 - 일별 이벤트 루프.

하루 안의 이벤트 순서 (IOG 규칙)

    1) 자재 입고        발주일 + LT == 오늘 -> 입고. 입고 당일 생산 투입 가능
    2) 생산 실행        제품별 생산가능일만. 자재 차감 -> 완제품 입고. Makespan -> 생산비
    3) 수요 실현·판매   당일 생산분 당일 판매 가능 (`produce_then_sell_same_day`)
    4) (토요일) 차주 계획  수요예측 -> Lot sizing -> 생산계획 확정 (자정 마감)
    5) 익일 할인 결정   Sales 화면 "다음날 할인율"
    6) 자재 발주 결정   생산계획 x BOM 소요를 보고 (s,S) / 이중조달
    7) 일 마감          재고유지비 부과, 기록 적재

토요일에는 제출할 차주 계획을 먼저 세운 뒤 그 계획을 보고 20시 발주를 정한다.

라운드 종료일(`GameCalendar.round_end_for`)을 넘겨서는 시뮬레이션·생산·발주를 하지 않는다.

상태 변경은 이 파일에서만 일어난다. 정책 객체는 상태를 읽고 '결정'만 반환한다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .calendar import GameCalendar
from .config import LOT_SIZE, SimConfig
from .costs import batch_production_cost
from .demand.forecaster import Forecaster
from .demand.generator import DemandGenerator
from .ledger import LedgerSummary, summarize
from .material.mrp import explode_bom, project_lots
from .material.policy import MaterialPolicy, to_purchase_orders
from .production.lotsizing import LotSizingPolicy
from .production.safety_stock import EndOfHorizon, SafetyStockRule, stockout_critical_ratio
from .production.sequencing import LTWK, Sequencer, SequenceResult
from .sales.discount import DiscountPolicy, effective_demand, unit_price
from .state import DailyRecord, MaterialRecord, ProductionOrder, PurchaseOrder, WorldState

# (제품, 생산일, Lot수) -> 처리시간 행렬
ProcessingTimeProvider = Callable[[str, dt.date, int], np.ndarray]

# 자재 소요를 내다보는 기간(일). M2 일반 LT 8일을 넉넉히 덮는다.
MATERIAL_HORIZON_DAYS = 21


@dataclass
class PolicySet:
    """what-if 실험에서 갈아끼우는 단위. 이 객체 하나가 곧 '하나의 전략'이다."""

    forecaster: Dict[str, Forecaster]
    lot_sizing: LotSizingPolicy
    sequencer: Sequencer                      # 실제 투입 순서 (제출용)
    material: Dict[str, MaterialPolicy]
    discount: DiscountPolicy
    # 완제품 안전재고 규칙. service_level None = 비용 임계비율(판매기회비 vs 재고유지비)
    safety_stock: SafetyStockRule = EndOfHorizon(0.95)
    label: str = "policy"
    # Lot sizing DP가 후보 배치마다 수백 번 호출하는 '싼' Makespan 추정기
    planning_sequencer: Sequencer = LTWK()


@dataclass
class SimResult:
    summary: LedgerSummary
    state: WorldState
    label: str = ""
    seed: int = 0
    warmup_days: int = 0
    end: Optional[dt.date] = None             # 라운드 종료로 잘린 실제 마지막 날


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
        self._round_end: Optional[dt.date] = None

    @property
    def _lag(self) -> int:
        """생산일 D의 산출물이 D+lag부터 판매 가능."""
        return 0 if self.cfg.produce_then_sell_same_day else 1

    def _in_round(self, d: dt.date) -> bool:
        return self._round_end is None or d <= self._round_end

    # ------------------------------------------------------------------
    # Makespan 오라클 / 시퀀스
    # ------------------------------------------------------------------
    def makespan_for(self, product: str, date: dt.date, n_lots: int) -> float:
        if n_lots <= 0:
            return 0.0
        key = (product, date, n_lots)
        if key not in self._ms_cache:
            self._ms_cache[key] = self.pol.planning_sequencer.solve(
                self.pt(product, date, n_lots)).makespan
        return self._ms_cache[key]

    def sequence_for(self, product: str, date: dt.date, n_lots: int) -> SequenceResult:
        key = (product, date, n_lots)
        if key not in self._seq_cache:
            self._seq_cache[key] = self.pol.sequencer.solve(self.pt(product, date, n_lots))
        return self._seq_cache[key]

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def run(self, start: dt.date, end: dt.date, seed: int = 0,
            warmup_days: int = 0,
            initial_fg_inventory: Optional[Dict[str, int]] = None,
            initial_mat_inventory: Optional[Dict[str, int]] = None,
            backfill_current_week: bool = False,
            locked_production_plan: Optional[Dict[dt.date, Dict[str, int]]] = None,
            initial_open_orders: Optional[List[PurchaseOrder]] = None) -> SimResult:
        """start~end를 하루씩 진행한다. end가 라운드 종료일을 넘으면 종료일에서 멈춘다.

        warmup_days: 집계(summary)에서 앞 구간을 뺀다. 시작 직후의 품절은 정책이 아니라
            시작 조건(재고 0, 계획 없음)의 결과이기 때문이다. 기록은 전부 남는다.
        initial_fg_inventory / initial_mat_inventory: 시작 재고를 실측값으로 덮어쓴다.
        backfill_current_week: 라운드 중간에 시작해 아직 계획이 없을 때, start-1을 예측 원점으로
            start~이번 주 토요일까지만 계획을 깐다. 다음 주 날짜는 건드리지 않는다.
        locked_production_plan: {날짜: {제품: Lot수}}. 이미 제출된 계획. 그날 계획을 통째로 대체한다.
        initial_open_orders: 이미 발주해 아직 도착하지 않은 주문. 비용은 이미 지출된 것으로 본다.
        """
        self.demand.reset(seed)
        self._round_end = self.cal.round_end_for(start)
        if self._round_end is not None:
            end = min(end, self._round_end)

        state = WorldState(
            fg_inventory=dict(initial_fg_inventory) if initial_fg_inventory is not None
                         else {p: 0 for p in self.cfg.products},
            mat_inventory=dict(initial_mat_inventory) if initial_mat_inventory is not None
                          else {m: s.initial_inventory for m, s in self.cfg.materials.items()},
        )
        state.forecast_cache = {p: {} for p in self.cfg.products}

        if initial_open_orders:
            # start 이전 도착분은 이미 실측 재고에 들어 있다. 넣으면 입고일을 영영 못 만나
            # 재고포지션만 부풀리므로 거부한다.
            stale = [o for o in initial_open_orders if o.arrival_date < start]
            if stale:
                raise ValueError(
                    f"initial_open_orders에 start({start}) 이전 도착 주문이 있다: "
                    + ", ".join(f"{o.material} {o.qty}@{o.arrival_date}" for o in stale))
            state.open_orders.extend(
                PurchaseOrder(material=o.material, option=o.option, qty=o.qty,
                              order_date=o.order_date, arrival_date=o.arrival_date)
                for o in initial_open_orders)

        if backfill_current_week:
            origin = start - dt.timedelta(days=1)
            self._plan_week(state, origin,
                            self.cal.date_range(start, self.cal.next_decision_day(start)))
        if locked_production_plan:
            for day, by_product in locked_production_plan.items():
                state.production_plan[day] = {
                    code: ProductionOrder(product=code, date=day, n_lots=lots)
                    for code, lots in by_product.items() if lots > 0
                }
            last = max(locked_production_plan)
            state.planned_through = max(filter(None, (state.planned_through, last)))

        for d in self.cal.date_range(start, end):
            recv = self._receive_materials(state, d)
            if self.cfg.produce_then_sell_same_day:
                produced, prod_costs, consumed = self._produce(state, d)
                sales = self._sell(state, d)
            else:
                sales = self._sell(state, d)
                produced, prod_costs, consumed = self._produce(state, d)
            if self.cal.is_decision_day(d):
                self._plan_week(state, d, self.cal.next_week(d))
            self._decide_discount(state, d)
            order_costs = self._order_materials(state, d)
            self._close_day(state, d, recv, produced, prod_costs, sales, order_costs, consumed)

        cutoff = start + dt.timedelta(days=warmup_days)
        daily = [r for r in state.daily_records if r.date >= cutoff]
        mats = [r for r in state.material_records if r.date >= cutoff]
        return SimResult(summary=summarize(daily, mats), state=state,
                         label=self.pol.label, seed=seed, warmup_days=warmup_days, end=end)

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
    # 2) 생산 (자재가 모자라면 가능한 Lot만)
    # ------------------------------------------------------------------
    def _produce(self, state: WorldState, d: dt.date):
        produced: Dict[str, int] = {}
        costs: Dict[str, dict] = {}
        consumed: Dict[str, int] = {}

        for code, order in sorted(state.production_plan.get(d, {}).items()):
            spec = self.cfg.products[code]
            if not self.cal.is_production_day(d, spec.weekday_production_only):
                continue
            feasible = order.n_lots
            for mat, per_unit in self.cfg.bom[code].items():
                feasible = min(feasible, state.mat_inventory.get(mat, 0) // (per_unit * LOT_SIZE))
            feasible = max(0, int(feasible))
            if feasible == 0:
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
    # 5) 익일 할인
    # ------------------------------------------------------------------
    def _decide_discount(self, state: WorldState, d: dt.date) -> None:
        tomorrow = d + dt.timedelta(days=1)
        lag = dt.timedelta(days=self._lag)
        for code, spec in self.cfg.products.items():
            fc = {k: float(v) for k, v in state.forecast_cache.get(code, {}).items()}
            receipts = {
                day + lag: float(by_product[code].n_lots * LOT_SIZE)
                for day, by_product in state.production_plan.items()
                if day > d and code in by_product
            }
            rate = self.pol.discount.decide(
                tomorrow, spec, state.fg_inventory.get(code, 0), fc,
                self.cfg.discount_elasticity, self.cfg.max_discount,
                scheduled_receipts=receipts)
            state.discount_plan.setdefault(tomorrow, {})[code] = rate

    # ------------------------------------------------------------------
    # 6) 자재 발주
    # ------------------------------------------------------------------
    def _material_requirements(self, state: WorldState, d: dt.date) -> Dict[str, Dict[dt.date, float]]:
        """d+1부터 MATERIAL_HORIZON_DAYS일간 자재별 일 소요량 (생산계획 x BOM).

        계획이 아직 한 번도 서지 않은 시작 직후에만 수요예측 x BOM으로 대신한다.
        라운드 종료 뒤 날짜는 생산이 없으므로 0이다.
        """
        days = [d + dt.timedelta(days=i) for i in range(1, MATERIAL_HORIZON_DAYS + 1)]
        out: Dict[str, Dict[dt.date, float]] = {m: {day: 0.0 for day in days}
                                                for m in self.cfg.materials}
        live = [day for day in days if self._in_round(day)]

        if state.planned_through is not None:
            plan_lots = {day: {c: o.n_lots for c, o in by.items()}
                         for day, by in state.production_plan.items()}
            gross = explode_bom(project_lots(plan_lots, state.planned_through, live,
                                             self._producing()), self.cfg.bom)
            for day, by_mat in gross.items():
                for mat, qty in by_mat.items():
                    out[mat][day] = float(qty)
            return out

        for code, spec in self.cfg.products.items():
            fc = state.forecast_cache.get(code, {})
            fallback = float(np.mean(list(fc.values()))) if fc else 0.0
            for day in live:
                if spec.market_days_only and not self.cal.is_market_day(day):
                    continue
                val = float(fc.get(day, fallback))
                for mat, per_unit in self.cfg.bom[code].items():
                    out[mat][day] += val * per_unit
        return out

    def _producing(self) -> Dict[str, Callable[[dt.date], bool]]:
        """제품별 '그날 생산이 일어날 수 있는가'. 개장일 수요만 있는 P1은 판매 가능일이 개장일인 날만."""
        lag = dt.timedelta(days=self._lag)

        def make(spec):
            def can(day: dt.date) -> bool:
                return (self.cal.is_production_day(day, spec.weekday_production_only)
                        and (not spec.market_days_only or self.cal.is_market_day(day + lag)))
            return can
        return {code: make(spec) for code, spec in self.cfg.products.items()}

    def _material_sigma(self, state: WorldState, mat: str) -> float:
        """자재 일 소요의 예측오차 표준편차 = 제품 1-step 예측오차 x BOM (제품 간 독립 가정).

        소요 계열 자체의 들쭉날쭉함(주말 0, 금요일 배치)은 계획된 것이지 불확실성이 아니므로 쓰지 않는다.
        """
        var = sum((per_unit * state.forecast_sigma.get(code, 0.0)) ** 2
                  for code, bom in self.cfg.bom.items()
                  for m, per_unit in bom.items() if m == mat)
        return float(np.sqrt(var))

    def _order_materials(self, state: WorldState, d: dt.date) -> Dict[str, dict]:
        req = self._material_requirements(state, d)
        horizon_end = d + dt.timedelta(days=MATERIAL_HORIZON_DAYS)
        result: Dict[str, dict] = {}
        for mat, spec in self.cfg.materials.items():
            pol = self.pol.material.get(mat)
            if pol is None:
                continue
            decisions = pol.decide(d, state, spec, req[mat], self._material_sigma(state, mat))
            orders = to_purchase_orders(decisions, d, self.cfg.materials)
            if self._round_end is not None:
                orders = self._trim_for_round_end(state, mat, orders, req[mat], horizon_end)
            if not orders:
                continue
            state.open_orders.extend(orders)
            result[mat] = dict(
                order_cost=spec.order_cost * len(orders),
                purchase_cost=sum(spec.option(o.option).unit_cost * o.qty for o in orders))
        return result

    def _trim_for_round_end(self, state: WorldState, mat: str, orders: List[PurchaseOrder],
                            req: Dict[dt.date, float], horizon_end: dt.date) -> List[PurchaseOrder]:
        """라운드 안에 쓸 수 없는 발주를 버린다.

        - 종료일 뒤에 도착하는 주문은 통째로 뺀다.
        - 종료일이 소요 시계 안에 들어오면, 남은 소요 - 재고포지션을 넘는 수량을 깎는다.
          (s,S)·EOQ는 운영이 계속된다고 보고 S까지 채우므로 그대로 두면 남는 자재에 구매비·주문비를 낸다.
        """
        kept = [o for o in orders if o.arrival_date <= self._round_end]
        if self._round_end > horizon_end:
            return kept
        room = sum(q for day, q in req.items() if day <= self._round_end) \
            - state.inventory_position(mat)
        trimmed = []
        for o in kept:
            qty = int(min(o.qty, max(0.0, room)))
            if qty <= 0:
                continue
            o.qty = qty
            room -= qty
            trimmed.append(o)
        return trimmed

    # ------------------------------------------------------------------
    # 7) 일 마감
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
    # 4) 주간 계획 (토요일)
    # ------------------------------------------------------------------
    def _service_level(self, code: str, spec, round_end: bool) -> float:
        """규칙에 고정 서비스수준이 없으면 비용 임계비율. 종료 직전에는 종료용 비율이 상한."""
        unit_mat = sum(per_unit * self.cfg.materials[m].option("normal").unit_cost
                       for m, per_unit in self.cfg.bom[code].items())
        end_ratio = stockout_critical_ratio(spec, unit_mat, round_end=True)
        fixed = self.pol.safety_stock.service_level
        if fixed is None:
            return end_ratio if round_end else stockout_critical_ratio(spec, unit_mat)
        return min(fixed, end_ratio) if round_end else fixed

    def _plan_week(self, state: WorldState, origin: dt.date, week: Sequence[dt.date]) -> None:
        """origin까지의 수요로 예측해 `week` 날짜에만 생산을 배정한다.

        정규 경로는 토요일 결정 -> 차주 일~토. backfill은 start-1 원점 -> 이번 주 남은 날.
        `week`의 기존 배정은 먼저 지우므로, 새 계획이 0으로 정한 날에 옛 오더가 남지 않는다.
        라운드 종료일 뒤 날짜는 계획하지 않는다.

        안전재고: 규칙이 준 누적 버퍼 목표 ss_h의 증분을 그날 수요에 더해 Lot sizing에 넘긴다.
        종료일이 커버 구간을 자르면 남는 재고는 못 팔므로 Co에 자재비를 넣은 임계비율로 낮춘다.
        """
        week = [x for x in week if self._in_round(x)]
        if not week:
            return
        lag = self._lag
        for code, spec in self.cfg.products.items():
            weekdays_only = spec.weekday_production_only
            full_horizon = self.cal.coverage_horizon(week, weekdays_only, availability_lag=lag)
            horizon = [x for x in full_horizon if self._in_round(x)]
            cut_by_round_end = len(horizon) < len(full_horizon)
            target_dates = self.cal.demand_days(horizon, spec.market_days_only)
            history = self.demand.history_until(origin, code)
            if not target_dates or len(history) < 10:
                continue
            fc = self.pol.forecaster[code].fit_predict(history, target_dates, origin)
            point = {k: max(0, int(round(v))) for k, v in zip(fc.dates, fc.mu)}
            state.forecast_cache[code].update(point)
            state.forecast_sigma[code] = float(fc.sigma[0])

            service = self._service_level(code, spec, cut_by_round_end)
            ss = self.pol.safety_stock.cumulative_targets(
                fc, history, self.pol.forecaster[code], service)
            increments = np.diff(np.concatenate([[0], ss]))
            planned_demand = {day: point[day] + int(inc)
                              for day, inc in zip(fc.dates, increments)}
            plan = self.pol.lot_sizing.plan(
                product=spec,
                demand_by_date=planned_demand,
                production_days=[x for x in week if self.cal.is_production_day(x, weekdays_only)],
                opening_inventory=state.fg_inventory.get(code, 0),
                makespan_oracle=lambda day, lots, c=code: self.makespan_for(c, day, lots),
                availability_lag=lag,
            )
            for day in week:
                state.production_plan.get(day, {}).pop(code, None)
            for day, lots in plan.lots_by_date.items():
                state.production_plan.setdefault(day, {})[code] = ProductionOrder(
                    product=code, date=day, n_lots=lots)
        state.planned_through = max(filter(None, (state.planned_through, week[-1])))
