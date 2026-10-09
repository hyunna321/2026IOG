"""IOG 원문 수치 예제로 규칙 구현을 검증한다.

여기가 깨지면 시뮬레이션 결과는 전부 무의미하므로, 정책을 바꾸기 전에 항상 먼저 돌린다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from iog_sim.config import PRODUCTS
from iog_sim.costs import labor_cost, labor_hours, batch_production_cost
from iog_sim.calendar import GameCalendar
from iog_sim.production.sequencing import NEH, SPT, makespan
from iog_sim.production.lotsizing import lots_for
from iog_sim.material.mrp import explode_bom

P1 = PRODUCTS["P1"]
P2 = PRODUCTS["P2"]


# --- 인건비 환산 (문제소개 '생산 비용 계산 예', 매뉴얼 Ledger 예) ----------

@pytest.mark.parametrize("ms,expected", [
    (6680, 6_600_000),    # 66시간 * 100,000
    (8850, 10_600_000),   # 70*100,000 + 18*200,000
    (9466, 11_800_000),   # 70*100,000 + 24*200,000
    (9363, 11_600_000),   # 93*100,000 + 23*100,000 (Ledger 예시)
])
def test_labor_cost_examples(ms, expected):
    assert labor_cost(ms, P1) == expected


def test_labor_hours_floor():
    assert labor_hours(6680) == 66
    assert labor_hours(9466) == 94
    assert labor_hours(99) == 0


def test_total_production_cost_ledger_example():
    bc = batch_production_cost(9363, P1, n_lots=93)
    assert bc.setup == 5_000_000
    assert bc.labor == 11_600_000
    assert bc.total == 16_600_000          # 매뉴얼 Ledger: 총 생산비 16,600,000
    assert bc.overtime_hours == 23


def test_no_batch_no_cost():
    assert batch_production_cost(0, P1, n_lots=0).total == 0


# --- Flow shop makespan (문제소개 4-job 2-machine 예) ----------------------

JOBS_2M = np.array([
    [4, 2],    # A
    [3, 6],    # B
    [8, 4],    # C
    [10, 7],   # D
], dtype=float)


def test_makespan_spt_example():
    """매뉴얼: SPT(공정1 짧은 순) = B,A,C,D -> 32시간."""
    order = [1, 0, 2, 3]
    assert makespan(JOBS_2M, order) == 32
    assert SPT(machine=0).solve(JOBS_2M).order == order


def test_makespan_optimal_is_better_than_spt():
    """매뉴얼 '최적화 순서' B,D,C,A. 본 구현으로는 27시간(Johnson 규칙 최적)으로 계산된다.

    매뉴얼 본문의 28시간과 1시간 차이가 있으므로, 시스템 Details 화면의
    실제 Makespan 값과 한 번 대조해 볼 것 (VERIFY).
    """
    opt_order = [1, 3, 2, 0]
    assert makespan(JOBS_2M, opt_order) == 27
    assert NEH().solve(JOBS_2M).makespan <= makespan(JOBS_2M, [1, 0, 2, 3])


def test_makespan_is_permutation_invariant_in_machines_count():
    rng = np.random.default_rng(0)
    p = rng.uniform(50, 150, size=(30, 20))
    order = list(range(30))
    # 파이프라인 하한: 어떤 순서든 (한 설비 총작업량 + 나머지 최소 통과시간) 이상
    assert makespan(p, order) >= p[:, -1].sum()


# --- Lot / BOM -------------------------------------------------------------

def test_lot_rounding():
    assert lots_for(0) == 0
    assert lots_for(1) == 1
    assert lots_for(1000) == 1
    assert lots_for(69_900) == 70


def test_bom_explosion():
    d = dt.date(2026, 9, 21)
    gross = explode_bom({d: {"P1": 70, "P2": 114}}, {"P1": {"M1": 1, "M2": 2}, "P2": {"M2": 2, "M3": 1}})
    assert gross[d]["M1"] == 70_000
    assert gross[d]["M3"] == 114_000
    assert gross[d]["M2"] == 70_000 * 2 + 114_000 * 2


# --- 캘린더 / 리드타임 ------------------------------------------------------

def test_lead_time_includes_weekend():
    cal = GameCalendar()
    # 2026-04-12(일) 발주 + M1 리드타임 3일 = 04-15(수) 입고 (매뉴얼 예)
    assert cal.arrival_date(dt.date(2026, 4, 12), 3) == dt.date(2026, 4, 15)


def test_decision_day_is_saturday():
    cal = GameCalendar()
    assert cal.is_decision_day(dt.date(2026, 9, 19))       # 토
    assert not cal.is_decision_day(dt.date(2026, 9, 18))   # 금


def test_next_week_from_saturday_is_sun_to_sat():
    """토요일 자정 마감 -> 차주는 일~토 (문제소개의 '카푸치노 수요 일~토'와 일치)."""
    cal = GameCalendar()
    week = cal.next_week(dt.date(2026, 9, 19))   # 토요일
    assert week[0] == dt.date(2026, 9, 20)       # 다음 일요일
    assert week[-1] == dt.date(2026, 9, 26)      # 그 주 토요일
    assert len(week) == 7


def test_production_days_differ_by_product():
    """Production 화면 관측: P1은 토·일 입력칸이 없고(0 고정), P2는 7일 모두 입력 가능."""
    cal = GameCalendar()
    sat, sun, mon = dt.date(2026, 10, 3), dt.date(2026, 9, 27), dt.date(2026, 9, 28)

    assert P1.weekday_production_only and not P2.weekday_production_only
    for weekend_day in (sat, sun):
        assert not cal.is_production_day(weekend_day, P1.weekday_production_only)
        assert cal.is_production_day(weekend_day, P2.weekday_production_only)
    for spec in (P1, P2):
        assert cal.is_production_day(mon, spec.weekday_production_only)


def test_decision_day_for_points_at_the_saturday_that_submitted_that_week():
    """d가 속한 주의 계획을 제출한 토요일. next_week(S)=S+1..S+7이므로 'd 직전 토요일'이다."""
    cal = GameCalendar()
    # 9/20(일)~9/26(토) 주는 9/19(토)에 제출됐다
    for day in (20, 21, 22, 25, 26):
        s = cal.decision_day_for(dt.date(2026, 9, day))
        assert s == dt.date(2026, 9, 19), day
        week = cal.next_week(s)
        assert week[0] <= dt.date(2026, 9, day) <= week[-1]
    # 결정일 당일(토)은 그 주의 마지막 날이므로 일주일 전 토요일이 제출일이다
    assert cal.decision_day_for(dt.date(2026, 9, 19)) == dt.date(2026, 9, 12)


def test_planning_horizon_extends_past_the_week():
    """차주 일~토 + 연장분.

    당일 판매 불가(lag=1)이면 차차주 월요일 생산으로 처음 감당 가능한 날이 화요일이므로,
    이번 금요일 배치가 '토 + 다음 일 + 다음 월'까지 커버해야 한다 -> 연장 2일.
    lag=0(당일 판매 가능 가정)이면 연장은 일요일 하루뿐이다.
    """
    cal = GameCalendar()
    h1 = cal.planning_horizon(dt.date(2026, 9, 19), availability_lag=1)
    assert len(h1) == 9
    assert h1[-1] == dt.date(2026, 9, 28)        # 다음 월요일

    h0 = cal.planning_horizon(dt.date(2026, 9, 19), availability_lag=0)
    assert len(h0) == 8
    assert h0[-1] == dt.date(2026, 9, 27)        # 다음 일요일


def test_production_days_submitted_are_only_next_weeks_weekdays():
    """연장분(다음 일·월)은 '커버해야 할 수요'일 뿐, 생산을 배정하는 날짜가 아니다."""
    cal = GameCalendar()
    week = cal.next_week(dt.date(2026, 9, 19))
    prod = [x for x in week if cal.is_production_day(x)]
    assert prod[0] == dt.date(2026, 9, 21)       # 월
    assert prod[-1] == dt.date(2026, 9, 25)      # 금
    assert len(prod) == 5


def test_sunday_has_no_producer_within_its_own_week():
    """일요일 수요는 그 주 안에서 생산 불가 -> 직전 주 금요일이 미리 만들어야 한다."""
    cal = GameCalendar()
    week = cal.next_week(dt.date(2026, 9, 19))
    sunday = week[0]
    assert not cal.is_production_day(sunday)
    assert [x for x in week if cal.is_production_day(x) and x <= sunday] == []


def test_market_day_is_weekday_minus_hardcoded_holidays():
    """주말은 자동 휴장, 평일은 holidays에 있을 때만 휴장."""
    cal = GameCalendar(holidays={dt.date(2026, 9, 25)})
    assert cal.is_market_day(dt.date(2026, 9, 21))      # 월
    assert not cal.is_market_day(dt.date(2026, 9, 25))  # 금, 공휴일
    assert not cal.is_market_day(dt.date(2026, 9, 26))  # 토, 목록에 없어도 휴장
    assert not cal.is_market_day(dt.date(2026, 9, 27))  # 일


def test_october_holidays():
    """10-05 개천절 대체공휴일, 10-09 한글날."""
    cal = GameCalendar()
    assert not cal.is_market_day(dt.date(2026, 10, 5))
    assert not cal.is_market_day(dt.date(2026, 10, 9))
    assert cal.is_market_day(dt.date(2026, 10, 6))
    assert not cal.is_market_day(dt.date(2026, 10, 10))  # 토


def test_max_job_per_day_matches_processing_time_table():
    """Production 화면 'Max job is 500' + ProcessingTimeTable이 500 Job까지."""
    from iog_sim.config import MAX_LOTS_PER_DAY

    assert MAX_LOTS_PER_DAY == 500


def test_p1_demand_days_match_demand_screen_for_planning_week():
    """Demand 화면(2026-09-27~10-03): P1 입력칸은 09-28~10-02만, 09-27·10-03은 0 고정."""
    cal = GameCalendar()
    open_days = [dt.date(2026, 9, 28), dt.date(2026, 9, 29), dt.date(2026, 9, 30),
                 dt.date(2026, 10, 1), dt.date(2026, 10, 2)]
    closed_days = [dt.date(2026, 9, 27), dt.date(2026, 10, 3)]
    assert all(cal.is_market_day(d) for d in open_days)
    assert not any(cal.is_market_day(d) for d in closed_days)


def test_default_calendar_september_holidays():
    """추석 평일분(24·25) + 주말(26·27)."""
    cal = GameCalendar()
    for day in (24, 25, 26, 27):
        assert not cal.is_market_day(dt.date(2026, 9, day))
    assert cal.is_market_day(dt.date(2026, 9, 23))
    assert cal.is_market_day(dt.date(2026, 9, 28))


# --- Ledger 버킷 ------------------------------------------------------------

def test_expense_buckets_partition_total():
    from iog_sim.ledger import LedgerSummary

    s = LedgerSummary(
        revenue=1000, setup_cost=10, labor_cost=20, fg_holding_cost=3,
        stockout_cost=7, material_order_cost=5, material_purchase_cost=50,
        material_holding_cost=4,
    )
    assert s.production_cost == 30          # 작업준비비 + 인건비
    assert s.inventory_cost == 14           # 완제품 3 + 자재 4 + 재고고갈 7
    assert s.material_cost == 55            # 주문비 5 + 구매비 50
    assert s.total_expense == 99
    assert s.balance == 901


# --- 당일 생산분 당일 판매 (produce_then_sell_same_day) ----------------------

def _same_day_sim(same_day: bool):
    """재고 0인 날에 첫 생산이 일어나는 상황을 만들어 그날 판매를 관측한다."""
    import datetime as dt

    import numpy as np

    from iog_sim.calendar import GameCalendar
    from iog_sim.config import SimConfig
    from iog_sim.demand.forecaster import NaiveForecaster
    from iog_sim.demand.generator import HistoricalReplay
    from iog_sim.engine import PolicySet, Simulator
    from iog_sim.material.policy import SsPolicy
    from iog_sim.production.lotsizing import LotForLot
    from iog_sim.production.sequencing import LTWK
    from iog_sim.sales.discount import NoDiscount

    cal = GameCalendar()
    start, end = dt.date(2026, 1, 3), dt.date(2026, 2, 7)
    series = {p: {d: 20_000 for d in cal.date_range(start - dt.timedelta(days=90), end)}
              for p in ("P1", "P2")}
    for d in list(series["P1"]):
        if not cal.is_market_day(d):
            series["P1"][d] = 0

    table = np.random.default_rng(0).uniform(30, 62, size=(500, 20))
    pol = PolicySet(
        forecaster={"P1": NaiveForecaster(), "P2": NaiveForecaster()},
        lot_sizing=LotForLot(),
        sequencer=LTWK(),
        planning_sequencer=LTWK(),
        material={m: SsPolicy() for m in ("M1", "M2", "M3")},
        discount=NoDiscount(),
        label="test",
    )
    cfg = SimConfig()
    cfg.produce_then_sell_same_day = same_day
    sim = Simulator(cfg, cal, HistoricalReplay(series=series), pol,
                    lambda product, date, n: table[:n])
    res = sim.run(start, end, seed=0)

    first_prod = [r for r in res.state.daily_records if r.produced > 0]
    assert first_prod, "생산이 한 번도 일어나지 않았다"
    return first_prod[0]


def test_same_day_production_is_sellable_by_default():
    """현행 규칙(당일 판매 가능): 첫 생산일에 그날 수요만큼 바로 팔려야 한다.

    09-27 관측(P2 85,000개 생산 -> 같은 날 84,856개 판매, 잔량 144)이 근거다.
    """
    r = _same_day_sim(same_day=True)
    assert r.sold > 0, "당일 생산분이 당일 팔리지 않았다"
    assert r.fg_inventory_end == r.produced - r.sold


def test_same_day_production_is_not_sellable_when_flag_off():
    """플래그를 끄면 옛 가정(익일부터 판매)으로 되돌아가야 한다."""
    r = _same_day_sim(same_day=False)
    assert r.sold == 0, "당일 생산분이 당일 판매되었다"
    assert r.fg_inventory_end == r.produced


# --- Lot sizing: 생산 불가일 처리 ------------------------------------------

def test_l4l_consumes_opening_inventory_on_unproducible_days():
    """생산 불가일(일·월)의 수요도 기초재고를 소모한다 -> 이후 소요량에 반영돼야 한다.

    반영하지 않으면 '기초재고가 아직 남아 있다'고 착각해 화요일 이후를 과소생산한다.
    """
    import datetime as dt

    from iog_sim.production.lotsizing import LotForLot

    sun = dt.date(2026, 9, 20)
    days = [sun + dt.timedelta(days=i) for i in range(4)]     # 일·월·화·수
    demand = {d: 10_000 for d in days}
    prod_days = [d for d in days if d.weekday() < 5]          # 월·화·수

    plan = LotForLot().plan(
        product=P2, demand_by_date=demand, production_days=prod_days,
        opening_inventory=20_000,                              # 일·월 이틀치만 커버
        makespan_oracle=lambda day, lots: 1000.0,
        availability_lag=1,
    )
    # 화·수 수요 20,000개는 전량 생산되어야 한다 (기초재고는 일·월에 전부 소진)
    assert plan.total_lots() == 20


# --- 주간계획 재계획 / 실측 수요 보존 ----------------------------------------

def test_replanning_a_week_clears_previous_assignment():
    """같은 주를 두 번 계획하면 이전 배정이 남으면 안 된다.

    `lots_by_date`는 Lot 0인 날을 담지 않으므로, 덮어쓰기만 하면 '전엔 잡혔는데
    이번엔 안 잡힌 날'이 옛 값 그대로 제출표에 실린다.
    """
    import datetime as dt

    import numpy as np

    from iog_sim.calendar import GameCalendar
    from iog_sim.config import SimConfig
    from iog_sim.demand.forecaster import NaiveForecaster
    from iog_sim.demand.generator import HistoricalReplay
    from iog_sim.engine import PolicySet, Simulator
    from iog_sim.material.policy import SsPolicy
    from iog_sim.production.lotsizing import LotForLot
    from iog_sim.production.sequencing import LTWK
    from iog_sim.sales.discount import NoDiscount
    from iog_sim.state import ProductionOrder, WorldState

    cal, cfg = GameCalendar(), SimConfig()
    decision = dt.date(2026, 10, 3)
    week = cal.next_week(decision)

    # 수요는 평탄하게 깔아 두고, 유령이 박힌 날만 관찰한다.
    hist = {d: 10_000 for d in cal.date_range(decision - dt.timedelta(days=60), decision)}
    pol = PolicySet(
        forecaster={"P1": NaiveForecaster(), "P2": NaiveForecaster()},
        lot_sizing=LotForLot(), sequencer=LTWK(), planning_sequencer=LTWK(),
        material={m: SsPolicy() for m in ("M1", "M2", "M3")},
        discount=NoDiscount(), label="test")
    table = np.random.default_rng(0).uniform(30, 62, size=(500, 20))
    sim = Simulator(cfg, cal, HistoricalReplay(series={"P1": dict(hist), "P2": dict(hist)}),
                    pol, lambda product, date, n: table[:n])

    state = WorldState(fg_inventory={"P1": 10_000_000, "P2": 10_000_000},
                       mat_inventory={m: 0 for m in cfg.materials})
    state.forecast_cache = {p: {} for p in cfg.products}
    # 이전(backfill) 계획이 남긴 유령 배정. 재고를 잔뜩 줬으므로 재계획은 이 날 Lot을
    # 잡지 않는다 -> 지워지지 않으면 999가 그대로 남는다.
    ghost = week[1]
    state.production_plan[ghost] = {"P1": ProductionOrder("P1", ghost, 999)}

    sim._plan_week(state, decision, week)

    order = state.production_plan.get(ghost, {}).get("P1")
    assert order is None or order.n_lots != 999, "이전 주간계획 배정이 남았다"


def test_extend_with_forecast_keeps_published_actuals():
    """미래 구간이라도 이미 값이 있는 날은 예측으로 덮지 않는다."""
    import datetime as dt

    from iog_sim.demand.forecaster import NaiveForecaster
    from iog_sim.demand.generator import HistoricalReplay, extend_with_forecast

    origin = dt.date(2026, 9, 29)
    series = {d: 100 for d in [origin - dt.timedelta(days=i) for i in range(1, 21)]}
    known = origin + dt.timedelta(days=1)
    series[known] = 777                                  # 시스템이 먼저 공시한 실측치
    replay = HistoricalReplay(series={"P2": dict(series)})

    window = [origin + dt.timedelta(days=i) for i in range(1, 5)]
    extend_with_forecast(replay, {"P2": NaiveForecaster()}, {"P2": window}, window,
                         origin=origin)

    assert replay.series["P2"][known] == 777, "공시된 실측치가 예측으로 덮였다"
    assert replay.series["P2"][window[-1]] == 100, "빈 날짜는 예측으로 채워야 한다"


# --- 자재 소요 = 생산계획 x BOM / 라운드 종료 / backfill ----------------------

def _flat_sim(start, end, demand=20_000, calendar=None):
    """평탄한 수요 + 랜덤 처리시간표로 만든 소형 시뮬레이터."""
    from iog_sim.config import SimConfig
    from iog_sim.demand.forecaster import NaiveForecaster
    from iog_sim.demand.generator import HistoricalReplay
    from iog_sim.engine import PolicySet, Simulator
    from iog_sim.material.policy import DualSourcingPolicy, SsPolicy
    from iog_sim.production.lotsizing import DynamicLotSizing
    from iog_sim.production.sequencing import LTWK
    from iog_sim.sales.discount import NoDiscount

    cal = calendar or GameCalendar()
    days = cal.date_range(start - dt.timedelta(days=90), end)
    series = {"P1": {d: (demand if cal.is_market_day(d) else 0) for d in days},
              "P2": {d: demand for d in days}}
    table = np.random.default_rng(0).uniform(30, 62, size=(500, 20))
    pol = PolicySet(
        forecaster={"P1": NaiveForecaster(), "P2": NaiveForecaster()},
        lot_sizing=DynamicLotSizing(), sequencer=LTWK(), planning_sequencer=LTWK(),
        material={"M1": SsPolicy(), "M2": DualSourcingPolicy(), "M3": SsPolicy()},
        discount=NoDiscount(), label="test")
    return Simulator(SimConfig(), cal, HistoricalReplay(series=series), pol,
                     lambda product, date, n: table[:n])


def test_project_lots_repeats_last_planned_weekday():
    from iog_sim.material.mrp import project_lots

    cal = GameCalendar()
    p1 = {"P1": cal.is_market_day}
    thu, mon = dt.date(2026, 10, 8), dt.date(2026, 10, 5)          # 10-05 대체공휴일
    plan = {thu: {"P1": 294}, mon - dt.timedelta(days=7): {"P1": 271}}
    out = project_lots(plan, planned_through=dt.date(2026, 10, 10),
                       days=[thu, thu + dt.timedelta(days=7), mon + dt.timedelta(days=7),
                             dt.date(2026, 10, 17)], producing=p1)
    assert out[thu]["P1"] == 294
    assert out[thu + dt.timedelta(days=7)]["P1"] == 294     # 다음 목요일 = 직전 목요일 패턴
    # 다음 월요일은 공휴일(0)이 아니라 그 전 개장 월요일(271)을 따른다
    assert out[mon + dt.timedelta(days=7)]["P1"] == 271
    assert out[dt.date(2026, 10, 17)] == {}                  # 토요일


def test_material_requirements_follow_production_plan():
    """소요는 생산일에 생산량만큼 잡혀야 한다 (수요일 기준 예측이 아니라)."""
    from iog_sim.state import ProductionOrder, WorldState

    sim = _flat_sim(dt.date(2026, 10, 3), dt.date(2026, 10, 10))
    state = WorldState()
    fri, sat = dt.date(2026, 10, 9), dt.date(2026, 10, 10)
    state.production_plan[fri] = {"P1": ProductionOrder("P1", fri, 290),
                                  "P2": ProductionOrder("P2", fri, 175)}
    state.planned_through = sat
    state.forecast_sigma = {"P1": 3_000.0, "P2": 4_000.0}

    req = sim._material_requirements(state, dt.date(2026, 10, 3))
    assert req["M1"][fri] == 290_000
    assert req["M2"][fri] == 2 * (290_000 + 175_000)
    assert req["M3"][fri] == 175_000
    assert req["M3"][sat] == 0
    # 계획 밖 다음 금요일: P2는 같은 패턴, P1은 10-09가 한글날이라 원본으로 쓰지 않는다
    assert req["M3"][fri + dt.timedelta(days=7)] == 175_000
    assert req["M1"][fri + dt.timedelta(days=7)] == 0
    # sigma는 소요 계열의 덩어리짐이 아니라 예측오차에서 온다
    assert sim._material_sigma(state, "M2") == pytest.approx(np.hypot(2 * 3_000, 2 * 4_000))
    assert sim._material_sigma(state, "M1") == pytest.approx(3_000)


def test_round_end_stops_simulation_planning_and_late_orders():
    from iog_sim.calendar import GameCalendar as Cal

    end = dt.date(2026, 2, 4)                                  # 수요일
    cal = Cal(round_ends=[end])
    sim = _flat_sim(dt.date(2026, 1, 3), end, calendar=cal)
    res = sim.run(dt.date(2026, 1, 3), dt.date(2026, 2, 28))

    assert res.end == end
    assert max(r.date for r in res.state.daily_records) == end
    assert all(day <= end for day, by in res.state.production_plan.items() if by)
    assert all(o.arrival_date <= end for o in res.state.open_orders
               if o.order_date >= dt.date(2026, 1, 10))


def test_backfill_plans_only_the_current_week():
    """화요일 시작 backfill이 다음 주(일~) 날짜에 오더를 만들면 안 된다."""
    tue = dt.date(2026, 1, 13)
    sim = _flat_sim(tue, tue)
    res = sim.run(tue, tue, backfill_current_week=True)
    sat = dt.date(2026, 1, 17)
    assert res.state.planned_through == sat
    assert all(day <= sat for day, by in res.state.production_plan.items() if by)


# --- 안전재고: 판매기회비 > 재고유지비 비대칭 -----------------------------------

def test_critical_ratio_reflects_stockout_vs_holding():
    """Cu = 판매기회비 250 + 놓친 마진, Co = 재고유지비 30 -> 임계비율이 0.9를 넘는다."""
    from iog_sim.production.safety_stock import stockout_critical_ratio

    p1 = stockout_critical_ratio(P1, unit_material_cost=80)       # M1 30 + M2 2x25
    assert p1 == pytest.approx((250 + 170) / (250 + 170 + 30))
    # 라운드 종료 직전에는 남는 재고를 못 팔아 Co에 자재비가 붙는다 -> 버퍼를 낮춘다
    assert stockout_critical_ratio(P1, 80, round_end=True) < p1


def test_cumulative_safety_stock_holds_buffer_every_day():
    """누적 규칙은 주중 매일 '누적 생산 >= 누적 예측 + ss_h'를 만족해야 한다."""
    from iog_sim.demand.forecaster import ForecastResult
    from iog_sim.production.lotsizing import LotForLot
    from iog_sim.production.safety_stock import CumulativeModelSigma, EndOfHorizon

    days = [dt.date(2026, 10, 12) + dt.timedelta(days=i) for i in range(5)]   # 월~금
    fc = ForecastResult(origin=None, dates=days, mu=np.full(5, 100_000.0),
                        sigma=10_000 * np.sqrt(np.arange(1, 6)))
    for rule, mid_week_buffer in ((CumulativeModelSigma(0.95), True), (EndOfHorizon(0.95), False)):
        ss = rule.cumulative_targets(fc, [], None, 0.95)
        inc = np.diff(np.concatenate([[0], ss]))
        plan = LotForLot().plan(P1, {d: 100_000 + int(i) for d, i in zip(days, inc)}, days, 0,
                                lambda day, lots: 1000.0)
        cum_prod = np.cumsum([plan.lots_by_date.get(d, 0) * 1000 for d in days])
        cum_dem = np.cumsum([100_000] * 5)
        assert np.all(cum_prod - cum_dem >= ss)
        assert ((cum_prod - cum_dem)[1] >= 10_000) == mid_week_buffer


# --- 자재 소요 추정: 휴장일 다음 주 -------------------------------------------

def test_project_lots_does_not_zero_the_week_after_a_holiday():
    """2026-10-09 사례: 확정 계획은 10/4~10/10, 지난주 월(10/5)·금(10/9)이 휴장.

    다음 주 월·금의 P1 생산을 같은 요일에서 찾다가 입력 범위 밖(9/28, 10/2)에 닿으면
    0이 아니라 최근 생산일 평균으로 추정해야 한다. 0으로 두면 M1을 리드타임 안에 발주하지 못한다.
    """
    from iog_sim.material.mrp import project_lots

    cal = GameCalendar()
    d = dt.date
    locked = {
        d(2026, 10, 4): {"P2": 86}, d(2026, 10, 5): {"P2": 86},
        d(2026, 10, 6): {"P1": 275, "P2": 86}, d(2026, 10, 7): {"P1": 275, "P2": 86},
        d(2026, 10, 8): {"P1": 294, "P2": 86}, d(2026, 10, 9): {"P2": 175}, d(2026, 10, 10): {},
    }
    producing = {"P1": lambda day: day.weekday() < 5 and cal.is_market_day(day),
                 "P2": lambda day: True}
    days = [d(2026, 10, 10) + dt.timedelta(days=i) for i in range(1, 8)]     # 10/11(일) ~ 10/17(토)
    proj = project_lots(locked, d(2026, 10, 10), days, producing)

    recent_avg = round((275 + 275 + 294) / 3)
    assert proj[d(2026, 10, 12)]["P1"] == recent_avg          # 지난주 월요일 휴장 -> 평균
    assert proj[d(2026, 10, 16)]["P1"] == recent_avg          # 지난주 금요일 휴장 -> 평균
    assert proj[d(2026, 10, 13)]["P1"] == 275                  # 같은 요일 계획이 있으면 그대로
    assert proj[d(2026, 10, 15)]["P1"] == 294
    assert "P2" not in proj[d(2026, 10, 17)]                   # 원본 계획이 실제 0이면 0 유지


# --- M2 긴급발주 판단 ---------------------------------------------------------

def test_urgent_m2_ignores_unfixable_tomorrow_and_orders_one_day_early():
    """긴급(LT 2일)은 오늘 주문하면 모레 닿는다.

    - 내일 부족은 긴급으로도 못 막으므로 발주하지 않고, 수량에도 넣지 않는다.
    - 긴급이 닿는 날 + 하루(slack) 안의 부족은 오늘 발주한다 ('마지막 가능일'에 몰리지 않게).
    - 그보다 먼 부족은 아직 이르므로 다음 날 다시 판단한다.
    """
    from iog_sim.config import MATERIALS
    from iog_sim.material.policy import DualSourcingPolicy
    from iog_sim.state import WorldState

    spec, d0, pol = MATERIALS["M2"], dt.date(2026, 10, 13), DualSourcingPolicy()

    def urgent_qty(on_hand, req_by_offset):
        st = WorldState(mat_inventory={"M2": on_hand})
        req = {d0 + dt.timedelta(days=k): q for k, q in req_by_offset.items()}
        return sum(x.qty for x in pol.decide(d0, st, spec, req, 0.0) if x.option == "urgent")

    assert urgent_qty(100_000, {1: 300_000}) == 0                                   # 내일만 부족
    assert urgent_qty(500_000, {1: 200_000, 2: 200_000, 3: 200_000}) == 100_000      # 3일 뒤 부족
    assert urgent_qty(100_000, {1: 300_000, 2: 150_000}) == 150_000                  # 내일분은 수량 제외
    assert urgent_qty(500_000, {4: 600_000}) == 0                                    # 아직 이름
