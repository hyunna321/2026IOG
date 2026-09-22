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


def test_market_day_follows_hardcoded_holiday_set_only():
    """휴장일은 요일 계산이 아니라 holidays 집합 멤버십으로만 판정한다.

    주말이어도 holidays에 없으면 개장일, 평일이어도 holidays에 있으면 휴장일이다.
    """
    cal = GameCalendar(holidays={dt.date(2026, 9, 25), dt.date(2026, 9, 26)})
    assert cal.is_market_day(dt.date(2026, 9, 21))      # 월, holidays에 없음
    assert not cal.is_market_day(dt.date(2026, 9, 26))  # 토, holidays에 명시됨
    assert not cal.is_market_day(dt.date(2026, 9, 25))  # 금, holidays에 명시됨
    assert cal.is_market_day(dt.date(2026, 9, 27))      # 일, holidays에 없으므로 개장일


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


def test_default_calendar_uses_hardcoded_september_holidays():
    """MARKET_HOLIDAYS_BY_MONTH에 하드코딩된 2026-09 휴장일(24~27일) 확인."""
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


# --- 당일 생산분 당일 판매 불가 --------------------------------------------

def test_same_day_production_is_not_sellable():
    """재고 0인 날에 생산이 일어나도 그날 판매는 0이어야 한다."""
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
    sim = Simulator(SimConfig(), cal, HistoricalReplay(series=series), pol,
                    lambda product, date, n: table[:n])
    res = sim.run(start, end, seed=0)

    first_prod = [r for r in res.state.daily_records
                  if r.produced > 0 and r.fg_inventory_end == r.produced]
    assert first_prod, "생산이 한 번도 일어나지 않았다"
    r = first_prod[0]
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
