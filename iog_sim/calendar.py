"""캘린더: 개장일 / 생산가능일 / 의사결정일 / 리드타임 계산.

규칙
- P1 수요는 주식시장 개장일(평일 - 공휴일)에만 발생, 휴장일 수요 0.
- P2 수요는 365일 매일 발생.
- 생산은 평일에만 수행(문제소개 '평일 생산').
- 자재 리드타임은 '달력일' 기준. 주말·공휴일 포함하여 발주일 + LT = 입고일.
- 수요예측/생산계획 입력 마감은 금주 금요일 23:59, 대상은 차주.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

# 휴장일은 평일/주말 계산으로 추정하지 않고, 시스템 화면에서 확인한 값을 월별로
# 직접 하드코딩한다. (year, month) -> 휴장일(day) 집합.
# 아직 확인되지 않은 월은 비어 있으므로 그 달은 전부 개장일로 처리된다 --
# 라운드가 진행되며 시스템에서 다음 달 휴장일이 공지되는 대로 이어서 채울 것.
MARKET_HOLIDAYS_BY_MONTH: Dict[Tuple[int, int], Set[int]] = {
    (2026, 9): {24, 25, 26, 27},
    # Demand/Production 화면에서 확인: 09-27, 10-03은 P1 입력칸이 없고 0으로 고정.
    # 10월 토·일은 전부 휴장으로 넣어 둔다(9월 관측과 과거 P1 계열 모두 주말 수요가 0).
    # VERIFY: 10-09(한글날)은 화면에서 아직 확인하지 못했다. 그 주 계획을 세우기 전에
    # Demand 화면에서 P1 입력칸 유무로 반드시 확인할 것.
    (2026, 10): {3, 4, 10, 11, 17, 18, 24, 25, 31},
}


def hardcoded_market_holidays() -> Set[dt.date]:
    return {
        dt.date(year, month, day)
        for (year, month), days in MARKET_HOLIDAYS_BY_MONTH.items()
        for day in days
    }


@dataclass
class GameCalendar:
    holidays: Set[dt.date] = field(default_factory=hardcoded_market_holidays)
    # 차주 계획(수요예측·생산계획)을 확정하는 요일 (0=월 ... 5=토, 6=일)
    # 규칙: 매주 토요일 자정까지 차주 계획 입력 -> 차주 = 다음 일요일 ~ 토요일
    decision_weekday: int = 5

    # --- 기본 판정 --------------------------------------------------------
    def is_weekend(self, d: dt.date) -> bool:
        return d.weekday() >= 5

    def is_market_day(self, d: dt.date) -> bool:
        """주식시장 개장일. 요일 계산이 아니라 MARKET_HOLIDAYS_BY_MONTH(하드코딩)에
        없는 날짜만 개장일로 판정한다."""
        return d not in self.holidays

    def is_production_day(self, d: dt.date, weekdays_only: bool = True) -> bool:
        if not weekdays_only:
            return True
        return not self.is_weekend(d)

    def is_decision_day(self, d: dt.date) -> bool:
        return d.weekday() == self.decision_weekday

    # --- 리드타임 ---------------------------------------------------------
    def arrival_date(self, order_date: dt.date, lead_time_days: int) -> dt.date:
        """발주일 + 리드타임(달력일) = 입고일. 입고 당일 생산 투입 가능."""
        return order_date + dt.timedelta(days=lead_time_days)

    def latest_order_date(self, need_date: dt.date, lead_time_days: int) -> dt.date:
        """need_date에 자재를 쓰려면 늦어도 언제까지 발주해야 하는가."""
        return need_date - dt.timedelta(days=lead_time_days)

    # --- 구간 생성 --------------------------------------------------------
    def date_range(self, start: dt.date, end: dt.date) -> List[dt.date]:
        n = (end - start).days
        return [start + dt.timedelta(days=i) for i in range(n + 1)]

    def next_week(self, decision_date: dt.date) -> List[dt.date]:
        """의사결정일 다음날부터 7일.

        토요일 자정 마감 규칙에서는 토요일 결정 -> 일~토 7일이 되며,
        이는 문제소개의 '카푸치노 수요(일~토)' / '아메리카노 수요(월~금)' 표기와 일치한다.
        """
        start = decision_date + dt.timedelta(days=1)
        return [start + dt.timedelta(days=i) for i in range(7)]

    def decision_day_for(self, d: dt.date) -> dt.date:
        """d가 속한 주의 계획을 제출한 결정일(직전 토요일).

        next_week(S) = S+1 .. S+7 이므로, d를 담는 주의 결정일은 'd보다 앞선 마지막 토요일'이다.
        d 자신이 토요일이면 그 주는 일주일 전 토요일에 제출된 것이다.
        """
        back = (d.weekday() - self.decision_weekday) % 7
        return d - dt.timedelta(days=back or 7)

    def next_decision_day(self, d: dt.date) -> dt.date:
        """d(당일 포함) 이후 첫 결정일 = 다음 주간계획 제출 마감일."""
        return d + dt.timedelta(days=(self.decision_weekday - d.weekday()) % 7)

    def first_production_day_after(self, d: dt.date, weekdays_only: bool = True) -> dt.date:
        cursor = d + dt.timedelta(days=1)
        while not self.is_production_day(cursor, weekdays_only):
            cursor += dt.timedelta(days=1)
        return cursor

    def planning_horizon(self, decision_date: dt.date, weekdays_only: bool = True,
                         availability_lag: int = 1) -> List[dt.date]:
        """차주 7일 + '차차주 생산으로는 도저히 못 대는 앞머리 날짜만큼의 연장분'.

        두 규칙이 겹쳐서 이 연장이 필요해진다.
          (1) 차주는 일~토인데 첫 생산가능일은 월요일 -> 일요일은 그 주 안에서 생산 불가
          (2) 당일 생산분은 당일 판매 불가 -> 월요일 수요도 월요일 생산으로는 못 댄다

        즉 **차차주 월요일 생산으로 처음 감당할 수 있는 날은 화요일**이므로,
        이번 계획의 금요일 배치가 '토 + 다음 일 + 다음 월'까지 미리 만들어 두어야 한다.
        이 연장이 없으면 매주 일·월에 구조적으로 품절이 난다.

        availability_lag = 0 (당일 판매 가능 가정)이면 연장분은 일요일 하루뿐이다.
        """
        week = self.next_week(decision_date)
        next_prod = self.first_production_day_after(week[-1], weekdays_only)
        first_serviceable = next_prod + dt.timedelta(days=availability_lag)
        extra = self.date_range(week[-1] + dt.timedelta(days=1),
                                first_serviceable - dt.timedelta(days=1))
        return week + extra

    def demand_days(self, dates: Iterable[dt.date], market_days_only: bool) -> List[dt.date]:
        if market_days_only:
            return [d for d in dates if self.is_market_day(d)]
        return list(dates)

    def weekday_key(self, d: dt.date) -> str:
        """ProcessingTimeTable 파일명 요일 키 (t_500_20_mon.csv 등)."""
        return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][d.weekday()]
