"""캘린더: 개장일 / 생산가능일 / 의사결정일 / 라운드 종료일 / 리드타임.

규칙
- P1 수요는 주식시장 개장일에만 발생한다. 주말은 자동 휴장, 평일은 하드코딩된 공휴일만 휴장.
- P2 수요는 365일 매일 발생한다.
- 생산가능일은 제품마다 다르다: P1 평일만 / P2 매일 (`ProductSpec.weekday_production_only`).
- 자재 리드타임은 달력일 기준(주말·공휴일 포함). 발주일 + LT = 입고일, 입고 당일 생산 투입 가능.
- 수요예측/생산계획은 매주 토요일 자정 마감, 대상은 차주 일~토.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 평일 공휴일만 적는다. 주말은 is_market_day가 자동으로 휴장 처리한다.
# 확인 방법: Demand 화면에서 P1 입력칸이 없고 0으로 고정된 평일.
MARKET_HOLIDAYS_BY_MONTH: Dict[Tuple[int, int], Set[int]] = {
    (2026, 9): {24, 25},          # 추석 (평일분만. 주말은 is_market_day가 자동 처리)
    (2026, 10): {5, 9},           # 개천절 대체공휴일, 한글날
    # 2026-11, 2026-12(1~4일): 평일 공휴일 없음
}

# 라운드 마지막 날(이 날까지 생산·판매가 정산된다). 다음 날부터 새 라운드가 초기재고로 시작한다.
# VERIFY: 2회차 종료일은 게임 기간 끝(12-04)으로 가정했다. 공지되면 고칠 것.
ROUND_ENDS: List[dt.date] = [
    dt.date(2026, 9, 25),         # 1회차
    dt.date(2026, 12, 4),         # 2회차 (VERIFY)
]


def hardcoded_market_holidays() -> Set[dt.date]:
    return {
        dt.date(year, month, day)
        for (year, month), days in MARKET_HOLIDAYS_BY_MONTH.items()
        for day in days
    }


@dataclass
class GameCalendar:
    holidays: Set[dt.date] = field(default_factory=hardcoded_market_holidays)
    round_ends: List[dt.date] = field(default_factory=lambda: list(ROUND_ENDS))
    # 차주 계획을 확정하는 요일 (0=월 ... 5=토). 토요일 결정 -> 차주 = 일~토
    decision_weekday: int = 5

    # --- 기본 판정 --------------------------------------------------------
    def is_market_day(self, d: dt.date) -> bool:
        """주말은 자동 휴장, 평일은 하드코딩된 공휴일만 휴장."""
        return d.weekday() < 5 and d not in self.holidays

    def is_production_day(self, d: dt.date, weekdays_only: bool = True) -> bool:
        return not weekdays_only or d.weekday() < 5

    def is_decision_day(self, d: dt.date) -> bool:
        return d.weekday() == self.decision_weekday

    def round_end_for(self, d: dt.date) -> Optional[dt.date]:
        """d가 속한 라운드의 마지막 날. 등록된 종료일이 없으면 None(끝없이 계속)."""
        later = [e for e in self.round_ends if e >= d]
        return min(later) if later else None

    # --- 리드타임 ---------------------------------------------------------
    def arrival_date(self, order_date: dt.date, lead_time_days: int) -> dt.date:
        """발주일 + 리드타임(달력일) = 입고일."""
        return order_date + dt.timedelta(days=lead_time_days)

    # --- 구간 생성 --------------------------------------------------------
    def date_range(self, start: dt.date, end: dt.date) -> List[dt.date]:
        return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    def next_week(self, decision_date: dt.date) -> List[dt.date]:
        """결정일 다음날부터 7일. 토요일 결정 -> 일~토."""
        start = decision_date + dt.timedelta(days=1)
        return [start + dt.timedelta(days=i) for i in range(7)]

    def decision_day_for(self, d: dt.date) -> dt.date:
        """d가 속한 주의 계획을 제출한 결정일(d보다 앞선 마지막 토요일)."""
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

    def coverage_horizon(self, days: Sequence[dt.date], weekdays_only: bool = True,
                         availability_lag: int = 1) -> List[dt.date]:
        """생산을 배정할 `days` + 그 뒤 첫 생산으로는 댈 수 없는 앞머리 날짜(연장분).

        P1은 차주 토·일에 생산할 수 없으므로 금요일 배치가 '토 + 다음 일'(lag=0)까지,
        당일 판매 불가(lag=1)라면 '다음 월'까지 커버해야 한다. P2는 매일 생산이라 lag=0에서 연장이 없다.
        """
        days = list(days)
        next_prod = self.first_production_day_after(days[-1], weekdays_only)
        first_serviceable = next_prod + dt.timedelta(days=availability_lag)
        extra = self.date_range(days[-1] + dt.timedelta(days=1),
                                first_serviceable - dt.timedelta(days=1))
        return days + extra

    def planning_horizon(self, decision_date: dt.date, weekdays_only: bool = True,
                         availability_lag: int = 1) -> List[dt.date]:
        """결정일 기준 차주 7일 + 연장분."""
        return self.coverage_horizon(self.next_week(decision_date), weekdays_only,
                                     availability_lag)

    def demand_days(self, dates: Iterable[dt.date], market_days_only: bool) -> List[dt.date]:
        if market_days_only:
            return [d for d in dates if self.is_market_day(d)]
        return list(dates)

    def weekday_key(self, d: dt.date) -> str:
        """ProcessingTimeTable 파일명 요일 키 (t_500_20_mon.csv 등)."""
        return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][d.weekday()]
