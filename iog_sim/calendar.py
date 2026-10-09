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
# 과거분(2016-03~2016-11, 2018-05~2026-09)은 P1 수요 파일(demand_data_P1_20261002)에서
# 수요가 없는 평일을 그대로 옮긴 것이다. 2016-11-24~2018-05-03은 파일 자체가 비어 있어 제외.
MARKET_HOLIDAYS_BY_MONTH: Dict[Tuple[int, int], Set[int]] = {
    # 2016
    (2016, 4): {13},
    (2016, 5): {5, 6},
    (2016, 6): {6},
    (2016, 8): {15},
    (2016, 9): {14, 15, 16},
    (2016, 10): {3},
    # 2018
    (2018, 5): {7, 22},
    (2018, 6): {6, 13},
    (2018, 8): {15},
    (2018, 9): {24, 25, 26},
    (2018, 10): {3, 9},
    (2018, 12): {25, 31},
    # 2019
    (2019, 1): {1},
    (2019, 2): {4, 5, 6},
    (2019, 3): {1},
    (2019, 5): {1, 6},
    (2019, 6): {6},
    (2019, 8): {15},
    (2019, 9): {12, 13},
    (2019, 10): {3, 9},
    (2019, 12): {25, 31},
    # 2020
    (2020, 1): {1, 24, 27},
    (2020, 4): {15, 30},
    (2020, 5): {1, 5},
    (2020, 8): {17},
    (2020, 9): {30},
    (2020, 10): {1, 2, 9},
    (2020, 12): {25, 31},
    # 2021
    (2021, 1): {1},
    (2021, 2): {11, 12},
    (2021, 3): {1},
    (2021, 5): {5, 19},
    (2021, 8): {16},
    (2021, 9): {20, 21, 22},
    (2021, 10): {4, 11},
    (2021, 12): {31},
    # 2022
    (2022, 1): {31},
    (2022, 2): {1, 2},
    (2022, 3): {1, 9},
    (2022, 5): {5},
    (2022, 6): {1, 6},
    (2022, 8): {15},
    (2022, 9): {9, 12},
    (2022, 10): {3, 10},
    (2022, 12): {30},
    # 2023
    (2023, 1): {23, 24},
    (2023, 3): {1},
    (2023, 5): {1, 5, 29},
    (2023, 6): {6},
    (2023, 8): {15},
    (2023, 9): {28, 29},
    (2023, 10): {2, 3, 9},
    (2023, 12): {25, 29},
    # 2024
    (2024, 1): {1},
    (2024, 2): {9, 12},
    (2024, 3): {1},
    (2024, 4): {10},
    (2024, 5): {1, 6, 15},
    (2024, 6): {6},
    (2024, 8): {15},
    (2024, 9): {16, 17, 18},
    (2024, 10): {1, 3, 9},
    (2024, 12): {25, 31},
    # 2025
    (2025, 1): {1, 27, 28, 29, 30},
    (2025, 3): {3},
    (2025, 5): {1, 5, 6},
    (2025, 6): {3, 6},
    (2025, 8): {15},
    (2025, 10): {3, 6, 7, 8, 9},
    (2025, 12): {25, 31},
    # 2026
    (2026, 1): {1},
    (2026, 2): {16, 17, 18},
    (2026, 3): {2},
    (2026, 5): {1, 5, 25},
    (2026, 6): {3},
    (2026, 7): {17},
    (2026, 8): {17},
    (2026, 9): {24, 25},  # 추석
    (2026, 10): {5, 9},  # 개천절 대체공휴일, 한글날
    # 2026-11, 2026-12(1~4일): 평일 공휴일 없음
}

# 라운드 마지막 날(이 날까지 생산·판매가 정산된다). 다음 날부터 새 라운드가 초기재고로 시작한다.
# 대회 일정: 연습 #1 9/13~9/25, 연습 #2 9/27~10/16, 대회 11/1~12/4. 라운드마다 초기화된다.
ROUND_ENDS: List[dt.date] = [
    dt.date(2026, 9, 25),         # 연습 라운드 1 (R2602A)
    dt.date(2026, 10, 16),        # 연습 라운드 2 (R2602B)
    dt.date(2026, 12, 4),         # 대회 라운드 (R2602C, 11/1 시작)
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
