"""Demand 모듈 - 실현 수요 (수요 파일 재생) + 로더.

`HistoricalReplay`는 Demand 화면에서 받은 파일을 그대로 재생한다. 라운드가 진행 중이라
미래 구간 실측치가 없을 때는 `extend_with_forecast()`로 점예측을 실현 수요 자리에 깐다
(예측오차 0인 낙관 시나리오).
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence


class DemandGenerator(ABC):
    """실현 수요 경로를 제공한다."""

    @abstractmethod
    def realized(self, date: dt.date, product: str) -> int:
        ...

    @abstractmethod
    def history_until(self, date: dt.date, product: str) -> List[float]:
        """origin 시점까지의 관측 계열 (예측기 입력). 휴장일은 제외한 거래계열."""

    def reset(self, seed: int) -> None:
        """replication 시작 시 호출. 확률적 생성기는 여기서 시드를 받아 CRN을 맞춘다."""


@dataclass
class HistoricalReplay(DemandGenerator):
    """수요 파일 계열을 그대로 재생 (결정론적)."""

    series: Dict[str, Dict[dt.date, int]]   # product -> {date: demand}

    def realized(self, date, product) -> int:
        return int(self.series[product].get(date, 0))

    def history_until(self, date, product) -> List[float]:
        s = self.series[product]
        return [float(v) for d, v in sorted(s.items()) if d <= date and v > 0]


def load_demand_csv(path: str) -> Dict[dt.date, int]:
    """IOG 수요 파일(`DATE,DEMAND_QTY`) 로더. CSV와 엑셀(.xls/.xlsx)을 모두 받는다.

    시스템 Demand 화면의 다운로드 버튼은 라운드/브라우저에 따라 CSV를 주기도 하고
    구형 BIFF `.xls`를 주기도 한다. 확장자만 보고 파서를 고르며, 그 아래 정규화 규칙은
    동일하다 -- 호출부(`_latest_demand_file`)는 확장자를 신경 쓰지 않는다.

    CSV에서 실제 제공 파일 기준으로 두 가지를 흡수해야 한다.
      - 인코딩: 파일마다 BOM 유무가 다르다 (utf-8-sig로 통일).
      - 파일 끝 트레일링 빈 줄(`,`만 있는 행) -> 날짜/수량이 비어 있으면 건너뛴다.
    정렬 순서(최신순/과거순)는 가리지 않는다: dict로 반환하므로 호출부가 정렬한다.
    """
    if path.lower().endswith((".xls", ".xlsx")):
        return _load_demand_excel(path)

    import csv

    out: Dict[dt.date, int] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_date = (row.get("DATE") or "").strip()
            raw_qty = (row.get("DEMAND_QTY") or "").strip()
            if not raw_date or not raw_qty:
                continue
            out[dt.date.fromisoformat(raw_date)] = int(float(raw_qty))
    return out


def _load_demand_excel(path: str) -> Dict[dt.date, int]:
    """`.xls`(BIFF, xlrd 필요) / `.xlsx`(openpyxl) 다운로드본.

    DATE 열은 파일에 따라 문자열('2026-09-27')로도 엑셀 날짜 셀로도 들어온다.
    """
    import pandas as pd

    frame = pd.read_excel(path)
    out: Dict[dt.date, int] = {}
    for raw_date, raw_qty in zip(frame["DATE"], frame["DEMAND_QTY"]):
        if pd.isna(raw_date) or pd.isna(raw_qty):
            continue
        day = pd.Timestamp(raw_date).date()
        out[day] = int(float(raw_qty))
    return out


def load_historical_replay(paths: Dict[str, str]) -> HistoricalReplay:
    """product -> CSV 경로 매핑을 받아 HistoricalReplay를 만든다."""
    return HistoricalReplay(series={p: load_demand_csv(path) for p, path in paths.items()})


def extend_with_forecast(
    replay: HistoricalReplay,
    forecasters: Dict[str, object],
    demand_dates: Dict[str, List[dt.date]],
    all_dates: Sequence[dt.date],
    origin: dt.date,
) -> None:
    """실측 데이터가 끝난 뒤 구간(미래)의 '실현 수요'를 점예측으로 채운다 (in-place).

    HistoricalReplay는 series에 없는 날짜를 전부 수요 0으로 돌려준다. 라운드가 진행 중이라
    미래 구간 실측치가 아직 없는 상태에서 그대로 쓰면 매출도 품절도 0이 되어, Balance 궤적이
    '비용만 쌓인 값'이 된다. 그래서 계획에 쓰는 것과 같은 예측기로 만든 점예측을
    '예측이 맞았을 때'의 기준 시나리오로 깔아 준다.

    즉 이 함수를 쓴 결과는 예측오차가 0인 낙관 시나리오다.

    demand_dates: 제품별 수요 발생 가능일 (P1은 개장일만).
    all_dates   : 미래 구간 전체. demand_dates에 없는 날은 0으로 채운다 (휴장일).
    """
    for code, forecaster in forecasters.items():
        targets = demand_dates.get(code, [])
        if targets:
            history = replay.history_until(origin, code)
            fc = forecaster.fit_predict(history, targets, origin)
            for d, mu in zip(fc.dates, fc.mu):
                # 시스템은 수요를 틱보다 앞서 공시하기도 한다(예: 09-30 20시 틱 전에
                # 이미 그날 P2 수요가 파일에 있다). 그런 실측치를 예측으로 덮으면
                # 알고 있는 값을 일부러 버리는 셈이라, 빈 날짜만 채운다.
                if d not in replay.series[code]:
                    replay.series[code][d] = int(max(0, round(float(mu))))
        for d in all_dates:
            replay.series[code].setdefault(d, 0)

