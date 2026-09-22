"""Demand 모듈 - 실현 수요 생성기 (what-if 실험의 핵심).

정책 실험에서 중요한 것은 '예측이 맞았을 때'가 아니라 '틀렸을 때 얼마나 손해인가'다.
따라서 시뮬레이터는 반드시 두 개의 계열을 동시에 만들어야 한다.

    realized(d)  : 실제로 발생한 수요
    forecast(o,h): 원점 o에서 본 예측

세 가지 모드를 제공한다.

| 모드 | 실현 수요 | 용도 |
|---|---|---|
| HistoricalReplay | 실제 과거 종가 계열 재생 | 가장 정직. 룰 검증·과거 재현 |
| BlockBootstrapPath | 과거 로그수익률 블록 재표집 | 표본 확장. 변동성 군집 보존 |
| ErrorInjection | forecast * (1+e), e~실측 오차분포 | 가장 빠름. 정책 민감도 스윕용 |

라운드가 진행 중이라 미래 구간 실측치가 없을 때는 `extend_with_forecast()`로
점예측을 실현 수요 자리에 깔아 쓴다 (예측오차 0인 낙관 시나리오).

Common Random Numbers: 동일 replication index에는 동일 시드를 주어, 정책 간 비교에서
수요 경로 차이가 아니라 정책 차이만 드러나게 한다.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


class DemandGenerator(ABC):
    """실현 수요 경로를 제공한다."""

    @abstractmethod
    def realized(self, date: dt.date, product: str) -> int:
        ...

    @abstractmethod
    def history_until(self, date: dt.date, product: str) -> List[float]:
        """origin 시점까지의 관측 계열 (예측기 입력). 휴장일은 제외한 거래계열."""

    def reset(self, seed: int) -> None:
        """replication 시작 시 호출. CRN을 위해 시드를 외부에서 주입."""
        self._rng = np.random.default_rng(seed)


@dataclass
class HistoricalReplay(DemandGenerator):
    """실제 삼성전자 종가 / BTC 종가 계열을 그대로 재생."""

    series: Dict[str, Dict[dt.date, int]]   # product -> {date: demand}

    def realized(self, date, product) -> int:
        return int(self.series[product].get(date, 0))

    def history_until(self, date, product) -> List[float]:
        s = self.series[product]
        return [float(v) for d, v in sorted(s.items()) if d <= date and v > 0]

    def reset(self, seed: int) -> None:  # 결정론적이므로 시드 무시
        self._rng = np.random.default_rng(seed)


def load_demand_csv(path: str) -> Dict[dt.date, int]:
    """IOG 수요 CSV(`DATE,DEMAND_QTY`) 로더.

    실제 제공 파일 기준으로 두 가지를 흡수해야 한다.
      - 인코딩: 파일마다 BOM 유무가 다르다 (utf-8-sig로 통일).
      - 파일 끝 트레일링 빈 줄(`,`만 있는 행) -> 날짜/수량이 비어 있으면 건너뛴다.
    정렬 순서(최신순/과거순)는 가리지 않는다: dict로 반환하므로 호출부가 정렬한다.
    """
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

    즉 이 함수를 쓴 결과는 예측오차가 0인 낙관 시나리오다. 오차 리스크까지 보려면
    ErrorInjection/BlockBootstrapPath로 여러 경로를 뽑아 비교해야 한다.

    demand_dates: 제품별 수요 발생 가능일 (P1은 개장일만).
    all_dates   : 미래 구간 전체. demand_dates에 없는 날은 0으로 채운다 (휴장일).
    """
    for code, forecaster in forecasters.items():
        targets = demand_dates.get(code, [])
        if targets:
            history = replay.history_until(origin, code)
            fc = forecaster.fit_predict(history, targets, origin)
            for d, mu in zip(fc.dates, fc.mu):
                replay.series[code][d] = int(max(0, round(float(mu))))
        for d in all_dates:
            replay.series[code].setdefault(d, 0)


@dataclass
class BlockBootstrapPath(DemandGenerator):
    """과거 로그수익률을 블록 단위로 재표집해 새 경로 생성.

    블록 길이 L은 변동성 군집(ARCH 효과)을 보존할 만큼 길게 잡는다(L=20 권장).
    """

    base_series: Dict[str, Sequence[float]]
    start_date: dt.date
    n_days: int
    block_len: int = 20

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._paths = {p: self._simulate(p) for p in self.base_series}

    def _simulate(self, product: str) -> np.ndarray:
        y = np.asarray(self.base_series[product], dtype=float)
        rets = np.diff(np.log(y))
        n_blocks = int(np.ceil(self.n_days / self.block_len))
        starts = self._rng.integers(0, len(rets) - self.block_len, size=n_blocks)
        path_rets = np.concatenate([rets[s:s + self.block_len] for s in starts])[: self.n_days]
        return y[-1] * np.exp(np.cumsum(path_rets))

    def realized(self, date, product) -> int:
        i = (date - self.start_date).days
        if i < 0 or i >= self.n_days:
            return 0
        return int(self._paths[product][i])

    def history_until(self, date, product) -> List[float]:
        i = (date - self.start_date).days
        base = list(np.asarray(self.base_series[product], dtype=float))
        return base + list(self._paths[product][: max(0, i + 1)])


@dataclass
class ErrorInjection(DemandGenerator):
    """예측값은 주어진 대로 쓰고, 실현치만 오차분포에서 뽑는다.

    가장 빠르고, 정책 파라미터(안전계수 z, lot 규칙, 할인율)를 넓게 스윕할 때 적합.
    error_quantiles: 백테스트에서 얻은 h별 상대오차 분포 (h -> 표본 배열).
    """

    planned_forecast: Dict[str, Dict[dt.date, int]]
    error_samples: Dict[str, Dict[int, np.ndarray]]   # product -> h -> 상대오차 표본
    origin_of: Dict[dt.date, int]                     # date -> 그 날의 h (1..7)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[tuple, int] = {}

    def realized(self, date, product) -> int:
        key = (date, product)
        if key in self._cache:
            return self._cache[key]
        base = self.planned_forecast[product].get(date, 0)
        if base <= 0:
            self._cache[key] = 0
            return 0
        h = self.origin_of.get(date, 1)
        pool = self.error_samples[product].get(h)
        e = float(self._rng.choice(pool)) if pool is not None and len(pool) else 0.0
        val = max(0, int(round(base * (1.0 + e))))
        self._cache[key] = val
        return val

    def history_until(self, date, product) -> List[float]:
        return [float(v) for d, v in sorted(self.planned_forecast[product].items()) if d <= date]
