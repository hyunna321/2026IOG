"""완제품 안전재고 규칙.

판매기회비(250원/개 + 놓친 마진)가 재고유지비(30원/개·일)보다 훨씬 크므로, 버퍼 수준은
뉴스벤더 임계비율 Cu / (Cu + Co)로 정한다 (`stockout_critical_ratio`).

주간계획은 토요일에 한 주치가 고정되고 주중에는 고칠 수 없다. 그래서 막아야 할 것은
'하루치 오차'가 아니라 **계획 원점부터 h일째까지의 누적 예측오차**다. 규칙은 수요일마다
누적 버퍼 목표 ss_h를 돌려주고, 엔진은 그 증분(ss_h - ss_{h-1})을 그날 수요에 더해
Lot sizing에 넘긴다. 그러면 계획은 매일 '누적 생산 >= 누적 예측 + ss_h'를 만족한다.

| 규칙 | ss_h |
|---|---|
| NoSafetyStock | 0 |
| EndOfHorizon | 마지막 날만 z * σ1 (구 방식) |
| CumulativeModelSigma | z * 예측기 σ의 누적 (sqrt(Σσ_k²)) |
| CumulativeEmpiricalSigma | z * 최근 백테스트의 누적오차 RMS (편향 포함) |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from ..config import ProductSpec
from ..demand.forecaster import ForecastResult, Forecaster


def stockout_critical_ratio(product: ProductSpec, unit_material_cost: float,
                            round_end: bool = False) -> float:
    """뉴스벤더 임계비율 Cu / (Cu + Co).

    Cu (한 개 모자람) = 판매기회비 + 놓친 마진(단가 - 자재비)
    Co (한 개 남음)   = 평소에는 하루 재고유지비(다음 날 팔린다),
                        라운드 종료 직전에는 팔 수 없으므로 자재비 + 재고유지비
    """
    cu = product.stockout_cost + max(0.0, product.price - unit_material_cost)
    co = product.holding_cost + (unit_material_cost if round_end else 0.0)
    return cu / (cu + co)


def backtest_cumulative_rms(forecaster: Forecaster, history: Sequence[float], horizon: int,
                            n_origins: int = 52, step: int = 5, window: int = 500) -> np.ndarray:
    """history 끝에서부터 n_origins개 원점으로 되돌려 예측하고, h일 누적오차의 RMS를 낸다.

    RMS라서 예측 편향(SES는 추세를 못 따라가 한쪽으로 치우친다)까지 버퍼에 반영된다.
    """
    y = np.asarray(history, dtype=float)
    errors = []
    for i in range(n_origins):
        t = len(y) - horizon - i * step
        if t < 30:
            break
        fit = y[max(0, t - window):t]
        fc = forecaster.fit_predict(list(fit), [None] * horizon, None)
        errors.append(np.cumsum(y[t:t + horizon] - fc.mu))
    if not errors:
        return np.zeros(horizon)
    return np.sqrt(np.mean(np.square(errors), axis=0))


class SafetyStockRule(ABC):
    """service_level이 None이면 엔진이 비용 임계비율을 쓴다."""

    name = "base"

    def __init__(self, service_level: Optional[float] = None):
        self.service_level = service_level

    @abstractmethod
    def sigmas(self, fc: ForecastResult, history: Sequence[float],
               forecaster: Forecaster) -> np.ndarray:
        """수요일별 누적오차 표준편차 (len = len(fc.dates))."""

    def cumulative_targets(self, fc: ForecastResult, history: Sequence[float],
                           forecaster: Forecaster, service_level: float) -> np.ndarray:
        """누적 버퍼 목표 ss_h (단조 비감소 정수)."""
        z = float(norm.ppf(service_level))
        ss = np.maximum(0.0, z * self.sigmas(fc, history, forecaster))
        return np.round(np.maximum.accumulate(ss)).astype(int)

    def __repr__(self) -> str:
        sl = "cost" if self.service_level is None else f"{self.service_level:.2f}"
        return f"{self.name}({sl})"


class NoSafetyStock(SafetyStockRule):
    name = "none"

    def sigmas(self, fc, history, forecaster):
        return np.zeros(len(fc.dates))


class EndOfHorizon(SafetyStockRule):
    """구 방식: 1일치 σ의 버퍼를 계획 구간 마지막 수요일에만 얹는다 (주중 버퍼 없음)."""

    name = "end"

    def sigmas(self, fc, history, forecaster):
        out = np.zeros(len(fc.dates))
        if len(out):
            out[-1] = fc.cumulative_sigma(1)
        return out


class CumulativeModelSigma(SafetyStockRule):
    """예측기가 내는 σ로 누적오차를 추정 (SES는 랜덤워크 가정이라 과소추정 경향)."""

    name = "cum-model"

    def sigmas(self, fc, history, forecaster):
        return np.array([fc.cumulative_sigma(h) for h in range(1, len(fc.dates) + 1)])


class CumulativeEmpiricalSigma(SafetyStockRule):
    """최근 n_origins개 원점 백테스트의 누적오차 RMS."""

    name = "cum-empirical"

    def __init__(self, service_level: Optional[float] = None, n_origins: int = 52):
        super().__init__(service_level)
        self.n_origins = n_origins

    def sigmas(self, fc, history, forecaster):
        return backtest_cumulative_rms(forecaster, history, len(fc.dates), self.n_origins)
