"""Demand 모듈 - 예측기.

설계 원칙
1. 모든 예측기는 동일 인터페이스를 갖는다: 과거 계열 + 원점 + 시계 -> (mu, sigma).
2. sigma는 옵션이 아니라 필수 산출물이다. 안전재고·Lot sizing·할인정책이 전부
   sigma를 입력으로 받기 때문에, 점예측만 내놓는 예측기는 이 시스템에 꽂을 수 없다.
3. 모든 추정은 origin 이전 데이터만으로 한다 (누수 금지).

채택 모델은 SES(α=0.9) 하나로 확정했다. NaiveForecaster는 그 성능을 재는
기준선으로만 남겨 둔다 (랜덤워크 계열에서 가장 이기기 어려운 벤치마크).
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class ForecastResult:
    """원점 하나에 대한 h=1..H 예측."""

    origin: dt.date
    dates: List[dt.date]          # 예측 대상 날짜 (휴장일 제외 후)
    mu: np.ndarray                # 점예측 (수요 개수)
    sigma: np.ndarray             # h별 예측오차 표준편차
    model: str = ""

    def cumulative_sigma(self, h: int) -> float:
        """h일 누적 수요의 표준편차. 안전재고 계산에 쓰는 값.

        주의: sqrt(sum(sigma^2))는 h별 오차가 독립일 때만 맞다.
        랜덤워크 계열은 오차가 강하게 누적되므로 실제로는 과소추정이다.
        """
        return float(np.sqrt(np.sum(self.sigma[:h] ** 2)))


class Forecaster(ABC):
    """모든 수요예측 모델의 공통 인터페이스."""

    name: str = "base"

    @abstractmethod
    def fit_predict(
        self,
        history: Sequence[float],
        target_dates: List[dt.date],
        origin: dt.date,
    ) -> ForecastResult:
        """history는 origin까지의 관측치(휴장일 제외한 거래 계열)."""

    def _residual_sigma(self, history: Sequence[float], horizon: int) -> np.ndarray:
        diffs = np.diff(np.asarray(history, dtype=float))
        s1 = float(np.std(diffs[-250:])) if len(diffs) else 0.0
        return s1 * np.sqrt(np.arange(1, horizon + 1))


class NaiveForecaster(Forecaster):
    """ŷ(t+h) = y(t). 랜덤워크 계열에서 가장 강한 기준선."""

    name = "naive"

    def fit_predict(self, history, target_dates, origin) -> ForecastResult:
        h = len(target_dates)
        last = float(history[-1])
        return ForecastResult(
            origin=origin,
            dates=list(target_dates),
            mu=np.full(h, last),
            sigma=self._residual_sigma(history, h),
            model=self.name,
        )


class SimpleExponentialSmoothingForecaster(Forecaster):
    """채택 모델. SES - 추세/계절성 없이 레벨만 추정한다.

    L_t = α*y_t + (1-α)*L_{t-1},  ŷ(t+h) = L_t (h와 무관하게 평평한 예측)

    α는 데이터로 재추정하지 않고 0.9로 고정한다. α가 클수록 최근 관측치에
    강하게 반응하는데, 이 시스템 수요는 랜덤워크성이 강해 옛 관측치의 정보량이
    빠르게 소멸하므로 0.9 수준의 높은 반응성이 적합하다.
    """

    name = "ses"
    ALPHA = 0.9

    def fit_predict(self, history, target_dates, origin) -> ForecastResult:
        h = len(target_dates)
        y = np.asarray(history, dtype=float)
        alpha = self.ALPHA

        level = y[0]
        resid = np.empty(len(y) - 1, dtype=float)
        for i, x in enumerate(y[1:]):
            resid[i] = x - level
            level = alpha * x + (1 - alpha) * level

        s1 = float(np.std(resid[-250:])) if resid.size else 0.0
        sigma = s1 * np.sqrt(np.arange(1, h + 1))

        return ForecastResult(
            origin=origin,
            dates=list(target_dates),
            mu=np.full(h, level),
            sigma=sigma,
            model=f"{self.name}(alpha={alpha})",
        )
