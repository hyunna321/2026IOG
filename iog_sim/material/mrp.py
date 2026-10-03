"""Material 모듈 - BOM 전개.

자재는 **생산일에, 생산량만큼** 소모된다. 그래서 발주 정책에 넘기는 소요량은 수요예측이 아니라
생산계획(Lot)을 BOM으로 전개한 값이어야 한다. 수요 기준으로 넘기면 금요일 큰 배치처럼
'먼저, 덩어리로' 생기는 소요를 발주 정책이 '나중에, 고르게' 필요하다고 보고 그날 자재가 모자란다.

확정 계획은 다음 토요일까지만 있으므로, 그 뒤는 같은 요일의 최근 계획을 반복해 추정한다
(`project_lots`). Lot sizing이 만든 배치 묶음 패턴이 그대로 유지된다.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, Iterable, Mapping

from ..config import LOT_SIZE


def explode_bom(
    production_lots: Mapping[dt.date, Mapping[str, int]],
    bom: Mapping[str, Mapping[str, int]],
) -> Dict[dt.date, Dict[str, int]]:
    """생산계획(날짜 -> 제품 -> Lot수) -> (날짜 -> 자재 -> 총소요량)."""
    out: Dict[dt.date, Dict[str, int]] = {}
    for date, by_product in production_lots.items():
        bucket = out.setdefault(date, {})
        for product, lots in by_product.items():
            units = lots * LOT_SIZE
            for mat, per_unit in bom.get(product, {}).items():
                bucket[mat] = bucket.get(mat, 0) + units * per_unit
    return out


def project_lots(
    planned_lots: Mapping[dt.date, Mapping[str, int]],
    planned_through: dt.date,
    days: Iterable[dt.date],
    producing: Mapping[str, Callable[[dt.date], bool]],
    max_weeks_back: int = 8,
) -> Dict[dt.date, Dict[str, int]]:
    """planned_through까지는 계획 그대로, 그 뒤는 같은 요일·같은 종류의 마지막 계획일 값을 쓴다.

    producing[제품](날짜): 그 제품이 그날 생산할 수 있는 날인가(P1은 개장일만).
    공휴일 주의 월요일(0)을 다음 주 월요일에 복사하지 않도록, 생산 가능 여부가 같은 날만 원본으로 쓴다.
    """
    week = dt.timedelta(days=7)
    out: Dict[dt.date, Dict[str, int]] = {}
    for day in days:
        if day <= planned_through:
            out[day] = dict(planned_lots.get(day, {}))
            continue
        out[day] = {}
        for product, can_produce in producing.items():
            if not can_produce(day):
                continue
            src = day - week
            for _ in range(max_weeks_back):
                if src <= planned_through and can_produce(src):
                    lots = planned_lots.get(src, {}).get(product, 0)
                    if lots:
                        out[day][product] = lots
                    break
                src -= week
    return out
