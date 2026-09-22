"""Material 모듈 - BOM 전개.

생산계획(Lot) -> BOM -> 자재 총소요량.

핵심 규칙
- 리드타임은 달력일(주말·공휴일 포함). 발주일 + LT = 입고일, 입고 당일 생산 투입 가능.
- 따라서 M2(원두, 일반 LT 8일)는 '차주 생산계획을 세우는 시점에는 이미 늦은' 자재다.
  차주 월요일 생산분 원두는 그 전주가 아니라 2주 전에 발주되어 있어야 한다.
  => M2 일반 발주는 '수요예측 기반 롤링 발주', M2 긴급(LT 2일)은 '오차 보정용'으로 역할을 나눈다.

순소요량 계산과 발주 시점 역산은 `material.policy`의 (s,S)·이중조달 정책이
재고 포지션(현재고 + 입고예정)을 직접 보고 처리하므로 여기서는 다루지 않는다.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Mapping

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
