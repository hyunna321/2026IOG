"""what-if 실험 러너.

이 시뮬레이터의 1차 목적은 '정책 비교'다. 따라서 다음 두 가지를 지킨다.

1. Common Random Numbers (CRN)
   replication r에는 모든 정책이 동일한 seed를 쓴다. 그래야 정책 A와 B의 Balance 차이가
   수요 경로 운이 아니라 정책 차이에서 온 것이라고 말할 수 있다.

2. 대응표본 비교 (paired comparison)
   정책 간 비교는 평균 차이가 아니라 replication별 차이 d_r = Balance_A(r) - Balance_B(r)의
   분포로 본다. CRN 덕분에 분산이 크게 줄어 훨씬 적은 replication으로 결론이 난다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from .config import LOT_SIZE
from .engine import PolicySet, SimResult, Simulator


@dataclass
class ExperimentResult:
    rows: List[dict] = field(default_factory=list)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def compare(self, baseline: str, metric: str = "balance"):
        """CRN 대응표본 비교. baseline 대비 각 정책의 (평균 차이, 표준오차, 승률)."""
        import pandas as pd

        df = self.to_frame()
        pivot = df.pivot_table(index="seed", columns="label", values=metric)
        out = []
        for label in pivot.columns:
            if label == baseline:
                continue
            diff = (pivot[label] - pivot[baseline]).dropna()
            out.append({
                "label": label,
                "mean_diff": float(diff.mean()),
                "se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
                "win_rate": float((diff > 0).mean()),
                "n": int(len(diff)),
            })
        return pd.DataFrame(out).sort_values("mean_diff", ascending=False)


def run_experiment(
    build_simulator: Callable[[PolicySet], Simulator],
    policies: Sequence[PolicySet],
    start: dt.date,
    end: dt.date,
    n_replications: int = 30,
    base_seed: int = 1000,
    warmup_days: int = 14,
) -> ExperimentResult:
    result = ExperimentResult()
    for r in range(n_replications):
        seed = base_seed + r          # CRN: 정책마다 같은 seed 재사용
        for pol in policies:
            sim = build_simulator(pol)
            res: SimResult = sim.run(start, end, seed=seed, warmup_days=warmup_days)
            row = res.summary.as_row()
            row.update({"label": pol.label, "seed": seed, "replication": r})
            result.rows.append(row)
    return result


def cost_decomposition(result: ExperimentResult):
    """정책별 비용 항목 분해. '어디서 돈이 새는가'를 먼저 보고 개선 순서를 정한다."""
    df = result.to_frame()
    cols = ["revenue", "production_cost", "inventory_cost", "material_cost", "balance"]
    return df.groupby("label")[cols].mean().sort_values("balance", ascending=False)


def cost_detail(result: ExperimentResult):
    """세부 항목 분해. cost_decomposition의 3개 버킷을 다시 쪼갠 것."""
    df = result.to_frame()
    cols = ["setup_cost", "labor_cost", "overtime_hours", "fg_holding_cost", "stockout_cost",
            "material_order_cost", "material_purchase_cost", "material_holding_cost",
            "units_sold", "units_lost"]
    return df.groupby("label")[cols].mean()


def daily_input_report(result: SimResult, baseline_balance: float = 0.0):
    """일별 예상 Balance + '당일 입력값'만 뽑는 의사결정용 리포트.

    비용 분해표 대신, 매일 실제로 입력/확인해야 하는 값만 남긴다.
      - expected_balance : baseline_balance(현재 실측 Balance)에서 시작해 그날의
        순증감(매출 - 생산비 - 재고비 - 자재비)을 누적한 예상 궤적.
      - order_M1/M3       : 그날 발주할 자재 수량 (일반 조달만 있는 자재).
      - order_M2_normal/order_M2_urgent : M2는 이중조달이라 일반/긴급을 나눠 보여준다
        (리드타임·단가가 달라 시스템 입력 화면도 두 필드로 나뉜다).
      - discount_target   : 그날 정한 할인율이 **적용되는 날**(= date + 1).
      - discount_P1/P2    : 그날 Sales 화면에 입력할 할인율(익일 적용분).

    차주 수요예측·생산계획(Demand/Production 탭)은 매일이 아니라 결정일(토) 한 번에
    제출하는 값이라 이 표에 넣지 않는다 -> `weekly_plan_report()`를 쓸 것.
    """
    import pandas as pd

    state = result.state
    daily_by_date: Dict[dt.date, Dict[str, object]] = {}
    for r in state.daily_records:
        daily_by_date.setdefault(r.date, {})[r.product] = r
    mat_by_date: Dict[dt.date, Dict[str, object]] = {}
    for r in state.material_records:
        mat_by_date.setdefault(r.date, {})[r.material] = r
    order_qty_by_date: Dict[dt.date, Dict[tuple, int]] = {}
    for o in state.open_orders:
        m = order_qty_by_date.setdefault(o.order_date, {})
        key = (o.material, o.option)
        m[key] = m.get(key, 0) + o.qty

    rows = []
    balance = baseline_balance
    for d in sorted(daily_by_date):
        drecs = daily_by_date[d]
        mrecs = mat_by_date.get(d, {})
        revenue = sum(r.revenue for r in drecs.values())
        prod_cost = sum(r.setup_cost + r.labor_cost for r in drecs.values())
        fg_hold = sum(r.fg_holding_cost for r in drecs.values())
        stockout = sum(r.stockout_cost for r in drecs.values())
        mat_cost = sum(m.order_cost + m.purchase_cost + m.holding_cost for m in mrecs.values())
        balance += revenue - prod_cost - fg_hold - stockout - mat_cost

        # d일에 결정 -> d+1일 적용. Sales 화면 입력 시점(=d)에 맞춰 d+1일 적용분을 싣는다.
        target = d + dt.timedelta(days=1)
        disc = state.discount_plan.get(target, {})
        oq = order_qty_by_date.get(d, {})

        rows.append({
            "date": d,
            "expected_balance": int(round(balance)),
            "order_M1": oq.get(("M1", "normal"), 0),
            "order_M2_normal": oq.get(("M2", "normal"), 0),
            "order_M2_urgent": oq.get(("M2", "urgent"), 0),
            "order_M3": oq.get(("M3", "normal"), 0),
            "discount_target": target,
            "discount_P1": disc.get("P1", 0.0),
            "discount_P2": disc.get("P2", 0.0),
        })
    return pd.DataFrame(rows)


def weekly_plan_report(result: SimResult, dates: Sequence[dt.date], sequence_fn=None):
    """결정일(토) 자정에 제출하는 차주 계획 - Demand + Production 탭.

    일일 입력(Material/Sales)과 달리 이 값들은 한 주치를 한 번에 제출한다.
    제출 시점은 `dates`가 속한 주의 직전 토요일이며(`GameCalendar.decision_day_for`),
    그 시점이 지났으면 더 이상 바꿀 수 없다.

      - forecast_P1/P2 : 그날 수요예측치 (Demand 탭). 휴장일은 0.
      - P1_jobs/P2_jobs, *_sequence : 그날 생산할 Lot(Job) 수와 투입 순서 (Production 탭).
      - P1_short/P2_short : 계획 Job 중 자재가 모자라 시뮬레이션에서 못 돌린 Job 수.
        0이 아니면 그날 자재가 부족하다는 경고이지 제출값이 아니다.

    sequence_fn(code, date, lots) -> List[int]: 시퀀스가 아직 계산되지 않은 날
    (자재 부족으로 생산이 실행되지 않은 날 등)을 채우는 데 쓴다.

    주의: `_produce`는 자재가 모자라면 가능한 Lot만 돌리고 job_sequence를 그 길이로
    덮어쓴다(n_lots는 계획값 그대로). 그 시퀀스를 그대로 내보내면 제출할 Job 수와
    시퀀스 길이가 어긋나므로, 길이가 다르면 계획 Job 수 기준으로 다시 만든다.
    """
    import pandas as pd

    state = result.state
    # 실제로 돌아간 Lot 수 (자재 부족 판정용). DailyRecord.produced는 개수 단위다.
    produced = {(r.date, r.product): r.produced // LOT_SIZE for r in state.daily_records}
    rows = []
    for d in dates:
        row = {"date": d, "weekday": d.strftime("%a")}
        for code in ("P1", "P2"):
            row[f"forecast_{code}"] = int(state.forecast_cache.get(code, {}).get(d, 0))
            order = state.production_plan.get(d, {}).get(code)
            lots = order.n_lots if order else 0
            seq = order.job_sequence if order else None
            if lots and len(seq or []) != lots and sequence_fn is not None:
                seq = sequence_fn(code, d, lots)
            row[f"{code}_jobs"] = lots
            row[f"{code}_sequence"] = seq or []
            row[f"{code}_short"] = max(0, lots - int(produced.get((d, code), 0)))
        rows.append(row)
    # 라운드 마지막 주처럼 dates가 비어도 열은 남겨 둔다 (CSV 저장·시퀀스 출력이 열 이름을 찾는다)
    columns = ["date", "weekday"] + [f"{kind}_{code}" if kind == "forecast" else f"{code}_{kind}"
                                     for code in ("P1", "P2")
                                     for kind in ("forecast", "jobs", "sequence", "short")]
    return pd.DataFrame(rows, columns=columns)


def material_flow_report(result: SimResult):
    """Material inventory in/out summary 화면과 같은 단위의 일별 예상값.

    M*_in = 그날 입고, M*_out = 그날 생산 투입, M*_level = 그날 마감 재고(Current level).
    다음 날 화면 값과 나란히 놓고 대조하는 용도다.
    """
    import pandas as pd

    rows: Dict[dt.date, dict] = {}
    for r in result.state.material_records:
        row = rows.setdefault(r.date, {"date": r.date})
        row[f"{r.material}_in"] = r.received
        row[f"{r.material}_out"] = r.consumed
        row[f"{r.material}_level"] = r.inventory_end
    return pd.DataFrame([rows[d] for d in sorted(rows)])


def sales_flow_report(result: SimResult):
    """Sales inventory in/out summary 화면과 같은 단위의 일별 예상값.

    P*_in = 그날 생산 입고, P*_out = 그날 판매, P*_level = 마감 재고, P*_lost = 품절 수량.
    """
    import pandas as pd

    rows: Dict[dt.date, dict] = {}
    for r in result.state.daily_records:
        row = rows.setdefault(r.date, {"date": r.date})
        row[f"{r.product}_in"] = r.produced
        row[f"{r.product}_out"] = r.sold
        row[f"{r.product}_level"] = r.fg_inventory_end
        row[f"{r.product}_lost"] = r.lost_sales
    return pd.DataFrame([rows[d] for d in sorted(rows)])
