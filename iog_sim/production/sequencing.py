"""Production 모듈 - Job sequencing (n-job, 20-stage permutation flow shop).

전 공정에서 Job 순서가 고정되므로(매뉴얼 "Job 순서 변경 없음"), 결정변수는
'단 하나의 순열'이다. 목적함수는 Makespan 최소화이며, 비용은 Makespan에
계단식(floor(makespan/100))으로 연동되므로 100시간 경계를 넘기느냐가 실제 이득을 가른다.

규칙별 성격
  - FCFS/SPT/LTWK/MWKR : 정렬 한 번으로 끝나는 기준선. Lot sizing DP가 수백 번
    호출하는 planning_sequencer 자리에는 이 중 하나를 쓴다.
  - NEH                : flow shop 전용 삽입 휴리스틱(Taillard 가속). 실제 투입
    순서를 정하는 sequencer 자리에 쓸 만큼 좋고, 그만큼 느리다.
  - BestOf             : 여러 규칙을 모두 돌려 Makespan 최소 해를 채택.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Makespan 평가기
# ---------------------------------------------------------------------------

def makespan(p: np.ndarray, order: Sequence[int]) -> float:
    """permutation flow shop makespan.

    p     : (n_jobs, n_machines) 처리시간 행렬
    order : 0-based job index 순열
    """
    m = p.shape[1]
    c = np.zeros(m, dtype=float)
    for job in order:
        row = p[job]
        c[0] += row[0]
        for j in range(1, m):
            c[j] = max(c[j], c[j - 1]) + row[j]
    return float(c[-1])


# ---------------------------------------------------------------------------
# Sequencer 인터페이스
# ---------------------------------------------------------------------------

@dataclass
class SequenceResult:
    order: List[int]          # 0-based
    makespan: float
    algorithm: str
    evaluations: int = 0

    def job_ids(self) -> List[int]:
        """시스템 입력용 1-based Job ID 순서."""
        return [i + 1 for i in self.order]


class Sequencer(ABC):
    name = "base"

    @abstractmethod
    def solve(self, p: np.ndarray, time_budget: float = 0.0) -> SequenceResult:
        ...


class FCFS(Sequencer):
    """Job ID 순서 그대로. 최하위 baseline."""

    name = "FCFS"

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        order = list(range(p.shape[0]))
        return SequenceResult(order, makespan(p, order), self.name, 1)


class SPT(Sequencer):
    """기준 설비(기본 M1=0번 공정) 처리시간 오름차순. 시스템 Auto와 동일 로직."""

    name = "SPT"

    def __init__(self, machine: int = 0):
        self.machine = machine

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        order = list(np.argsort(p[:, self.machine], kind="stable"))
        return SequenceResult([int(i) for i in order], makespan(p, order),
                              f"{self.name}(M{self.machine + 1})", 1)


class LTWK(Sequencer):
    """총 작업량(20공정 합) 오름차순."""

    name = "LTWK"

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        order = list(np.argsort(p.sum(axis=1), kind="stable"))
        return SequenceResult([int(i) for i in order], makespan(p, order), self.name, 1)


class MWKR(Sequencer):
    """총 작업량 내림차순 (LTWK의 역순)."""

    name = "MWKR"

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        order = list(np.argsort(-p.sum(axis=1), kind="stable"))
        return SequenceResult([int(i) for i in order], makespan(p, order), self.name, 1)


class NEH(Sequencer):
    """Nawaz-Enscore-Ham + Taillard 가속.

    소박한 NEH는 삽입 위치마다 makespan을 새로 계산해 O(n^3·m)이 되고,
    이 문제의 n(=일 생산 Lot 수, 70~400)에서는 현실적으로 돌지 않는다.
    Taillard 가속은 head(e) / tail(q) / 삽입완료시각(f)을 한 번씩만 계산해 O(n^2·m)로 줄인다.
    """

    name = "NEH"

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        n, m = p.shape
        if n == 0:
            return SequenceResult([], 0.0, self.name, 0)
        seeds = [int(i) for i in np.argsort(-p.sum(axis=1), kind="stable")]
        seq: List[int] = [seeds[0]]
        evals = 0
        for job in seeds[1:]:
            j = len(seq)
            sub = p[seq]                              # (j, m)

            # head: e[i, r] = 부분수열 i번째 job이 설비 r에서 끝나는 시각
            e = np.empty((j, m))
            prev = np.zeros(m)
            for i in range(j):
                acc = 0.0
                row = sub[i]
                for r in range(m):
                    acc = (prev[r] if acc < prev[r] else acc) + row[r]
                    e[i, r] = acc
                prev = e[i]

            # tail: q[i, r] = i번째 job을 설비 r에서 시작한 뒤 남은 총 소요시간
            q = np.empty((j, m))
            nxt = np.zeros(m)
            for i in range(j - 1, -1, -1):
                acc = 0.0
                row = sub[i]
                for r in range(m - 1, -1, -1):
                    acc = (nxt[r] if acc < nxt[r] else acc) + row[r]
                    q[i, r] = acc
                nxt = q[i]

            pk = p[job]
            best_pos, best_ms = 0, float("inf")
            for i in range(j + 1):
                acc = 0.0
                ms = 0.0
                for r in range(m):
                    head = e[i - 1, r] if i > 0 else 0.0
                    acc = (head if acc < head else acc) + pk[r]
                    tail = q[i, r] if i < j else 0.0
                    val = acc + tail
                    if val > ms:
                        ms = val
                evals += 1
                if ms < best_ms:
                    best_ms, best_pos = ms, i
            seq.insert(best_pos, job)
        return SequenceResult(seq, makespan(p, seq), self.name, evals)


class BestOf(Sequencer):
    """여러 알고리즘을 모두 돌리고 Makespan 최소 해 채택 (운영 기본값)."""

    name = "BestOf"

    def __init__(self, members: Sequence[Sequencer]):
        self.members = list(members)

    def solve(self, p, time_budget=0.0) -> SequenceResult:
        share = time_budget / max(len(self.members), 1)
        results = [m.solve(p, share) for m in self.members]
        best = min(results, key=lambda r: r.makespan)
        return SequenceResult(best.order, best.makespan,
                              f"{self.name}[{best.algorithm}]",
                              sum(r.evaluations for r in results))


DEFAULT_SEQUENCER = BestOf([FCFS(), SPT(), LTWK(), MWKR(), NEH()])


# ---------------------------------------------------------------------------
# 처리시간 테이블 로더
# ---------------------------------------------------------------------------

def load_processing_times(path: str, n_jobs: Optional[int] = None) -> np.ndarray:
    """ProcessingTimeTable.zip 의 t_{jobs}_{machines}_{weekday}.csv 로딩.

    n_jobs가 주어지면 JobID 1..n_jobs만 잘라 쓴다 (생산량 n Lot -> Job 1..n).
    """
    import pandas as pd

    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    # JobID 컬럼이 있으면 제외
    cols = [c for c in num.columns if str(c).lower() not in ("jobid", "job_id", "job", "id")]
    p = num[cols].to_numpy(dtype=float)
    return p[:n_jobs] if n_jobs else p
