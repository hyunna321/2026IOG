"""Production 모듈 - Job sequencing (n-job, 20-stage permutation flow shop).

전 공정에서 Job 순서가 고정되므로(매뉴얼 "Job 순서 변경 없음"), 결정변수는
'단 하나의 순열'이다. 목적함수는 Makespan 최소화이며, 비용은 Makespan에
계단식(floor(makespan/100))으로 연동되므로 100시간 경계를 넘기느냐가 실제 이득을 가른다.

규칙별 성격
  - SPT / LTWK : 정렬 한 번으로 끝나는 기준선. Lot sizing DP가 수백 번 호출하는
    planning_sequencer 자리에 쓴다. SPT는 시스템 Auto와 같은 로직이다.
  - NEH        : flow shop 삽입 휴리스틱(Taillard 가속). 300 Job에서 약 0.5초,
    Makespan은 SPT보다 약 10% 짧다. 실제 투입 순서(sequencer)에 쓴다.
  - BestOf     : 여러 규칙을 모두 돌려 Makespan 최소 해를 채택.
  - IteratedGreedy : BestOf 해에서 출발해 파괴·재삽입을 반복 (P1 실제 투입 순서).
    대회형 5주 라운드에 나온 P1 배치 675종에서 makespan 평균 −0.86%, 생산비 배치당 약 0.26M,
    라운드당 약 6.2M 절감(나빠진 배치 0). P2는 순서 영향이 작아(−0.03%) BestOf 유지.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

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

    def job_ids(self) -> List[int]:
        """시스템 입력용 1-based Job ID 순서."""
        return [i + 1 for i in self.order]


class Sequencer(ABC):
    name = "base"

    @abstractmethod
    def solve(self, p: np.ndarray) -> SequenceResult:
        ...


class SPT(Sequencer):
    """기준 설비(기본 M1=0번 공정) 처리시간 오름차순. 시스템 Auto와 동일 로직."""

    name = "SPT"

    def __init__(self, machine: int = 0):
        self.machine = machine

    def solve(self, p) -> SequenceResult:
        order = list(np.argsort(p[:, self.machine], kind="stable"))
        return SequenceResult([int(i) for i in order], makespan(p, order),
                              f"{self.name}(M{self.machine + 1})")


class LTWK(Sequencer):
    """총 작업량(20공정 합) 오름차순."""

    name = "LTWK"

    def solve(self, p) -> SequenceResult:
        order = list(np.argsort(p.sum(axis=1), kind="stable"))
        return SequenceResult([int(i) for i in order], makespan(p, order), self.name)


class NEH(Sequencer):
    """Nawaz-Enscore-Ham + Taillard 가속.

    소박한 NEH는 삽입 위치마다 makespan을 새로 계산해 O(n^3·m)이 되고,
    이 문제의 n(=일 생산 Lot 수, 70~400)에서는 현실적으로 돌지 않는다.
    Taillard 가속은 head(e) / tail(q) / 삽입완료시각(f)을 한 번씩만 계산해 O(n^2·m)로 줄인다.
    """

    name = "NEH"

    def solve(self, p) -> SequenceResult:
        n, m = p.shape
        if n == 0:
            return SequenceResult([], 0.0, self.name)
        seeds = [int(i) for i in np.argsort(-p.sum(axis=1), kind="stable")]
        seq: List[int] = [seeds[0]]
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
                if ms < best_ms:
                    best_ms, best_pos = ms, i
            seq.insert(best_pos, job)
        return SequenceResult(seq, makespan(p, seq), self.name)


def _best_insertion(p: np.ndarray, seq: List[int], job: int):
    """Taillard 가속: 부분수열 seq에 job을 넣을 최적 위치와 그때의 makespan.

    head(e)·tail(q)는 job 순서대로 한 번씩 계산하고, 삽입 위치 스캔은 위치 축으로 벡터화한다.
    """
    sub = p[seq]
    j, m = sub.shape
    e = np.zeros((j + 1, m))          # e[k] = 앞 k개 job이 끝나는 시각 (위치 k의 head)
    q = np.zeros((j + 2, m))          # q[k+1] = k번째부터 끝까지 남은 시간 (위치 k의 tail)
    for i in range(j):
        row, prev, cur, acc = sub[i], e[i], e[i + 1], 0.0
        for r in range(m):
            v = prev[r]
            acc = (v if acc < v else acc) + row[r]
            cur[r] = acc
    for i in range(j - 1, -1, -1):
        row, nxt, cur, acc = sub[i], q[i + 2], q[i + 1], 0.0
        for r in range(m - 1, -1, -1):
            v = nxt[r]
            acc = (v if acc < v else acc) + row[r]
            cur[r] = acc
    pk = p[job]
    f = np.zeros(j + 1)
    best = np.zeros(j + 1)
    for r in range(m):
        f = np.maximum(f, e[:, r]) + pk[r]
        best = np.maximum(best, f + q[1:, r])
    k = int(np.argmin(best))
    return k, float(best[k])


class IteratedGreedy(Sequencer):
    """Ruiz & Stützle(2007) Iterated Greedy. 시작 순서(기본 NEH)에서 d개를 빼고 최적 위치에
    다시 끼우기를 반복, 온도 T로 약간 나쁜 해도 받아들인다. 최선 해만 반환하므로 시작 순서보다 나빠지지 않는다.

    시드와 반복 횟수를 고정해 같은 행렬이면 어느 PC에서나 같은 순서가 나온다 (시간 제한은 쓰지 않는다).
    """

    name = "IG"

    def __init__(self, iterations: int = 400, d: int = 4, seed: int = 0,
                 start: Sequencer = None):
        self.iterations = iterations
        self.d = d
        self.seed = seed
        self.start = start or NEH()

    def solve(self, p) -> SequenceResult:
        n = p.shape[0]
        init = self.start.solve(p)
        if n <= self.d + 1:
            return SequenceResult(init.order, init.makespan, self.name)
        rng = np.random.default_rng(self.seed)
        cur, cur_ms = list(init.order), init.makespan
        best, best_ms = list(cur), cur_ms
        temperature = 0.4 * float(p.sum()) / (n * p.shape[1] * 10)
        for _ in range(self.iterations):
            seq = list(cur)
            removed = [seq.pop(int(rng.integers(len(seq)))) for _ in range(self.d)]
            ms = cur_ms
            for job in removed:
                k, ms = _best_insertion(p, seq, job)
                seq.insert(k, job)
            if ms < cur_ms or rng.random() < np.exp(-(ms - cur_ms) / temperature):
                cur, cur_ms = seq, ms
                if ms < best_ms:
                    best, best_ms = list(seq), ms
        return SequenceResult(best, makespan(p, best), f"{self.name}({self.iterations})")


class BestOf(Sequencer):
    """여러 알고리즘을 모두 돌리고 Makespan 최소 해 채택 (운영 기본값)."""

    name = "BestOf"

    def __init__(self, members: Sequence[Sequencer]):
        self.members = list(members)

    def solve(self, p) -> SequenceResult:
        best = min((m.solve(p) for m in self.members), key=lambda r: r.makespan)
        return SequenceResult(best.order, best.makespan, f"{self.name}[{best.algorithm}]")


class Memoized(Sequencer):
    """같은 처리시간 행렬에 대한 결과를 재사용한다 (여러 시뮬레이터가 공유 가능).

    처리시간표는 (제품, 요일, Job 수)로 정해지므로 정책 비교처럼 같은 행렬이 반복될 때 NEH를 아낀다.
    """

    def __init__(self, inner: Sequencer):
        self.inner = inner
        self.name = inner.name
        self._cache: dict = {}

    def solve(self, p) -> SequenceResult:
        key = (p.shape, p.tobytes())
        if key not in self._cache:
            self._cache[key] = self.inner.solve(p)
        return self._cache[key]


class DiskCached(Sequencer):
    """결과 순서를 JSON 파일에 남겨 다음 실행에서 재사용한다 (IG처럼 느린 sequencer용).

    키는 처리시간 행렬의 해시 + sequencer 이름(반복 횟수 포함)이라, 행렬이나 설정이 바뀌면 다시 계산한다.
    파일이 없거나 깨져 있으면 그냥 계산한다. makespan은 불러올 때 다시 계산해 검산한다.
    """

    def __init__(self, inner: Sequencer, path, label: str):
        self.inner = inner
        self.name = inner.name
        self.path = path
        self.label = label                 # 예: "IG800(BestOf)" -- 설정이 바뀌면 키가 달라지게
        self._mem: dict = {}

    def _load(self) -> dict:
        import json
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def solve(self, p) -> SequenceResult:
        import hashlib, json
        key = f"{self.label}:{p.shape[0]}x{p.shape[1]}:{hashlib.sha1(p.tobytes()).hexdigest()}"
        if key in self._mem:
            return self._mem[key]
        cached = self._load().get(key)
        if cached is not None and sorted(cached) == list(range(p.shape[0])):
            res = SequenceResult(list(cached), makespan(p, cached), self.label)
        else:
            res = self.inner.solve(p)
            data = self._load()
            data[key] = [int(i) for i in res.order]
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except OSError:
                pass
        self._mem[key] = res
        return res


# ---------------------------------------------------------------------------
# 처리시간 테이블 로더
# ---------------------------------------------------------------------------

def load_processing_times(path: str) -> np.ndarray:
    """ProcessingTimeTable의 t_{jobs}_{machines}_{weekday}.csv -> (n_jobs, n_machines) 행렬."""
    import pandas as pd

    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    # JobID 컬럼이 있으면 제외
    cols = [c for c in num.columns if str(c).lower() not in ("jobid", "job_id", "job", "id")]
    return num[cols].to_numpy(dtype=float)
