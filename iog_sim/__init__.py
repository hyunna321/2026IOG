"""iog_sim - IOG(Industrial Optimization Ground) 공급사슬 시뮬레이터.

모듈 구성
    config      게임 규칙 상수 (단가, 비용, BOM, 리드타임)
    calendar    개장일 / 생산가능일 / 라운드 종료일 / 리드타임
    costs       비용 함수 (Makespan -> 인건비, 작업준비비)
    state       WorldState, 발주/생산 오더, 일별 기록
    demand      수요예측기 + 실현수요 생성기
    production  Lot sizing + Job sequencing
    material    BOM 전개(생산계획 -> 자재 소요) + 발주정책
    sales       할인정책
    engine      일별 이벤트 루프
    ledger      비용 집계 / KPI
    experiment  what-if 실험 러너 (CRN, 대응표본 비교)
"""

from .config import BOM, MATERIALS, PRODUCTS, SimConfig  # noqa: F401
from .calendar import GameCalendar  # noqa: F401
from .engine import PolicySet, SimResult, Simulator  # noqa: F401
from .ledger import LedgerSummary, summarize  # noqa: F401
from .experiment import ExperimentResult, daily_input_report, run_experiment  # noqa: F401

__version__ = "0.1.0"
