# -*- coding: utf-8 -*-
"""
================================================================================
 P1 수요 데이터 탐색적 분석(EDA) 스크립트  —  수요예측 모델링 전(前)단계
================================================================================
 목적 : 모델링(ARIMA/SARIMA, Prophet, ML 등) 이전에 데이터의 구조를 파악한다.
 구성 : 0) 데이터 로드 / 품질 점검 / 전처리
        1) 시간 흐름 시각화
        2) 트렌드 분석
        3) 계절성 / 주기성 분석
        4) 정상성 검정
        5) 변동성 확인
        6) 결과 저장 (PNG / CSV)

 실행 : python eda_demand_P1.py
        python eda_demand_P1.py --input ./demand_data_P1_20260914.csv --outdir ./eda_output_P1

 필요 패키지 : pandas, numpy, scipy, statsmodels, matplotlib
        pip install pandas numpy scipy statsmodels matplotlib
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 PNG 저장 가능
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy import stats

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG  — P1 / P2 간 차이는 이 블록에만 존재한다
# ==============================================================================
PRODUCT = "P1"
INPUT_PATH = "C:/머신러닝/IOG/Demand/demand_data_P1_20260914.csv"
FREQ = "B"                    # "B"=영업일(월~금) / "D"=매일
SEASONAL_PERIODS = [5, 252]   # 후보 주기: 주간(5영업일), 연간(252영업일)
# ==============================================================================

DATE_COL = "DATE"
VALUE_COL = "DEMAND_QTY"
IS_BUSINESS_DAY = (FREQ == "B")
PERIODS_PER_YEAR = SEASONAL_PERIODS[-1]


def detrend_window(n: int) -> int:
    """추세 제거용 중심 이동평균 창 길이(데이터가 짧으면 자동 축소)."""
    return int(min(PERIODS_PER_YEAR, max(SEASONAL_PERIODS[0] * 2, n // 6)))



_PV = tuple(int(x) for x in pd.__version__.split(".")[:2])
MONTH_END = "ME" if _PV >= (2, 2) else "M"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------------------
# 유틸: 한글 폰트 설정 (없으면 자동으로 영문 라벨 사용)
# ------------------------------------------------------------------------------
USE_KO = False


def setup_font() -> None:
    """시스템에 한글 폰트가 있으면 사용하고, 없으면 영문 라벨로 전환한다."""
    global USE_KO
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                  "Noto Sans CJK KR", "Noto Sans KR", "Pretendard", "D2Coding"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            USE_KO = True
            print(f"[FONT] 한글 폰트 사용: {name}")
            return
    print("[FONT] 한글 폰트를 찾지 못해 그래프 라벨을 영문으로 출력합니다.")


def T(ko: str, en: str) -> str:
    """폰트 가용 여부에 따라 한/영 라벨을 선택."""
    return ko if USE_KO else en


# ------------------------------------------------------------------------------
# 유틸: 저장 헬퍼
# ------------------------------------------------------------------------------
class Out:
    def __init__(self, outdir: str):
        self.fig_dir = os.path.join(outdir, "figures")
        self.tab_dir = os.path.join(outdir, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)

    def fig(self, fig, name: str) -> None:
        path = os.path.join(self.fig_dir, f"{PRODUCT}_{name}.png")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PNG] {path}")

    def tab(self, df: pd.DataFrame, name: str, index: bool = True) -> None:
        path = os.path.join(self.tab_dir, f"{PRODUCT}_{name}.csv")
        df.to_csv(path, index=index, encoding="utf-8-sig")
        print(f"  [CSV] {path}")


# ==============================================================================
# 0. 데이터 로드 / 품질 점검 / 전처리
# ==============================================================================
def load_and_clean(path: str, out: Out):
    """CSV를 읽어 결측·중복·빈행을 정리하고, 균일 간격 시계열로 재색인한다."""
    print("\n[0] 데이터 로드 및 전처리")
    raw = pd.read_csv(path)
    raw.columns = [c.strip().upper() for c in raw.columns]
    n_raw = len(raw)

    # 빈 행 / 결측 제거
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw[VALUE_COL] = pd.to_numeric(raw[VALUE_COL], errors="coerce")
    df = raw.dropna(subset=[DATE_COL, VALUE_COL]).copy()
    n_empty = n_raw - len(df)

    # 중복 날짜 처리 (마지막 값 유지)
    n_dup = int(df.duplicated(subset=[DATE_COL]).sum())
    df = df.drop_duplicates(subset=[DATE_COL], keep="last")

    # 정렬 및 인덱싱
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    s_obs = df[VALUE_COL].astype(float)

    n_nonpos = int((s_obs <= 0).sum())

    # 균일 간격(FREQ)으로 재색인 → 달력 상 비어 있는 날짜 파악
    if IS_BUSINESS_DAY:
        full_idx = pd.bdate_range(s_obs.index.min(), s_obs.index.max())
    else:
        full_idx = pd.date_range(s_obs.index.min(), s_obs.index.max(), freq="D")
    s_raw = s_obs.reindex(full_idx)
    s_raw.index.name = DATE_COL

    missing_dates = s_raw[s_raw.isna()].index
    # 주말/공휴일 등 결측은 시간 보간으로 채워 분석용 시리즈를 만든다
    s = s_raw.interpolate(method="time").ffill().bfill()
    s.name = VALUE_COL

    # 관측 자체가 FREQ 밖에 있는 날짜(예: 영업일 기준인데 주말 데이터)
    off_freq = s_obs.index.difference(full_idx)

    quality = pd.DataFrame(
        {
            "항목": [
                "원본 행 수", "빈 행/파싱 실패 제거", "중복 날짜 제거", "유효 관측치 수",
                "관측 시작일", "관측 종료일", "재색인 후 기간 길이(FREQ=%s)" % FREQ,
                "달력상 결측 일수(보간 처리)", "결측 비율(%)", "0 이하 값 개수",
                "FREQ 밖 관측치(주말 등)",
            ],
            "값": [
                n_raw, n_empty, n_dup, len(s_obs),
                str(s_obs.index.min().date()), str(s_obs.index.max().date()), len(s_raw),
                len(missing_dates), round(100 * len(missing_dates) / max(len(s_raw), 1), 2),
                n_nonpos, len(off_freq),
            ],
        }
    )
    print(quality.to_string(index=False))
    out.tab(quality, "00_data_quality", index=False)

    if len(missing_dates) > 0:
        out.tab(pd.DataFrame({"missing_date": missing_dates.astype(str)}),
                "00_missing_dates", index=False)

    clean = pd.DataFrame({VALUE_COL: s, "is_imputed": s_raw.isna().astype(int)})
    out.tab(clean, "00_clean_series")
    return s, s_raw


# ==============================================================================
# 1. 기초 통계 + 시간 흐름 시각화
# ==============================================================================
def basic_profile(s: pd.Series, out: Out) -> None:
    print("\n[1] 기초 통계 및 시간 흐름 시각화")

    desc = s.describe()
    extra = pd.Series(
        {
            "skew(왜도)": float(stats.skew(s.values)),
            "kurtosis(첨도)": float(stats.kurtosis(s.values)),
            "CV(변동계수)": float(s.std() / s.mean()),
            "range": float(s.max() - s.min()),
            "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
            "first_value": float(s.iloc[0]),
            "last_value": float(s.iloc[-1]),
            "total_growth_%": float(100 * (s.iloc[-1] / s.iloc[0] - 1)),
        }
    )
    summary = pd.concat([desc, extra]).to_frame("value")
    print(summary.to_string())
    out.tab(summary, "01_basic_summary")

    # 연도별 / 월별 집계
    by_year = s.groupby(s.index.year).agg(["count", "mean", "std", "min", "max", "sum"])
    by_year.index.name = "year"
    out.tab(by_year, "01_stats_by_year")

    monthly = s.resample(MONTH_END).mean()
    by_ym = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month,
                          "mean_demand": monthly.values})
    out.tab(by_ym.pivot(index="year", columns="month", values="mean_demand"),
            "01_monthly_mean_pivot")

    # --- Figure: 시간 흐름 개요 -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(s.index, s.values, lw=0.8, color="#4C78A8", label=T("실측", "Actual"))
    ax.plot(s.index, s.rolling(30, min_periods=1).mean(), lw=1.6, color="#F58518",
            label=T("이동평균 30", "MA 30"))
    ax.plot(s.index, s.rolling(90, min_periods=1).mean(), lw=1.8, color="#E45756",
            label=T("이동평균 90", "MA 90"))
    ax.set_title(T(f"{PRODUCT} 전체 수요 추이", f"{PRODUCT} Full demand series"))
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(s.index, s.values, lw=0.8, color="#54A24B")
    ax.set_yscale("log")
    ax.set_title(T("로그 스케일 (증가율 관점)", "Log scale (growth view)"))

    ax = axes[1, 0]
    for y, g in s.groupby(s.index.year):
        ax.plot(g.index.dayofyear, g.values, lw=1.0, alpha=0.85, label=str(y))
    ax.set_xlabel(T("연중 일자 (1~366)", "Day of year"))
    ax.set_title(T("연도별 겹쳐 보기", "Year-over-year overlay"))
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    recent = s.iloc[-180:]
    ax.plot(recent.index, recent.values, marker="o", ms=2.5, lw=1.0, color="#B279A2")
    ax.plot(recent.index, recent.rolling(7, min_periods=1).mean(), lw=1.8, color="#333333",
            label=T("이동평균 7", "MA 7"))
    ax.set_title(T("최근 180개 구간 확대", "Last 180 observations"))
    ax.legend(fontsize=8)
    for a in axes.ravel():
        a.tick_params(axis="x", labelsize=8)

    fig.suptitle(T(f"[{PRODUCT}] 1. 시간 흐름 시각화", f"[{PRODUCT}] 1. Time series overview"),
                 fontsize=13)
    out.fig(fig, "01_timeseries_overview")


# ==============================================================================
# 2. 트렌드 분석
# ==============================================================================
def trend_analysis(s: pd.Series, out: Out) -> None:
    print("\n[2] 트렌드 분석")

    t = np.arange(len(s), dtype=float)
    y = s.values.astype(float)

    # 선형 / 2차 추세 적합
    lin = stats.linregress(t, y)
    quad = np.polyfit(t, y, 2)
    quad_fit = np.polyval(quad, t)
    ss_res = float(np.sum((y - quad_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    quad_r2 = 1 - ss_res / ss_tot

    # 로그 선형(지수 성장률) 적합
    log_lin = stats.linregress(t, np.log(np.clip(y, 1e-9, None)))

    # Mann-Kendall 대용: Spearman 상관으로 단조 추세 검정
    rho, rho_p = stats.spearmanr(t, y)

    trend_tbl = pd.DataFrame(
        {
            "지표": ["선형 기울기(일당)", "선형 R^2", "선형 p-value",
                     "2차항 계수", "2차 추세 R^2",
                     "로그선형 일간 성장률(%)", "로그선형 연환산 성장률(%)",
                     "Spearman rho", "Spearman p-value"],
            "값": [lin.slope, lin.rvalue ** 2, lin.pvalue,
                   quad[0], quad_r2,
                   100 * (np.exp(log_lin.slope) - 1),
                   100 * (np.exp(log_lin.slope * PERIODS_PER_YEAR) - 1),
                   rho, rho_p],
        }
    )
    print(trend_tbl.to_string(index=False))
    out.tab(trend_tbl, "02_trend_regression", index=False)

    # LOWESS 평활
    frac = min(0.25, max(0.05, 200 / len(s)))
    low = lowess(y, t, frac=frac, return_sorted=False)

    # 성장률: 월평균 기준 MoM / YoY
    monthly = s.resample(MONTH_END).mean()
    growth = pd.DataFrame({
        "monthly_mean": monthly,
        "MoM_%": monthly.pct_change() * 100,
        "YoY_%": monthly.pct_change(12) * 100,
    })
    out.tab(growth, "02_growth_rates")

    # CUSUM: 평균 대비 누적 편차 → 구조 변화 시점 탐색
    z = (y - y.mean()) / y.std()
    cusum = np.cumsum(z)
    break_idx = int(np.argmax(np.abs(cusum)))
    print(f"  CUSUM 최대 이탈 시점(구조 변화 후보): {s.index[break_idx].date()}")

    out.tab(pd.DataFrame({"t": t, "value": y, "lowess": low, "linear_fit": lin.intercept + lin.slope * t,
                          "quad_fit": quad_fit, "cusum": cusum}, index=s.index), "02_trend_components")

    # --- Figure -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(s.index, y, lw=0.7, color="#BBBBBB", label=T("실측", "Actual"))
    ax.plot(s.index, low, lw=2.0, color="#E45756", label="LOWESS")
    ax.plot(s.index, lin.intercept + lin.slope * t, lw=1.5, ls="--", color="#4C78A8",
            label=T("선형 추세", "Linear"))
    ax.plot(s.index, quad_fit, lw=1.5, ls=":", color="#54A24B", label=T("2차 추세", "Quadratic"))
    ax.set_title(T("추세선 적합", "Trend fits"))
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for w, c in zip([7, 30, 90, 180], ["#9ECAE9", "#4C78A8", "#F58518", "#E45756"]):
        ax.plot(s.index, s.rolling(w, min_periods=1).mean(), lw=1.3, color=c, label=f"MA {w}")
    ax.set_title(T("이동평균 비교", "Moving averages"))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(growth.index, growth["YoY_%"], width=20, color="#4C78A8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(T("전년 동월 대비 증감률 (월평균)", "YoY growth (monthly mean)"))
    ax.set_ylabel("%")

    ax = axes[1, 1]
    ax.plot(s.index, cusum, lw=1.2, color="#B279A2")
    ax.axvline(s.index[break_idx], color="#E45756", ls="--", lw=1.2,
               label=T("변화 후보 시점", "Change point"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(T("CUSUM (구조 변화 탐색)", "CUSUM (structural break)"))
    ax.legend(fontsize=8)
    for a in axes.ravel():
        a.tick_params(axis="x", labelsize=8)

    fig.suptitle(T(f"[{PRODUCT}] 2. 트렌드 분석", f"[{PRODUCT}] 2. Trend analysis"), fontsize=13)
    out.fig(fig, "02_trend")


# ==============================================================================
# 3. 계절성 / 주기성 분석
# ==============================================================================
def seasonality_analysis(s: pd.Series, out: Out) -> None:
    print("\n[3] 계절성 / 주기성 분석")

    df = pd.DataFrame({"value": s})
    df["year"] = s.index.year
    df["month"] = s.index.month
    df["quarter"] = s.index.quarter
    df["dow"] = s.index.dayofweek          # 0=월 ... 6=일
    df["dom"] = s.index.day
    df["woy"] = s.index.isocalendar().week.astype(int)

    # 추세 제거 후 계절 지수 계산 (장기 추세가 강해 원계열 평균 비교는 왜곡됨)
    win = detrend_window(len(s))
    trend = s.rolling(win, center=True, min_periods=max(2, win // 4)).mean()
    detr = (s / trend).replace([np.inf, -np.inf], np.nan).dropna()
    d = pd.DataFrame({"ratio": detr})
    d["month"] = detr.index.month
    d["dow"] = detr.index.dayofweek

    dow_idx = d.groupby("dow")["ratio"].agg(["mean", "std", "count"])
    mon_idx = d.groupby("month")["ratio"].agg(["mean", "std", "count"])
    dow_idx.index.name, mon_idx.index.name = "dayofweek(0=Mon)", "month"
    print("  요일별 계절지수(추세 제거, 1.0=평균):")
    print(dow_idx["mean"].round(4).to_string())
    print("  월별 계절지수(추세 제거, 1.0=평균):")
    print(mon_idx["mean"].round(4).to_string())
    out.tab(dow_idx, "03_seasonal_index_dow")
    out.tab(mon_idx, "03_seasonal_index_month")

    # 계절성 유의성: Kruskal-Wallis
    kw_rows = []
    for key, label in [("dow", "요일"), ("month", "월")]:
        groups = [g["ratio"].values for _, g in d.groupby(key) if len(g) > 2]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            kw_rows.append({"구분": label, "H-stat": h, "p-value": p,
                            "유의(5%)": "예" if p < 0.05 else "아니오"})
    kw = pd.DataFrame(kw_rows)
    print(kw.to_string(index=False))
    out.tab(kw, "03_seasonality_test", index=False)

    # --- Figure: 계절성 박스플롯 / 히트맵 --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    dow_labels_ko = ["월", "화", "수", "목", "금", "토", "일"]
    dow_labels_en = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    present = sorted(d["dow"].unique())
    ax.boxplot([d.loc[d["dow"] == k, "ratio"].values for k in present])
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([(dow_labels_ko if USE_KO else dow_labels_en)[k] for k in present])
    ax.axhline(1.0, color="#E45756", ls="--", lw=1)
    ax.set_title(T("요일별 분포 (추세 제거 비율)", "By weekday (detrended ratio)"))

    ax = axes[0, 1]
    present_m = sorted(d["month"].unique())
    ax.boxplot([d.loc[d["month"] == m, "ratio"].values for m in present_m])
    ax.set_xticks(range(1, len(present_m) + 1))
    ax.set_xticklabels([str(m) for m in present_m])
    ax.axhline(1.0, color="#E45756", ls="--", lw=1)
    ax.set_title(T("월별 분포 (추세 제거 비율)", "By month (detrended ratio)"))

    ax = axes[1, 0]
    piv = df.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_title(T("연-월 평균 수요 히트맵", "Year-Month mean demand heatmap"))
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1, 1]
    q = df.groupby(["year", "quarter"])["value"].mean().unstack()
    for qq in q.columns:
        ax.plot(q.index, q[qq], marker="o", ms=4, label=f"Q{qq}")
    ax.set_title(T("연도별 분기 평균 추이", "Quarterly mean by year"))
    ax.legend(fontsize=8)

    fig.suptitle(T(f"[{PRODUCT}] 3-1. 계절성 패턴", f"[{PRODUCT}] 3-1. Seasonal patterns"), fontsize=13)
    out.fig(fig, "03_seasonality_patterns")

    # --- STL 분해 ---------------------------------------------------------------
    usable = [p for p in SEASONAL_PERIODS if len(s) >= 2 * p + 1]
    if usable:
        period = usable[0]
        stl = STL(s, period=period, robust=True).fit()
        comp = pd.DataFrame({"observed": s, "trend": stl.trend,
                             "seasonal": stl.seasonal, "resid": stl.resid})
        out.tab(comp, "03_stl_components")

        var_total = float(np.var(s.values))
        strength_trend = max(0.0, 1 - np.var(stl.resid) / max(np.var(stl.trend + stl.resid), 1e-12))
        strength_seas = max(0.0, 1 - np.var(stl.resid) / max(np.var(stl.seasonal + stl.resid), 1e-12))
        st = pd.DataFrame({"지표": ["STL period", "추세 강도(0~1)", "계절 강도(0~1)",
                                     "잔차 분산 비중(%)"],
                           "값": [period, strength_trend, strength_seas,
                                  100 * np.var(stl.resid) / var_total]})
        print(st.to_string(index=False))
        out.tab(st, "03_stl_strength", index=False)

        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        for ax, (name, ser, color) in zip(
            axes,
            [(T("실측", "Observed"), s, "#4C78A8"),
             (T("추세", "Trend"), stl.trend, "#E45756"),
             (T("계절", "Seasonal"), stl.seasonal, "#54A24B"),
             (T("잔차", "Residual"), stl.resid, "#888888")],
        ):
            ax.plot(ser.index, ser.values, lw=0.9, color=color)
            ax.set_ylabel(name, fontsize=9)
        axes[-1].axhline(0, color="black", lw=0.8)
        fig.suptitle(T(f"[{PRODUCT}] 3-2. STL 분해 (period={period})",
                       f"[{PRODUCT}] 3-2. STL decomposition (period={period})"), fontsize=13)
        out.fig(fig, "03_stl_decomposition")

    # --- ACF / PACF -------------------------------------------------------------
    nlags = int(min(max(SEASONAL_PERIODS) + 10, len(s) // 3, 400))
    d1 = s.diff().dropna()
    acf_lv, ci_lv = acf(s, nlags=nlags, alpha=0.05, fft=True)
    acf_d1, ci_d1 = acf(d1, nlags=nlags, alpha=0.05, fft=True)
    pacf_d1, ci_p = pacf(d1, nlags=min(nlags, len(d1) // 2 - 1), alpha=0.05)

    out.tab(pd.DataFrame({"lag": np.arange(len(acf_lv)), "acf_level": acf_lv,
                          "acf_diff1": acf_d1}), "03_acf_values", index=False)
    out.tab(pd.DataFrame({"lag": np.arange(len(pacf_d1)), "pacf_diff1": pacf_d1}),
            "03_pacf_diff1", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (vals, ci, title) in zip(
        axes.ravel(),
        [(acf_lv, ci_lv, T("ACF (원계열)", "ACF (level)")),
         (acf_d1, ci_d1, T("ACF (1차 차분)", "ACF (1st diff)")),
         (pacf_d1, ci_p, T("PACF (1차 차분)", "PACF (1st diff)"))],
    ):
        lags = np.arange(len(vals))
        ax.bar(lags, vals, width=0.8, color="#4C78A8")
        ax.fill_between(lags, ci[:, 0] - vals, ci[:, 1] - vals, color="#F58518", alpha=0.25)
        for p in SEASONAL_PERIODS:
            if p < len(vals):
                ax.axvline(p, color="#E45756", ls="--", lw=1, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("lag")

    # 자기상관 유의성: Ljung-Box
    lb = acorr_ljungbox(d1, lags=[min(10, nlags), min(PERIODS_PER_YEAR, nlags)], return_df=True)
    out.tab(lb, "03_ljungbox_diff1")
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.02, 0.95, T("Ljung-Box (1차 차분)", "Ljung-Box (1st diff)"),
            fontsize=11, va="top", transform=ax.transAxes)
    ax.text(0.02, 0.80, lb.round(4).to_string(), fontsize=9, va="top",
            family="monospace", transform=ax.transAxes)

    fig.suptitle(T(f"[{PRODUCT}] 3-3. 자기상관 구조", f"[{PRODUCT}] 3-3. Autocorrelation"), fontsize=13)
    out.fig(fig, "03_acf_pacf")

    # --- FFT 주기 탐지 ----------------------------------------------------------
    y = s.values - s.rolling(detrend_window(len(s)), center=True,
                             min_periods=1).mean().values  # 추세 제거
    y = np.nan_to_num(y - np.nanmean(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0)
    power = np.abs(np.fft.rfft(y)) ** 2
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
    mask = (periods >= 2) & (periods <= min(len(s) / 2, PERIODS_PER_YEAR * 1.5))
    top = np.argsort(power[mask])[::-1][:15]
    fft_tbl = pd.DataFrame({"period(단위=관측치)": periods[mask][top],
                            "power": power[mask][top]}).reset_index(drop=True)
    print("  FFT 상위 주기 후보:")
    print(fft_tbl.head(8).round(2).to_string(index=False))
    out.tab(fft_tbl, "03_fft_top_periods", index=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(periods[mask], power[mask], lw=0.9, color="#4C78A8")
    for p in SEASONAL_PERIODS:
        ax.axvline(p, color="#E45756", ls="--", lw=1.2, label=f"period={p}")
    ax.set_xscale("log")
    ax.set_xlabel(T("주기 (관측치 수)", "Period (observations)"))
    ax.set_ylabel("power")
    ax.set_title(T(f"[{PRODUCT}] 3-4. 주기도 (FFT)", f"[{PRODUCT}] 3-4. Periodogram (FFT)"))
    ax.legend(fontsize=8)
    out.fig(fig, "03_fft_periodogram")


# ==============================================================================
# 4. 정상성 검정
# ==============================================================================
def stationarity_tests(s: pd.Series, out: Out) -> None:
    print("\n[4] 정상성 검정 (ADF / KPSS)")

    log_s = np.log(s.clip(lower=1e-9))
    variants = {
        "원계열": s,
        "로그변환": log_s,
        "1차 차분": s.diff().dropna(),
        "로그 1차 차분": log_s.diff().dropna(),
        "2차 차분": s.diff().diff().dropna(),
    }
    for p in SEASONAL_PERIODS:
        if len(s) > 2 * p:
            variants[f"계절 차분(lag={p})"] = s.diff(p).dropna()
            variants[f"계절+1차 차분(lag={p})"] = s.diff(p).diff().dropna()

    rows = []
    for name, x in variants.items():
        x = pd.Series(x).dropna()
        if len(x) < 20:
            continue
        adf = adfuller(x, autolag="AIC")
        kp_c = kpss(x, regression="c", nlags="auto")
        kp_ct = kpss(x, regression="ct", nlags="auto")
        adf_stat = "정상" if adf[1] < 0.05 else "비정상"
        kpss_stat = "정상" if kp_c[1] > 0.05 else "비정상"
        rows.append({
            "변환": name,
            "n": len(x),
            "ADF_stat": adf[0], "ADF_p": adf[1], "ADF_사용lag": adf[2],
            "ADF_판정(H0=단위근)": adf_stat,
            "KPSS_c_stat": kp_c[0], "KPSS_c_p": kp_c[1],
            "KPSS_판정(H0=정상)": kpss_stat,
            "KPSS_ct_stat": kp_ct[0], "KPSS_ct_p": kp_ct[1],
            "종합": ("정상" if (adf_stat == "정상" and kpss_stat == "정상")
                     else "비정상" if (adf_stat == "비정상" and kpss_stat == "비정상")
                     else "판단보류(추세정상 가능)"),
        })
    res = pd.DataFrame(rows)
    print(res[["변환", "ADF_p", "ADF_판정(H0=단위근)", "KPSS_c_p", "KPSS_판정(H0=정상)", "종합"]]
          .round(4).to_string(index=False))
    out.tab(res, "04_stationarity_tests", index=False)

    rec = res[res["종합"] == "정상"]
    if len(rec):
        print(f"  → 권장 변환: {rec.iloc[0]['변환']} (ARIMA d/D 차수 결정에 활용)")
    else:
        print("  → 모든 변환에서 정상성 확보 실패. 추가 차분/변환 검토 필요.")

    # --- Figure -----------------------------------------------------------------
    show = ["원계열", "로그변환", "1차 차분", "로그 1차 차분"]
    show += [k for k in variants if k.startswith("계절 차분")][:2]
    show = [k for k in show if k in variants][:6]

    fig, axes = plt.subplots(len(show), 1, figsize=(15, 2.2 * len(show)))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, show):
        x = pd.Series(variants[name]).dropna()
        ax.plot(x.index, x.values, lw=0.7, color="#4C78A8")
        ax.plot(x.index, x.rolling(60, min_periods=1).mean(), lw=1.3, color="#E45756",
                label=T("롤링 평균", "rolling mean"))
        ax.plot(x.index, x.rolling(60, min_periods=1).std(), lw=1.3, color="#54A24B",
                label=T("롤링 표준편차", "rolling std"))
        ax.set_ylabel(T(name, name), fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(T(f"[{PRODUCT}] 4. 정상성: 변환별 평균/분산 안정성",
                   f"[{PRODUCT}] 4. Stationarity: mean/variance by transform"), fontsize=13)
    out.fig(fig, "04_stationarity")


# ==============================================================================
# 5. 변동성 확인
# ==============================================================================
def volatility_analysis(s: pd.Series, out: Out) -> None:
    print("\n[5] 변동성 분석")

    ret = np.log(s.clip(lower=1e-9)).diff().dropna()  # 로그 수익률(일간 변화율)
    roll = pd.DataFrame({
        "rolling_mean_30": s.rolling(30).mean(),
        "rolling_std_30": s.rolling(30).std(),
        "rolling_cv_30": s.rolling(30).std() / s.rolling(30).mean(),
        "rolling_std_90": s.rolling(90).std(),
        "rolling_cv_90": s.rolling(90).std() / s.rolling(90).mean(),
        "rolling_ret_std_30": ret.rolling(30).std(),
    })
    out.tab(roll, "05_rolling_volatility")

    # 연도별 변동성
    by_year = pd.DataFrame({
        "mean": s.groupby(s.index.year).mean(),
        "std": s.groupby(s.index.year).std(),
        "cv": s.groupby(s.index.year).std() / s.groupby(s.index.year).mean(),
        "ret_std": ret.groupby(ret.index.year).std(),
        "max_abs_ret_%": 100 * ret.abs().groupby(ret.index.year).max(),
    })
    by_year.index.name = "year"
    print(by_year.round(4).to_string())
    out.tab(by_year, "05_volatility_by_year")

    # 이상치: 로그 수익률 기준 IQR / z-score
    q1, q3 = ret.quantile(0.25), ret.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    zs = (ret - ret.mean()) / ret.std()
    flag = (ret < lo) | (ret > hi) | (zs.abs() > 4)
    outliers = pd.DataFrame({
        "log_return": ret[flag],
        "pct_change_%": 100 * (np.exp(ret[flag]) - 1),
        "zscore": zs[flag],
        "value": s.reindex(ret[flag].index),
    }).sort_index()
    print(f"  이상 변동일 {len(outliers)}건 (IQR 3배 또는 |z|>4)")
    out.tab(outliers, "05_outliers")

    # 변동성 군집(ARCH 효과) 검정
    arch_lm, arch_p, _, _ = het_arch(ret.values, nlags=min(10, len(ret) // 5))
    lb_sq = acorr_ljungbox(ret.values ** 2, lags=[10], return_df=True)
    vol_tbl = pd.DataFrame({
        "지표": ["전체 CV", "로그수익률 표준편차(일)", "로그수익률 연환산 변동성",
                 "수익률 왜도", "수익률 첨도", "ARCH LM stat", "ARCH LM p-value",
                 "변동성 군집(5%)", "Ljung-Box(제곱수익률,lag10) p"],
        "값": [s.std() / s.mean(), ret.std(), ret.std() * np.sqrt(PERIODS_PER_YEAR),
               float(stats.skew(ret.values)), float(stats.kurtosis(ret.values)),
               arch_lm, arch_p, "있음" if arch_p < 0.05 else "없음",
               float(lb_sq["lb_pvalue"].iloc[0])],
    })
    print(vol_tbl.to_string(index=False))
    out.tab(vol_tbl, "05_volatility_summary", index=False)

    # 정규성 검정
    norm_rows = []
    jb, jb_p = stats.jarque_bera(ret.values)
    norm_rows.append({"검정": "Jarque-Bera", "stat": jb, "p-value": jb_p})
    if len(ret) <= 5000:
        sw, sw_p = stats.shapiro(ret.values)
        norm_rows.append({"검정": "Shapiro-Wilk", "stat": sw, "p-value": sw_p})
    out.tab(pd.DataFrame(norm_rows), "05_normality_tests", index=False)

    # --- Figure: 변동성 -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(s.index, s.values, lw=0.7, color="#BBBBBB", label=T("실측", "Actual"))
    m30 = s.rolling(30).mean()
    sd30 = s.rolling(30).std()
    ax.plot(s.index, m30, lw=1.4, color="#4C78A8", label=T("30 이동평균", "MA 30"))
    ax.fill_between(s.index, m30 - 2 * sd30, m30 + 2 * sd30, color="#4C78A8", alpha=0.2,
                    label=T("±2σ 밴드", "±2σ band"))
    ax.set_title(T("이동평균 ± 2σ 밴드", "MA with ±2σ band"))
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(roll.index, roll["rolling_cv_30"], lw=1.1, color="#F58518", label="CV 30")
    ax.plot(roll.index, roll["rolling_cv_90"], lw=1.4, color="#E45756", label="CV 90")
    ax.set_title(T("롤링 변동계수 (변동성 추이)", "Rolling coefficient of variation"))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(ret.index, 100 * ret.values, lw=0.6, color="#54A24B")
    if len(outliers):
        ax.scatter(outliers.index, 100 * outliers["log_return"], s=18, color="#E45756",
                   zorder=3, label=T("이상 변동일", "Outliers"))
        ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("%")
    ax.set_title(T("일간 로그 변화율 및 이상치", "Log returns with outliers"))

    ax = axes[1, 1]
    ax.plot(roll.index, roll["rolling_ret_std_30"], lw=1.2, color="#B279A2")
    ax.set_title(T("30기간 롤링 수익률 변동성 (군집 확인)",
                   "Rolling 30 return volatility (clustering)"))
    fig.suptitle(T(f"[{PRODUCT}] 5. 변동성 분석", f"[{PRODUCT}] 5. Volatility analysis"), fontsize=13)
    out.fig(fig, "05_volatility")

    # --- Figure: 분포 -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.hist(s.values, bins=50, color="#4C78A8", alpha=0.85)
    ax.set_title(T("수요 분포", "Demand distribution"))

    ax = axes[1]
    ax.hist(100 * ret.values, bins=60, color="#54A24B", alpha=0.85, density=True)
    xs = np.linspace(100 * ret.min(), 100 * ret.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, 100 * ret.mean(), 100 * ret.std()), color="#E45756", lw=1.5,
            label=T("정규분포", "Normal"))
    ax.set_title(T("일간 변화율 분포 (%)", "Log return distribution (%)"))
    ax.legend(fontsize=8)

    ax = axes[2]
    stats.probplot(ret.values, dist="norm", plot=ax)
    ax.set_title(T("변화율 Q-Q plot", "Return Q-Q plot"))
    fig.suptitle(T(f"[{PRODUCT}] 5-2. 분포 특성", f"[{PRODUCT}] 5-2. Distribution"), fontsize=13)
    out.fig(fig, "05_distribution")


# ==============================================================================
# main
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=f"{PRODUCT} 수요 데이터 EDA")
    parser.add_argument("--input", default=INPUT_PATH, help="입력 CSV 경로")
    parser.add_argument("--outdir", default=f"./eda_output_{PRODUCT}", help="결과 저장 폴더")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {args.input}")
        return 1

    setup_font()
    out = Out(args.outdir)
    print("=" * 78)
    print(f" {PRODUCT} 수요 데이터 EDA  |  입력: {args.input}  |  출력: {args.outdir}")
    print("=" * 78)

    s, s_raw = load_and_clean(args.input, out)
    basic_profile(s, out)
    trend_analysis(s, out)
    seasonality_analysis(s, out)
    stationarity_tests(s, out)
    volatility_analysis(s, out)

    print("\n" + "=" * 78)
    print(f" 완료. PNG → {out.fig_dir} , CSV → {out.tab_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())