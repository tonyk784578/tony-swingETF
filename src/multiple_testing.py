"""다중검정 보정 — 이미 실행·기록된 시험들의 사후 진단 (2026-08-14 추가).

왜 필요한가: 이 프로젝트는 "누적 스크리닝 테스트 144회"를 성실히 기록해 왔지만,
각 후보를 평가할 때는 그 후보 하나만 놓고 t>=2 를 봤다. 144번 시도했다는 사실이
평가식에 들어간 적이 없다. 동전을 144번 던지면 '앞면 5연속'이 우연히도 흔한 것과
같은 문제 — 탐색을 많이 할수록 최고 성적의 기준선 자체가 올라간다.

**이 모듈이 하는 일과 하지 않는 일**

- 한다: 기존 결과를 다시 계산해 '탐색 강도를 감안한 유의성'을 낸다.
- 하지 않는다: 새 전략·새 파라미터·새 데이터 도입. 시험 수가 늘지 않으므로
  이 모듈을 돌리는 것 자체는 다중검정 부담을 **추가하지 않는다**.
- 하지 않는다: 판정 기준(config etf_paper.judgment) 변경. 그 기준은 장부 0행
  시점에 동결됐고, 데이터를 본 뒤 바꾸면 사후 기준 변경이 된다. 이 모듈의
  출력은 **해석 자료**이며, 포워드 판정을 대체하지도 완화하지도 않는다.

**세 개의 렌즈** (같은 질문을 서로 다른 가정으로 본다)

1. 기대 최대 t (Bailey & López de Prado 2014) — 엣지가 전혀 없어도 N회 시험 중
   최고 t가 얼마까지 나오는지. 독립 가정이라 상한 성격.
2. Reality Check / SPA (White 2000, Hansen 2005) — 시험 간 상관을 부트스트랩으로
   그대로 재현해 '전체 탐색 대비 최고 성적'의 p값을 낸다. 이 프로젝트처럼 같은
   ETF·같은 날 동시 발동하는 시험이 많으면 독립 가정은 지나치게 보수적이므로,
   이쪽이 주 판단 근거다. 부산물로 '유효 독립 시험 수'가 나온다.
3. Deflated Sharpe Ratio — 후보별로 선택 편의(SR0)와 비정규성(왜도·첨도)까지
   반영한 개별 유의확률.

보조로 Benjamini-Hochberg FDR(위양성 발견율)과 Holm 보정을 시험 t값에 적용한다.

실행: python -m src.main mtest  (측정 전용 — 신호·장부·판정 기준 불변)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import RESULTS_DIR, load_config

GAMMA = 0.5772156649015329   # 오일러-마스케로니 상수


# ---------------------------------------------------------------- 기대 최대 t

def expected_max_z(n_trials: float) -> float:
    """독립 시행 n회에서 표준정규 최댓값의 기대치 (BLdP 2014 식 (5) 근사).

    귀무가설(모든 시험의 진짜 엣지 = 0)에서 t는 근사적으로 N(0,1)이므로,
    이 값이 곧 '엣지가 없어도 기대되는 최고 t'다.
    """
    n = max(float(n_trials), 2.0)
    return float((1 - GAMMA) * stats.norm.ppf(1 - 1 / n)
                 + GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))


def effective_trials(expected_max: float) -> float:
    """기대 최대 t를 거꾸로 풀어 얻는 '유효 독립 시험 수'.

    부트스트랩이 상관을 반영해 낸 귀무 최대 t가 예컨대 2.1이라면, 실제 144회
    시험은 독립 시험 몇 회에 해당하는가 — expected_max_z 의 역함수(단조증가라
    이분탐색). 시험들이 서로 닮아 있을수록 유효 수는 작아진다.
    """
    if not np.isfinite(expected_max) or expected_max <= expected_max_z(2):
        return 2.0
    lo, hi = 2.0, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if expected_max_z(mid) < expected_max:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ------------------------------------------------------- Deflated Sharpe Ratio

def deflated_sharpe(r: np.ndarray, var_sr: float, n_trials: float) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    r: **일간** 수익 시리즈 (daily_matrix 의 한 열). 트레이드 단위가 아니다 —
      SR0 는 '전 시험 SR의 분산'에서 나오는데, 보유기간이 다른 시험(1일 보유
      1,200건 vs 18일 보유 44건)의 트레이드 SR을 한 통에 섞으면 표본 크기 차이가
      그대로 분산으로 잡혀 기준선이 왜곡된다. 모든 시험을 같은 달력·같은 관측 수
      위에 올려야 SR이 비교 가능해진다 (원논문도 고정 빈도 수익률을 전제).
    var_sr: 전 시험의 일간 SR 추정치 분산 — 선택 편의의 크기
    반환 dsr: '이 후보의 진짜 SR > 0' 확률. 0.95 이상이면 통상 유의로 본다.
    """
    r = pd.Series(np.asarray(r, dtype=float)).dropna()
    t_obs = len(r)
    sd = r.std(ddof=1)
    if t_obs < 3 or not np.isfinite(sd) or sd <= 0:
        return {"n": t_obs, "sr": np.nan, "sr0": np.nan, "dsr": np.nan,
                "skew": np.nan, "kurt": np.nan}
    sr = float(r.mean() / sd)
    sr0 = float(np.sqrt(max(var_sr, 0.0)) * expected_max_z(n_trials))
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    var_term = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if not np.isfinite(var_term) or var_term <= 0:
        return {"n": t_obs, "sr": sr, "sr0": sr0, "dsr": np.nan,
                "skew": g3, "kurt": g4}
    z = (sr - sr0) * np.sqrt(t_obs - 1) / np.sqrt(var_term)
    return {"n": t_obs, "sr": sr, "sr0": sr0, "dsr": float(stats.norm.cdf(z)),
            "skew": g3, "kurt": g4}


# ------------------------------------------------ 정상 부트스트랩 + RC/SPA 검정

def stationary_bootstrap_idx(n: int, block: int, draws: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) 정상(stationary) 부트스트랩 인덱스 — (draws, n).

    블록 길이를 평균 `block`의 기하분포로 뽑아 순환 재표집한다. 블록으로 끊어
    담기 때문에 (a) 수익률의 자기상관과 (b) 같은 날 여러 시험이 동시에 움직이는
    횡단면 상관이 함께 보존된다 — 후자가 이 프로젝트의 핵심 (풀링 t 인플레의 원인).
    """
    p = 1.0 / max(int(block), 1)
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, draws)
    new_block = rng.random((draws, n)) < p
    starts = rng.integers(0, n, (draws, n))
    for j in range(1, n):
        idx[:, j] = np.where(new_block[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return idx


def bootstrap_means(mat: np.ndarray, block: int, draws: int, seed: int) -> np.ndarray:
    """재표집별 시험 평균 행렬 (draws, n_trials).

    인덱스를 카운트로 바꿔 행렬곱 한 번으로 계산한다 (재표집마다 gather 하면
    같은 결과에 수십 배 느리다).
    """
    n = mat.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = stationary_bootstrap_idx(n, block, draws, rng)
    counts = np.zeros((draws, n))
    for b in range(draws):
        counts[b] = np.bincount(idx[b], minlength=n)
    return counts @ mat / n


def reality_check(mat: np.ndarray, block: int, draws: int, seed: int) -> dict:
    """White(2000) Reality Check + Hansen(2005) SPA.

    mat: (거래일 x 시험) 일간 수익 행렬. 벤치마크는 '매매 안 함'(0)이다.
    귀무가설: 어떤 시험도 벤치마크보다 낫지 않다. p가 작아야 '탐색 전체를
    감안해도 최고 성적은 우연이 아니다'가 된다.
    """
    n, k = mat.shape
    fbar = mat.mean(axis=0)
    boot = bootstrap_means(mat, block, draws, seed)
    centered = boot - fbar                       # 귀무로 재중심화

    omega = centered.std(axis=0, ddof=1) * np.sqrt(n)
    omega = np.where(omega > 0, omega, np.inf)   # 무거래 시험은 배제되도록

    v_obs = float(np.sqrt(n) * fbar.max())
    v_boot = np.sqrt(n) * centered.max(axis=1)
    p_rc = float((v_boot >= v_obs).mean())

    t_stats = np.sqrt(n) * fbar / omega
    t_obs = float(max(t_stats.max(), 0.0))
    # SPA 재중심화: 성적이 확실히 나쁜 시험은 귀무 최댓값에 기여시키지 않는다
    keep = t_stats >= -np.sqrt(2 * np.log(np.log(n)))
    ghat = np.where(keep, fbar, 0.0)
    t_boot = np.maximum((np.sqrt(n) * (boot - ghat) / omega).max(axis=1), 0.0)
    p_spa = float((t_boot > t_obs).mean())

    # 귀무 최대 t 분포 — 재중심화 후 스튜던트화한 값의 재표집별 최댓값
    null_max_t = (np.sqrt(n) * centered / omega).max(axis=1)
    return {
        "n_days": n, "n_trials": k,
        "best_idx": int(np.argmax(t_stats)), "best_t": t_obs,
        "v_obs": v_obs, "p_rc": p_rc, "p_spa": p_spa,
        "null_max_t_mean": float(np.mean(null_max_t)),
        "null_max_t_q95": float(np.quantile(null_max_t, 0.95)),
        "trial_t": t_stats,
    }


# ------------------------------------------------------------- 다중성 보정 p값

def bh_fdr(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg — 기각된 것 중 위양성 비율을 q 이하로 통제. 반환: 기각 여부."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        out[order[:cutoff + 1]] = True
    return out


def holm(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Holm-Bonferroni — 하나라도 위양성일 확률을 alpha 이하로 통제 (엄격)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    out = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


# ------------------------------------------------------------------ 시험 수집

def collect_trials(force: bool = False) -> tuple[list[dict], pd.DatetimeIndex]:
    """재현 가능한 전 시험의 트레이드 목록 + 공통 거래일 달력.

    스크리닝은 `iter_screen_trades`(스크리닝 통계와 동일 경로)를 그대로 쓰고,
    로테이션 2건은 동결된 에피소드 CSV를 읽는다. 새로 만드는 시험은 없다.
    """
    from .etf_swing import iter_screen_trades

    ecfg = load_config()["etf"]
    strats = ecfg["strategies"]
    trials: list[dict] = []
    days: set = set()

    # 본 스크리닝 (전략별 제한 유니버스 선언은 run_screening 과 같은 규칙)
    base = [s for s in strats if "universe" not in strats[s]]
    groups = [(ecfg["universe"], base, "screening")]
    for s, p in strats.items():
        if "universe" in p:
            groups.append(({c: ecfg["universe"][c] for c in p["universe"]}, [s], "screening"))
    if ecfg.get("universe_ext"):
        groups.append((ecfg["universe_ext"], list(ecfg["ext_strategies"]), "ext"))

    for universe, strategies, source in groups:
        for item in iter_screen_trades(universe, strategies, force):
            days |= set(item["df"].index)
            if item["trades"].empty:
                continue
            trials.append({
                "label": f"{item['etf']} {item['strategy']}",
                "etf": item["etf"], "strategy": item["strategy"],
                "source": source, "trades": item["trades"],
            })

    # 로테이션 2건 — 조합 스크리닝이 아니라 단일 실험이지만 같은 탐색 예산에 속한다
    for stem, label in [("rotation_episodes", "rotation(12종) dual_momentum"),
                        ("rotation2_episodes", "rotation2(확장) dual_momentum")]:
        path = RESULTS_DIR / f"{stem}.csv"
        if not path.exists():
            continue
        ep = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
        days |= set(ep["exit_date"])
        trials.append({"label": label, "etf": "(포트폴리오)",
                       "strategy": stem.replace("_episodes", ""),
                       "source": "rotation", "trades": ep})

    return trials, pd.DatetimeIndex(sorted(days))


def daily_matrix(trials: list[dict], calendar: pd.DatetimeIndex) -> np.ndarray:
    """(거래일 x 시험) 일간 수익 행렬 — 청산일에 트레이드 순수익을 놓고 나머지는 0.

    0 = '그날 포지션이 없었다'. 상장 전 구간도 0이라 역사가 짧은 시험은 평균이
    희석되는데, 이는 그 시험에 불리한 방향이라 보수적이다. 이렇게 같은 달력 위에
    올려야 회전율이 다른 전략(1일 보유 vs 18일 보유)을 같은 시간 단위로 비교할 수 있다.
    """
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    mat = np.zeros((len(calendar), len(trials)))
    for j, tr in enumerate(trials):
        ex = pd.DatetimeIndex(tr["trades"]["exit_date"])
        i = pos.reindex(ex).to_numpy()
        ok = ~pd.isna(i)
        np.add.at(mat[:, j], i[ok].astype(int), tr["trades"]["net_ret"].to_numpy()[ok])
    return mat


# ---------------------------------------------------------------------- 실행

def run_multiple_testing(force: bool = False) -> dict:
    cfg = load_config()
    mcfg = cfg["multiple_testing"]
    bcfg = mcfg["bootstrap"]
    declared = int(mcfg["declared_trials"])

    trials, calendar = collect_trials(force)
    mat = daily_matrix(trials, calendar)

    # 시험별 트레이드 단위 통계 (프로젝트의 게이트와 같은 단위)
    rows = []
    for tr in trials:
        r = tr["trades"]["net_ret"]
        sd = r.std(ddof=1) if len(r) > 1 else np.nan
        sr = r.mean() / sd if np.isfinite(sd) and sd > 0 else np.nan
        t = sr * np.sqrt(len(r)) if np.isfinite(sr) else np.nan
        rows.append({"label": tr["label"], "etf": tr["etf"], "strategy": tr["strategy"],
                     "source": tr["source"], "n": len(r), "mean": r.mean(),
                     "sr_trade": sr, "t_stat": t})
    tri = pd.DataFrame(rows)
    tri["p_one_sided"] = np.where(
        tri["n"] > 1, stats.t.sf(tri["t_stat"], np.maximum(tri["n"] - 1, 1)), np.nan)

    # 일간 기준 SR — 시험 간 비교가 성립하는 유일한 단위 (deflated_sharpe docstring)
    sd_daily = mat.std(axis=0, ddof=1)
    tri["sr_daily"] = np.where(sd_daily > 0, mat.mean(axis=0) / np.where(sd_daily > 0, sd_daily, 1), np.nan)
    var_sr = float(np.nanvar(tri["sr_daily"].to_numpy(), ddof=1))
    rc = reality_check(mat, int(bcfg["block_days"]), int(bcfg["draws"]), int(bcfg["seed"]))

    # 렌즈 1: 독립 가정 기대 최대 t / 렌즈 2: 상관 반영 유효 시험 수
    emax_declared = expected_max_z(declared)
    emax_observed = expected_max_z(len(tri))
    n_eff = effective_trials(rc["null_max_t_mean"])
    indep_ratio = n_eff / len(tri)            # 재현 시험 기준 독립성 비율
    emax_adjusted = expected_max_z(declared * indep_ratio)

    # 렌즈 3: 후보별 DSR — Stage 2 등록 후보 + 로테이션 실험
    reg = {(str(c["name"]), str(c["strategy"])) for c in cfg["etf_paper"]["candidates"]}
    tri["registered"] = [(r.etf, r.strategy) in reg for r in tri.itertuples()]
    dsr_rows = []
    for j, tr in enumerate(trials):
        key = (tr["etf"], tr["strategy"])
        if key not in reg and tr["source"] != "rotation":
            continue
        d = deflated_sharpe(mat[:, j], var_sr, declared)
        dsr_rows.append({"label": tr["label"], "trades": len(tr["trades"]), **d})
    dsr = pd.DataFrame(dsr_rows).sort_values("dsr", ascending=False).reset_index(drop=True)

    # 보조: 다중성 보정 p값 (트레이드 단위 단측 p 기준)
    valid = tri["p_one_sided"].notna()
    tri["bh_pass"] = False
    tri["holm_pass"] = False
    tri.loc[valid, "bh_pass"] = bh_fdr(tri.loc[valid, "p_one_sided"].to_numpy(),
                                       float(mcfg["fdr_q"]))
    tri.loc[valid, "holm_pass"] = holm(tri.loc[valid, "p_one_sided"].to_numpy(), 0.05)

    tri = tri.sort_values("t_stat", ascending=False).reset_index(drop=True)
    res = {"trials": tri, "dsr": dsr, "rc": rc, "var_sr": var_sr,
           "declared": declared, "emax_declared": emax_declared,
           "emax_observed": emax_observed, "n_eff": n_eff,
           "indep_ratio": indep_ratio, "emax_adjusted": emax_adjusted,
           "best_label": trials[rc["best_idx"]]["label"]}
    tri.to_csv(RESULTS_DIR / "multiple_testing.csv", index=False, encoding="utf-8-sig")
    _write_report(res, mcfg)
    return res


def _write_report(res: dict, mcfg: dict) -> None:
    tri, dsr, rc = res["trials"], res["dsr"], res["rc"]
    b = mcfg["bootstrap"]
    gate2 = tri[tri["t_stat"] >= 2]
    lines = [f"""# 다중검정 보정 (사후 진단)

생성일: {pd.Timestamp.today().date()} | 재현 시험 {len(tri)}개 (스크리닝 {int((tri['source'] != 'rotation').sum())} + 로테이션 {int((tri['source'] == 'rotation').sum())})
| 선언 누적 탐색 {res['declared']}회 (CLAUDE.md) | 부트스트랩 {b['draws']}회 x 블록 {b['block_days']}일, seed {b['seed']}

**측정 전용.** 새 전략·새 파라미터·새 데이터를 도입하지 않으므로 이 문서를
만드는 것 자체는 시험 수를 늘리지 않는다. 판정 기준(config `etf_paper.judgment`)은
장부 0행 시점에 동결됐고 이 결과로 바꾸지 않는다 — 여기 나온 수치는 **해석 자료**이며,
포워드 판정을 대체하지도 완화하지도 않는다.

## 1. 엣지가 전혀 없어도 나오는 최고 t (독립 가정)

시험을 많이 할수록 '최고 성적'의 기준선이 올라간다. 진짜 엣지가 0이어도:

| 시험 수 | 기대 최대 t |
|---|---|
| {res['declared']}회 (선언 누적 탐색) | **{res['emax_declared']:.2f}** |
| {len(tri)}회 (여기서 재현된 시험) | {res['emax_observed']:.2f} |

Stage 1 게이트는 t >= 2 였다. 독립 가정에서는 **게이트가 잡음 상한보다 낮다** —
t=2 하나만으로는 우연과 구분되지 않는다는 뜻이다. 다만 이 계산은 시험들이 서로
독립이라고 본 것이라 지나치게 보수적이다 (실제로는 같은 ETF·같은 날 겹친다).

## 2. 상관을 반영한 검정 — Reality Check / SPA

같은 날 동시에 움직이는 구조를 부트스트랩으로 그대로 재현해 귀무분포를 만든다.
시험을 모두 같은 거래일 달력 위에 올리고(청산일에 손익 배치, 무포지션일은 0),
평균 {b['block_days']}일 블록으로 재표집해 자기상관과 횡단면 상관을 함께 보존한다.

- 최고 성적 시험: **{res['best_label']}** (일간 스튜던트화 t = {rc['best_t']:.2f})
- White Reality Check p = **{rc['p_rc']:.3f}**
- Hansen SPA p = **{rc['p_spa']:.3f}**  ← 주 근거
- 귀무 최대 t: 평균 {rc['null_max_t_mean']:.2f}, 95퍼센타일 {rc['null_max_t_q95']:.2f}
- 재현 시험 {len(tri)}개의 **유효 독립 수 ≈ {res['n_eff']:.0f}개** (독립성 비율 {res['indep_ratio']:.0%})
  → 선언 {res['declared']}회에 같은 비율을 적용하면 기대 최대 t = **{res['emax_adjusted']:.2f}**

두 p값이 갈리는 이유: RC는 표준화하지 않은 평균의 최댓값을 쓰기 때문에 변동성이
큰 시험(레버리지 ETF)이 귀무분포를 지배해 과도하게 보수적이 된다. Hansen의 SPA는
바로 이 결함을 고치려고 나온 검정(스튜던트화 + 성적이 명백히 나쁜 시험을 귀무
최댓값에서 제외)이라 이쪽을 주 근거로 본다.

유효 독립 수가 명목보다 크게 줄지 않은 것도 정직하게 기록해 둔다 — 시험들이
서로 닮아 있긴 하지만 자산군·전략 메커니즘이 실제로 갈려 있어, "상관이 높으니
보정을 완화해도 된다"는 변명은 이 데이터에서 성립하지 않는다.

## 3. 후보별 Deflated Sharpe Ratio

선택 편의(SR0)와 비정규성(왜도·첨도)까지 반영한 개별 유의확률. DSR >= 0.95 가 통상 기준.

SR은 **일간** 기준이다 (연율 환산은 x sqrt(252)). 트레이드 단위를 쓰지 않는 이유:
SR0는 '전 시험 SR의 분산'에서 나오는데, 1일 보유 1,200건짜리 시험과 18일 보유
44건짜리 시험의 트레이드 SR을 한 통에 섞으면 표본 크기 차이가 그대로 분산으로
잡혀 기준선이 부풀고, 표본이 큰 시험이 부당하게 처벌된다. 같은 달력 위의 일간
수익으로 통일해야 시험 간 SR 비교가 성립한다.

| 후보 | 트레이드 | 관측일 | 일간 SR | 연율 SR | 선택편의 SR0(일간) | 왜도 | 첨도 | DSR |
|---|---|---|---|---|---|---|---|---|"""]
    for _, r in dsr.iterrows():
        d = "-" if not np.isfinite(r["dsr"]) else (
            f"**{r['dsr']:.3f}**" if r["dsr"] >= 0.95 else f"{r['dsr']:.3f}")
        lines.append(f"| {r['label']} | {int(r['trades'])} | {int(r['n'])} "
                     f"| {r['sr']:.4f} | {r['sr'] * np.sqrt(252):.2f} | {r['sr0']:.4f} "
                     f"| {r['skew']:+.2f} | {r['kurt']:.0f} | {d} |")

    sr0 = float(dsr["sr0"].iloc[0])
    low_freq = dsr[dsr["trades"] < 200]
    lf_note = ("이 후보들은 전부 SR < SR0 라 왜도·첨도 항과 무관하게 결론(미달)이 같다"
               if (low_freq["sr"] < sr0).all() else
               "**주의: 이 중 SR > SR0 인 후보가 있어 첨도 항이 결론에 영향을 준다**")
    lines.append(f"""
SR0 = sqrt(전 시험 일간 SR 분산 {res['var_sr']:.6f}) x 기대최대z({res['declared']}) = {sr0:.4f} —
"{res['declared']}회 탐색이라면 이 정도 일간 SR은 운으로도 나온다"는 문턱값이다.

**한계 (저빈도 전략의 왜도·첨도)**: 연 4~12회 매매하는 전략은 일간 시리즈의
95% 이상이 0이라 왜도·첨도가 구조적으로 커진다 (표의 세 자릿수 첨도). 이는 수익
분포의 성질이 아니라 '거래일이 드물다'는 사실의 반영이므로 액면 그대로 읽지 말 것.
{lf_note}.

## 4. 다중성 보정 후 살아남는 시험

트레이드 단위 단측 p값에 보정을 적용한 결과 (전 {len(tri)}개 시험 기준):

| 기준 | 통과 수 | 통과 시험 |
|---|---|---|
| 무보정 t >= 2 (Stage 1 게이트의 t 조건만) | {len(gate2)} | {', '.join(gate2['label'].head(8)) or '없음'}{' 외' if len(gate2) > 8 else ''} |
| Benjamini-Hochberg FDR q={mcfg['fdr_q']} | {int(tri['bh_pass'].sum())} | {', '.join(tri.loc[tri['bh_pass'], 'label']) or '없음'} |
| Holm-Bonferroni α=0.05 (최엄격) | {int(tri['holm_pass'].sum())} | {', '.join(tri.loc[tri['holm_pass'], 'label']) or '없음'} |

FDR은 "기각한 것 중 위양성 비율"을, Holm은 "하나라도 위양성일 확률"을 통제한다.
Holm까지 살아남은 시험은 다중검정에 가장 강건하다. (Stage 1 게이트는 t 조건 외에
N>=30·전후반 양수도 함께 요구했으므로 위 첫 줄이 곧 Stage 2 등록 목록은 아니다 —
보정 강도의 비교용 기준선이다.)

전체 시험표는 `multiple_testing.csv` (t 내림차순).

## 읽는 법 (한 줄)

1번은 상한(가장 가혹), 2번이 이 프로젝트 구조에 맞는 주 근거, 3번은 후보별 참고.
셋 중 어느 것도 포워드 섀도 판정을 대신하지 않는다 — 전부 같은 인샘플 데이터를
다르게 볼 뿐이고, 인샘플에서 살아남는 것과 앞으로 벌어들이는 것은 다른 질문이다.
""")
    (RESULTS_DIR / "multiple_testing.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
