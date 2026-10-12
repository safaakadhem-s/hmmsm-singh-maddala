# ============================================================
# IRISH RAINFALL — Monthly HMM-SM Study
# MASTER FILE (v7) — Fully self-contained, no external patches
#
# Workflow:
#   1. Robust download + gzip integrity check
#   2. Auto-detect date columns, enforce calendar length
#   3. Monthly aggregation + unit conversion (0.1 mm → mm)
#   4. Automatic K-selection over K=1..5 via WAIC/DIC/LOO
#      with Burnham & Anderson parsimony rule
#   5. Focused inference on selected K:
#        - 4 independent chains × 30,000 iterations (15,000 burn)
#        - Rank-normalized R-hat, folded R-hat, ESS, Geweke
#        - 12 publication-quality figures (300 dpi)
#        - Compact + extended summary tables
#   6. Extended summary with numerical stabilisation:
#        - E[Y|S=k] computed in log-space (log-gamma)
#        - MAD-based clipping of extreme values
#        - NaN replacement by posterior median
#        - Duration clipped at DURATION_CAP = 100 months
# ============================================================
import os
import gzip
import calendar
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln, gamma as gamma_fn
from scipy.stats import norm, rankdata
from numba import njit
from collections import Counter
import warnings
import time

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"D:\فكرة بحث 2027\irish_rainfall"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "tables"), exist_ok=True)

BASE_URL = "https://clidata.met.ie/cli/grids_daily/latest/rain/"
YEARS = list(range(2000, 2026))

# --- K-selection settings ---
K_VALUES      = [1, 2, 3, 4, 5]
K_SEL_ITER    = 6000
K_SEL_BURN    = 3000
PARSIMONY_TOL = 2.0

# --- Final inference settings ---
N_CHAINS      = 4
N_ITER_FINAL  = 30000
BURN_FINAL    = 15000
SEED          = 42

# --- Numerical stabilisation settings ---
DURATION_CAP  = 100.0   # clip durations at 100 months
MAD_K         = 5.0     # clip E[Y|S=k] at median ± MAD_K × MAD

print("=" * 78)
print("IRISH RAINFALL — MONTHLY HMM-SM (MASTER v7)")
print("=" * 78)
print(f"Years       : {YEARS[0]}–{YEARS[-1]}")
print(f"K candidates: {K_VALUES}")
print(f"Final chains: {N_CHAINS} × {N_ITER_FINAL} (burn {BURN_FINAL})")
print()


# ============================================================
# PART 0: DOWNLOAD + VERIFY
# ============================================================
def _is_valid_gzip(path, min_size_mb=1.0):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < min_size_mb * 1024 * 1024:
        return False
    try:
        with gzip.open(path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
        return True
    except (EOFError, OSError, gzip.BadGzipFile):
        return False


def _download_one(url, dest, timeout=300, min_size_mb=1.0):
    tmp = dest + ".part"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            print(f"FAIL HTTP {r.status_code}")
            return False
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        if not _is_valid_gzip(tmp, min_size_mb=min_size_mb):
            os.remove(tmp)
            print("FAIL (corrupt after download)")
            return False
        os.replace(tmp, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"FAIL {str(e)[:60]}")
        return False


def download_missing_years():
    print("[STEP 0] Verifying local files...")
    downloaded = re_downloaded = skipped = 0
    failed = []
    for year in YEARS:
        filename = f"IRL_DLY_RR_{year}_grid.csv.gz"
        local_path = os.path.join(BASE_DIR, filename)
        url = BASE_URL + filename

        if os.path.exists(local_path):
            print(f"  [{year}] verifying ...", end=" ", flush=True)
            if _is_valid_gzip(local_path):
                print("OK"); skipped += 1; continue
            print("CORRUPT -> re-downloading")
            try: os.remove(local_path)
            except OSError: pass
            if _download_one(url, local_path):
                re_downloaded += 1
            else:
                failed.append(year)
        else:
            print(f"  [{year}] missing -> downloading ...",
                  end=" ", flush=True)
            if _download_one(url, local_path):
                downloaded += 1
            else:
                failed.append(year)

    print(f"\n  {downloaded} new | {re_downloaded} re-downloaded "
          f"| {skipped} valid")
    if failed:
        print(f"  FAILED: {failed}")
    print()


# ============================================================
# PART 1: ROBUST LOADER
# ============================================================
def _is_date_column(s):
    if len(s) == 9 and s[0] in 'XxDd':
        return s[1:].isdigit()
    if len(s) == 8:
        return s.isdigit()
    return False


def load_daily_series_for_year(year):
    filename = f"IRL_DLY_RR_{year}_grid.csv.gz"
    local_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(local_path):
        return None

    col_sums = None; n_rows = 0; n_cols = None; skip_cols = None
    try:
        with gzip.open(local_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip().strip('"') for p in line.split(',')]

                if col_sums is None:
                    n_cols = len(parts)
                    skip_cols = 0
                    for p in parts:
                        if _is_date_column(p):
                            break
                        skip_cols += 1
                    if skip_cols == 0 or skip_cols >= n_cols:
                        skip_cols = 2
                    col_sums = np.zeros(n_cols - skip_cols)
                    try:
                        float(parts[0])
                    except ValueError:
                        continue
                if len(parts) != n_cols:
                    continue
                try:
                    vals = np.array([float(x) if x else 0.0
                                    for x in parts[skip_cols:]])
                except (ValueError, IndexError):
                    continue
                col_sums += vals
                n_rows += 1
    except (EOFError, OSError, gzip.BadGzipFile):
        return None
    if col_sums is None or n_rows == 0:
        return None
    return col_sums / n_rows


def load_daily_data():
    print("[STEP 1] Loading daily data...")
    all_series, all_dates, missing = [], [], []
    for year in YEARS:
        print(f"  [{year}] ", end="", flush=True)
        daily = load_daily_series_for_year(year)
        if daily is None:
            print("MISSING"); missing.append(year); continue

        n_expected = 366 if calendar.isleap(year) else 365
        n_days = len(daily)
        if n_days > n_expected:
            daily = daily[:n_expected]
        elif n_days < n_expected:
            pad = np.full(n_expected - n_days, np.nan)
            daily = np.concatenate([daily, pad])

        dates = pd.date_range(f"{year}-01-01", periods=n_expected, freq='D')
        all_series.append(daily)
        all_dates.extend(dates)
        print(f"OK {n_expected} days")

    if missing:
        print(f"  WARNING: missing {missing}")

    Y = np.concatenate(all_series)
    dates = pd.DatetimeIndex(all_dates)
    valid = ~np.isnan(Y)
    Y, dates = Y[valid], dates[valid]

    assert dates.is_monotonic_increasing
    assert not dates.has_duplicates

    print(f"\n  Total: {len(Y)} days")
    print(f"  Period: {dates.min().date()} → {dates.max().date()}")
    return Y, dates


# ============================================================
# PART 2: MONTHLY AGGREGATION + UNIT CONVERSION
# ============================================================
def aggregate_to_monthly(Y_daily, dates_daily, agg="total"):
    print(f"\n[STEP 2] Aggregating to monthly ({agg})...")
    df = pd.DataFrame({'rain': Y_daily}, index=dates_daily)

    if agg == "total":
        monthly = df['rain'].resample('MS').sum()
    elif agg == "max":
        monthly = df['rain'].resample('MS').max()
    elif agg == "mean":
        monthly = df['rain'].resample('MS').mean()
    else:
        raise ValueError("agg must be total/max/mean")

    monthly = monthly.dropna()
    Y_m, dates_m = monthly.values, monthly.index
    Y_m = Y_m / 10.0    # 0.1 mm → mm

    print(f"  Monthly series : {len(Y_m)} months")
    print(f"  Period         : {dates_m.min().date()} → "
          f"{dates_m.max().date()}")
    print(f"  Range          : [{Y_m.min():.2f}, {Y_m.max():.2f}] mm")
    print(f"  Mean           : {Y_m.mean():.2f} mm")
    print(f"  SD             : {Y_m.std():.2f} mm")
    print(f"  Units          : converted 0.1 mm → mm")
    return Y_m, dates_m


# ============================================================
# PART 3: HMM-SM CORE (hard-wall prior on b)
# ============================================================
@njit(fastmath=True)
def sm_logpdf(y, a, b, q):
    log_yb = np.log(y) - np.log(b)
    a_log_yb = a * log_yb
    if a_log_yb > 30.0:
        l1p = a_log_yb
    elif a_log_yb < -30.0:
        l1p = 0.0
    else:
        l1p = np.log1p(np.exp(a_log_yb))
    return (np.log(a) + np.log(q) - np.log(b)
            + (a - 1.0) * log_yb - (q + 1.0) * l1p)


@njit(fastmath=True)
def emission_matrix(Y, a, b, q):
    T, K = Y.shape[0], a.shape[0]
    log_em = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            log_em[t, k] = sm_logpdf(Y[t], a[k], b[k], q[k])
    return log_em


@njit(fastmath=True)
def forward_filter(log_em, pi, Gamma):
    T, K = log_em.shape
    log_pi = np.log(pi); log_G = np.log(Gamma)
    log_alpha = np.empty((T, K))
    for k in range(K):
        log_alpha[0, k] = log_pi[k] + log_em[0, k]
    for t in range(1, T):
        for k in range(K):
            m = -np.inf
            for j in range(K):
                v = log_alpha[t-1, j] + log_G[j, k]
                if v > m: m = v
            s = 0.0
            for j in range(K):
                s += np.exp(log_alpha[t-1, j] + log_G[j, k] - m)
            log_alpha[t, k] = m + np.log(s) + log_em[t, k]
    return log_alpha


@njit(fastmath=True)
def log_target(log_abc, y_state):
    a = np.exp(log_abc[0])
    b = np.exp(log_abc[1])
    q = np.exp(log_abc[2])

    if b < 20.0 or b > 400.0:
        return -1e15

    log_prior = ((2.0 - 1.0) * np.log(a) - 2.0 * a
                 + (8.0 - 1.0) * np.log(b) - 0.08 * b
                 + (2.0 - 1.0) * np.log(q) - 1.0 * q)

    log_lik = 0.0
    for i in range(len(y_state)):
        log_lik += sm_logpdf(y_state[i], a, b, q)
    return log_prior + log_lik + log_abc.sum()


def ffbs(log_em, pi, Gamma, rng):
    T, K = log_em.shape
    log_alpha = forward_filter(log_em, pi, Gamma)
    log_G = np.log(Gamma)
    S = np.empty(T, dtype=np.int64)
    probs = np.exp(log_alpha[-1] - log_alpha[-1].max())
    probs /= probs.sum()
    S[-1] = rng.choice(K, p=probs)
    for t in range(T-2, -1, -1):
        log_w = log_alpha[t] + log_G[:, S[t+1]]
        probs = np.exp(log_w - log_w.max())
        probs /= probs.sum()
        S[t] = rng.choice(K, p=probs)
    return S


def run_mcmc(Y, K, n_iter, burn, seed, verbose=False):
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    T = len(Y)
    t0 = time.time()

    b_pos = Y[Y > 0]
    if len(b_pos) > 0:
        b_q = np.linspace(0.30, 0.85, K)
        b = np.clip(np.percentile(b_pos, b_q * 100), 25.0, 300.0)
    else:
        b = np.linspace(50.0, 150.0, K)
    a = np.full(K, 1.5)
    q = np.full(K, 2.0)

    pi = np.ones(K) / K
    Gamma = np.full((K, K), 1.0/K) + 0.1*np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.ones(K) * 2.0

    prop_cov = [np.eye(3) * 0.3**2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K); n_prop = np.zeros(K)
    samples = {k: [] for k in ['a', 'b', 'q', 'pi', 'Gamma', 'S']}

    for it in range(n_iter):
        log_em = emission_matrix(Y, a, b, q)
        S = ffbs(log_em, pi, Gamma, rng)

        counts = np.bincount(S[:1], minlength=K).astype(float)
        pi = rng.dirichlet(pi_prior + counts)

        for i in range(K):
            n_ij = np.zeros(K)
            mask = (S[:-1] == i)
            for j in range(K):
                n_ij[j] = np.sum(mask & (S[1:] == j))
            Gamma[i] = rng.dirichlet(np.ones(K) + n_ij)

        for k in range(K):
            y_k = Y[S == k]
            if len(y_k) < 2:
                continue
            y_k = np.ascontiguousarray(y_k, dtype=np.float64)
            log_curr = np.array([np.log(a[k]), np.log(b[k]), np.log(q[k])])
            L = np.linalg.cholesky(prop_cov[k])
            log_prop = log_curr + L @ rng.standard_normal(3)
            ltc = log_target(log_curr, y_k)
            ltp = log_target(log_prop, y_k)
            n_prop[k] += 1
            if np.log(rng.uniform()) < ltp - ltc:
                a[k] = np.exp(log_prop[0])
                b[k] = np.exp(log_prop[1])
                q[k] = np.exp(log_prop[2])
                n_acc[k] += 1
                hist[k].append(log_prop)
            else:
                hist[k].append(log_curr)

            if it < 0.6*burn and (it+1) % 100 == 0 and len(hist[k]) >= 200:
                h = np.array(hist[k][-500:])
                prop_cov[k] = (2.38**2/3.0) * np.cov(h.T) + 1e-5*np.eye(3)

        if it >= burn:
            samples['a'].append(a.copy())
            samples['b'].append(b.copy())
            samples['q'].append(q.copy())
            samples['pi'].append(pi.copy())
            samples['Gamma'].append(Gamma.copy())
            samples['S'].append(S.copy())

        if verbose and (it+1) % 2000 == 0:
            acc = n_acc / np.maximum(n_prop, 1)
            print(f"      iter {it+1}/{n_iter} | {time.time()-t0:.0f}s | "
                  f"acc=[" + " ".join(f"{x:.2f}" for x in acc) + "]")

    for key in samples:
        samples[key] = np.array(samples[key])
    return samples


# ============================================================
# PART 4: WAIC / DIC / LOO
# ============================================================
def compute_criteria(samples, Y, K, n_sub=300):
    T = len(Y)
    n_post = samples['a'].shape[0]
    idx = np.random.choice(n_post, size=min(n_sub, n_post), replace=False)
    log_lik = np.zeros((len(idx), T))

    for i, ix in enumerate(idx):
        a_k, b_k, q_k = samples['a'][ix], samples['b'][ix], samples['q'][ix]
        pi_k, G_k = samples['pi'][ix], samples['Gamma'][ix]
        log_em = np.zeros((T, K))
        for k in range(K):
            log_yb = np.log(Y) - np.log(b_k[k])
            alyb = a_k[k] * log_yb
            l1p = np.logaddexp(0, alyb)
            log_em[:, k] = (np.log(a_k[k]) + np.log(q_k[k])
                            - np.log(b_k[k])
                            + (a_k[k] - 1.0) * log_yb
                            - (q_k[k] + 1.0) * l1p)
        log_pi, log_G = np.log(pi_k), np.log(G_k)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_em[0]
        for t in range(1, T):
            m = log_alpha[t-1][:, None] + log_G
            log_alpha[t] = logsumexp(m, axis=0) + log_em[t]
        log_p_y = np.zeros(T)
        log_p_y[0] = logsumexp(log_alpha[0])
        for t in range(1, T):
            log_p_y[t] = logsumexp(log_alpha[t]) - logsumexp(log_alpha[t-1])
        log_lik[i] = log_p_y

    log_lik = np.clip(log_lik, -1e10, 100.0)
    lppd = (logsumexp(log_lik, axis=0) - np.log(len(idx))).sum()
    p_waic = log_lik.var(axis=0).sum()
    waic = -2.0 * (lppd - p_waic)
    D_bar = -2.0 * log_lik.mean(axis=0).sum()
    p_D = D_bar - (-2.0 * lppd)
    dic = D_bar + p_D
    try:
        S_n = log_lik.shape[0]
        log_p_loo_t = np.log(S_n) - logsumexp(-log_lik, axis=0)
        loo = -2.0 * log_p_loo_t.sum()
    except Exception:
        loo = np.nan
    if waic < 0 or not np.isfinite(waic):
        waic = dic = loo = np.nan
    return waic, dic, loo, lppd, p_waic


# ============================================================
# PART 5: AUTOMATIC K-SELECTION (Burnham & Anderson parsimony)
# ============================================================
def run_k_selection(Y, K_values, n_iter, burn, seed,
                    parsimony_tol=2.0):
    print("\n" + "=" * 78)
    print("AUTOMATIC K-SELECTION")
    print("=" * 78)

    results = {}
    for K_test in K_values:
        print(f"\n{'-' * 78}")
        print(f"  [K = {K_test}]  (single chain, {n_iter} iters)")
        print(f"{'-' * 78}")
        t0 = time.time()
        samples = run_mcmc(Y, K=K_test, n_iter=n_iter,
                           burn=burn, seed=seed, verbose=False)
        elapsed = time.time() - t0
        waic, dic, loo, lppd, p_waic = compute_criteria(
            samples, Y, K_test)

        n_params = 3*K_test + K_test*(K_test - 1) + (K_test - 1)
        results[K_test] = {
            'WAIC': waic, 'DIC': dic, 'LOO': loo,
            'lppd': lppd, 'p_waic': p_waic,
            'n_params': n_params, 'time': elapsed,
            'samples': samples,
        }
        print(f"  K={K_test}: WAIC={waic:.2f}  DIC={dic:.2f}  "
              f"LOO={loo:.2f}  #params={n_params}  ({elapsed:.0f}s)")

    print("\n" + "=" * 78)
    print("K-SELECTION SUMMARY")
    print("=" * 78)
    print(f"{'K':<4}{'# params':>10}{'WAIC':>12}{'DIC':>12}{'LOO':>12}")
    print("-" * 78)
    for K_test, r in results.items():
        print(f"{K_test:<4}{r['n_params']:>10}"
              f"{r['WAIC']:>12.2f}{r['DIC']:>12.2f}{r['LOO']:>12.2f}")

    def _pick(results, criterion):
        vals = {K: r[criterion] for K, r in results.items()
                if np.isfinite(r[criterion])}
        if not vals:
            return None
        min_val = min(vals.values())
        candidates = [K for K, v in vals.items()
                      if v - min_val < parsimony_tol]
        return min(candidates)

    best_by_waic = _pick(results, 'WAIC')
    best_by_dic  = _pick(results, 'DIC')
    best_by_loo  = _pick(results, 'LOO')
    votes = [best_by_waic, best_by_dic, best_by_loo]
    vote_counts = Counter(votes)
    best_K = vote_counts.most_common(1)[0][0]

    print("\n" + "=" * 78)
    print(f"PARSIMONY-ADJUSTED SELECTION  (ΔIC < {parsimony_tol})")
    print("=" * 78)
    print(f"  Best K by WAIC : {best_by_waic}")
    print(f"  Best K by DIC  : {best_by_dic}")
    print(f"  Best K by LOO  : {best_by_loo}")
    print(f"  Votes          : {dict(vote_counts)}")
    print(f"  ─────────────────────────")
    print(f"  >>> FINAL K = {best_K} <<<")

    for crit in ['WAIC', 'DIC', 'LOO']:
        vals = {K: r[crit] for K, r in results.items()
                if np.isfinite(r[crit])}
        if not vals: continue
        mn = min(vals.values())
        print(f"\n  Δ{crit} relative to best:")
        for K_test in sorted(vals.keys()):
            d = vals[K_test] - mn
            tag = ("(best)" if d == 0 else
                   "negligible" if d < 2 else
                   "positive" if d < 10 else "decisive")
            print(f"    K={K_test}: Δ{crit} = {d:>8.2f}  [{tag}]")

    df_K = pd.DataFrame([{
        'K': K_test,
        '# params': r['n_params'],
        'WAIC': round(r['WAIC'], 2) if np.isfinite(r['WAIC']) else np.nan,
        'DIC':  round(r['DIC'], 2)  if np.isfinite(r['DIC'])  else np.nan,
        'LOO':  round(r['LOO'], 2)  if np.isfinite(r['LOO'])  else np.nan,
    } for K_test, r in results.items()])

    return results, df_K, best_K


# ============================================================
# PART 6: PLOT K-SELECTION
# ============================================================
def plot_k_selection(df_K, best_K, save_path):
    print("\n[PLOT] Generating K-selection plots...")
    Ks = df_K['K'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ['WAIC', 'DIC', 'LOO']):
        vals = df_K[col].values.astype(float)
        ax.plot(Ks, vals, 'o-', lw=2, ms=10, color='steelblue',
                markeredgecolor='black', markeredgewidth=1.5)
        bi = np.nanargmin(vals)
        ax.scatter([Ks[bi]], [vals[bi]], color='red', s=250,
                   zorder=5, marker='*', edgecolor='black', lw=1.4,
                   label=f'min at K={Ks[bi]}')
        if best_K in Ks:
            idx = np.where(Ks == best_K)[0][0]
            ax.scatter([Ks[idx]], [vals[idx]], color='green',
                       s=200, zorder=6, marker='D',
                       edgecolor='black', lw=1.4,
                       label=f'chosen K={best_K}')
        ax.set_xlabel('K'); ax.set_ylabel(col)
        ax.set_title(f'{col} vs K', fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xticks(Ks)

    plt.suptitle('K-selection (parsimony rule applied)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ============================================================
# PART 7: FINAL MODEL on selected K
# ============================================================
def _match_states_by_b(samples):
    b_mean = samples['b'].mean(axis=0)
    order = np.argsort(b_mean)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = {}
    for key in ['a', 'b', 'q']:
        out[key] = samples[key][:, order]
    out['pi'] = samples['pi'][:, order]
    out['Gamma'] = samples['Gamma'][:, order][:, :, order]
    out['S'] = inv[samples['S']]
    return out


def run_multiple_chains(Y, K, n_chains, n_iter, burn, base_seed=42):
    print(f"\n[FINAL] Running {n_chains} chains "
          f"(K = {K}, n_iter = {n_iter}, burn = {burn})")
    chains = []
    for c in range(n_chains):
        seed = base_seed + 1000 * c
        print(f"  Chain {c+1}/{n_chains} (seed={seed}) ...",
              end=" ", flush=True)
        t0 = time.time()
        s = run_mcmc(Y, K=K, n_iter=n_iter, burn=burn,
                     seed=seed, verbose=False)
        s = _match_states_by_b(s)
        chains.append(s)
        print(f"done in {time.time()-t0:.1f}s")

    b_matrix = np.array([ch['b'].mean(axis=0) for ch in chains])
    print("\n  Posterior mean of b per chain (mm):")
    print("  " + str(np.round(b_matrix, 2)).replace('\n', '\n  '))
    spread = (b_matrix.max(axis=0) - b_matrix.min(axis=0)) \
             / b_matrix.mean(axis=0)
    print(f"  Max spread: {100*spread.max():.2f}% "
          f"({'OK' if spread.max() < 0.10 else 'WARN'})")
    return chains


# ============================================================
# PART 8: PUBLICATION-QUALITY FIGURES
# ============================================================
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.dpi'] = 300
from matplotlib.patches import Patch

FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
_PALETTE = plt.cm.viridis(np.linspace(0.15, 0.90, 8))


def _sm_pdf(y, a, b, q):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    mask = y > 0
    yy = y[mask]
    log_yb = np.log(yy) - np.log(b)
    alyb = a * log_yb
    l1p = np.logaddexp(0.0, alyb)
    out[mask] = np.exp(np.log(a) + np.log(q) - np.log(b)
                       + (a - 1.0) * log_yb - (q + 1.0) * l1p)
    return out


def plot_timeseries_regimes(Y, dates, samples, K, save_path):
    S_samples = samples['S']; n_post, T = S_samples.shape
    S_prob = np.zeros((T, K))
    for t in range(T):
        S_prob[t] = np.bincount(S_samples[:, t], minlength=K) / n_post
    S_mode = S_prob.argmax(axis=1)
    colors = _PALETTE[:K]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    ax = axes[0]
    for t in range(T - 1):
        ax.axvspan(dates[t], dates[t + 1],
                   color=colors[S_mode[t]], alpha=0.28, lw=0)
    ax.plot(dates, Y, 'k-', lw=1.3, zorder=5)
    ax.set_ylabel('Monthly rainfall (mm)', fontsize=12)
    ax.set_title(f'Decoded rainfall regimes (K = {K})', fontsize=13)
    handles = [Patch(facecolor=colors[k], alpha=0.6, edgecolor='black',
                     label=f'State {k+1}') for k in range(K)]
    ax.legend(handles=handles, loc='upper right', ncol=K, fontsize=10)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    bottom = np.zeros(T)
    for k in range(K):
        ax2.fill_between(dates, bottom, bottom + S_prob[:, k],
                         color=colors[k], alpha=0.85, lw=0)
        bottom = bottom + S_prob[:, k]
    ax2.set_ylabel('Posterior state\nprobability', fontsize=11)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(0, 1); ax2.set_xlim(dates[0], dates[-1])
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_parameter_posteriors(samples, K, save_path):
    fig, axes = plt.subplots(3, K, figsize=(3.8 * K, 9))
    if K == 1: axes = axes.reshape(3, 1)
    params = [('a', r'$a$'), ('b', r'$b$ (mm)'), ('q', r'$q$')]
    colors = _PALETTE[:K]
    for i, (pname, plab) in enumerate(params):
        for k in range(K):
            ax = axes[i, k]
            x = samples[pname][:, k]
            ax.hist(x, bins=40, density=True, color=colors[k],
                    alpha=0.75, edgecolor='black', lw=0.4)
            ax.axvline(x.mean(), color='red', ls='--', lw=2.0,
                       label=f'mean={x.mean():.3f}')
            ax.axvline(np.percentile(x, 2.5), color='red', ls=':', lw=1.4)
            ax.axvline(np.percentile(x, 97.5), color='red', ls=':', lw=1.4)
            ax.set_xlabel(plab, fontsize=11)
            if k == 0: ax.set_ylabel('Density', fontsize=11)
            if i == 0: ax.set_title(f'State {k+1}', fontsize=11)
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle('Posterior distributions of SM parameters',
                 fontsize=13, y=1.005)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_transition_matrix(samples, K, save_path):
    G_mean = samples['Gamma'].mean(axis=0)
    G_std = samples['Gamma'].std(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(G_mean, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Posterior mean', fontsize=12)
    for i in range(K):
        for j in range(K):
            axes[0].text(j, i, f'{G_mean[i,j]:.2f}', ha='center',
                         va='center', fontsize=11,
                         color='white' if G_mean[i, j] > 0.55 else 'black')
    im2 = axes[1].imshow(G_std, cmap='Reds', vmin=0)
    axes[1].set_title('Posterior SD', fontsize=12)
    for i in range(K):
        for j in range(K):
            axes[1].text(j, i, f'{G_std[i,j]:.3f}', ha='center',
                         va='center', fontsize=10)
    for ax in axes:
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f'S{i+1}' for i in range(K)])
        ax.set_yticklabels([f'S{i+1}' for i in range(K)])
        ax.set_xlabel('To'); ax.set_ylabel('From')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_trace(samples, K, save_path):
    fig, axes = plt.subplots(3, K, figsize=(3.8 * K, 8), sharex=True)
    if K == 1: axes = axes.reshape(3, 1)
    params = [('a', r'$a$'), ('b', r'$b$'), ('q', r'$q$')]
    colors = _PALETTE[:K]
    for i, (pname, plab) in enumerate(params):
        for k in range(K):
            ax = axes[i, k]
            ax.plot(samples[pname][:, k], color=colors[k],
                    lw=0.55, alpha=0.85)
            ax.axhline(samples[pname][:, k].mean(), color='red',
                       ls='--', lw=1.0)
            ax.set_ylabel(plab, fontsize=11)
            if i == 0: ax.set_title(f'State {k+1}', fontsize=11)
            if i == 2: ax.set_xlabel('Post-burn iteration', fontsize=10)
            ax.grid(alpha=0.3)
    plt.suptitle('Trace plots', fontsize=13, y=1.005)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_acf(samples, K, save_path, max_lag=80):
    def _acf(x, lag):
        x = x - x.mean(); n = len(x); d = np.sum(x * x)
        return np.array([np.sum(x[:n-l] * x[l:]) / d
                         for l in range(lag + 1)])
    fig, axes = plt.subplots(3, K, figsize=(3.8 * K, 8))
    if K == 1: axes = axes.reshape(3, 1)
    for i, pname in enumerate(['a', 'b', 'q']):
        for k in range(K):
            ax = axes[i, k]
            x = samples[pname][:, k]
            r = _acf(x, max_lag)
            ci = 1.96 / np.sqrt(len(x))
            ax.bar(range(len(r)), r, width=1.0, color='steelblue')
            ax.axhline(0, color='black', lw=0.5)
            ax.axhline(ci, color='red', ls='--', lw=0.8)
            ax.axhline(-ci, color='red', ls='--', lw=0.8)
            ax.set_ylabel(f'ACF({pname})', fontsize=10)
            if i == 0: ax.set_title(f'State {k+1}', fontsize=11)
            if i == 2: ax.set_xlabel('Lag', fontsize=10)
            ax.grid(alpha=0.3)
    plt.suptitle('Autocorrelation functions', fontsize=13, y=1.005)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_state_densities(samples, Y, K, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = _PALETTE[:K]
    y_grid = np.linspace(0.01, max(Y.max() * 1.15, 1.0), 600)

    ax = axes[0]
    for k in range(K):
        a_k = samples['a'][:, k].mean()
        b_k = samples['b'][:, k].mean()
        q_k = samples['q'][:, k].mean()
        ax.plot(y_grid, _sm_pdf(y_grid, a_k, b_k, q_k),
                color=colors[k], lw=2.5,
                label=f'S{k+1}: a={a_k:.2f}, b={b_k:.1f} mm, q={q_k:.2f}')
    ax.set_xlabel('Monthly rainfall (mm)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('State-specific SM densities', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    S_last = samples['S'][-1]
    bins = np.linspace(0, Y.max() * 1.1, 30)
    for k in range(K):
        y_k = Y[S_last == k]
        if len(y_k) > 0:
            ax.hist(y_k, bins=bins, alpha=0.55, color=colors[k],
                    label=f'S{k+1} (n={len(y_k)})',
                    density=True, edgecolor='black', lw=0.4)
    ax.set_xlabel('Monthly rainfall (mm)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Empirical distribution by decoded state', fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_monthly_heatmap(samples, dates, K, save_path):
    S_samples = samples['S']; n_post, T = S_samples.shape
    months = pd.DatetimeIndex(dates).month
    M = np.zeros((12, K))
    for t in range(T):
        M[months[t] - 1] += np.bincount(S_samples[:, t],
                                        minlength=K) / n_post
    row = M.sum(axis=1, keepdims=True)
    Mn = M / np.maximum(row, 1e-10)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(max(5, K*2.2), 6))
    im = ax.imshow(Mn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'S{i+1}' for i in range(K)], fontsize=11)
    ax.set_yticks(range(12)); ax.set_yticklabels(month_names, fontsize=11)
    for i in range(12):
        for j in range(K):
            ax.text(j, i, f'{Mn[i,j]:.2f}', ha='center', va='center',
                    fontsize=10,
                    color='white' if Mn[i, j] > 0.6 else 'black')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    ax.set_title('Monthly regime occurrence probability', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_stationary(samples, K, save_path):
    G_samples = samples['Gamma']
    n_post = G_samples.shape[0]
    stat = np.zeros((n_post, K))
    for i in range(n_post):
        G = G_samples[i]
        A = np.vstack([G.T - np.eye(K), np.ones(K)])
        b_vec = np.concatenate([np.zeros(K), [1.0]])
        sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
        stat[i] = np.maximum(sol, 0); stat[i] /= stat[i].sum()
    means = stat.mean(axis=0)
    lo = np.percentile(stat, 2.5, axis=0)
    hi = np.percentile(stat, 97.5, axis=0)
    colors = _PALETTE[:K]
    fig, ax = plt.subplots(figsize=(max(5, K*1.8), 5))
    x = np.arange(K)
    ax.bar(x, means, color=colors, alpha=0.9, edgecolor='black', lw=1.3)
    ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt='none',
                ecolor='red', capsize=8, lw=2.2)
    for i in range(K):
        ax.text(i, hi[i] + 0.015, f'{means[i]:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'State {i+1}' for i in range(K)], fontsize=11)
    ax.set_ylabel('Stationary probability', fontsize=12)
    ax.set_title('Stationary distribution (mean & 95% CI)', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_ppc(samples, Y, K, save_path, n_sim=200):
    T = len(Y)
    n_post = samples['a'].shape[0]
    idx = np.random.choice(n_post, min(n_sim, n_post), replace=False)
    y_grid = np.linspace(0.01, Y.max() * 1.5, 1000)
    sim_mean, sim_max, sim_dry = [], [], []
    for ix in idx:
        a_k, b_k = samples['a'][ix], samples['b'][ix]
        q_k, pi_k = samples['q'][ix], samples['pi'][ix]
        G_k = samples['Gamma'][ix]
        S = np.zeros(T, dtype=int)
        S[0] = np.random.choice(K, p=pi_k)
        for t in range(1, T):
            S[t] = np.random.choice(K, p=G_k[S[t-1]])
        Y_sim = np.zeros(T)
        for t in range(T):
            k = S[t]
            pdf = _sm_pdf(y_grid, a_k[k], b_k[k], q_k[k])
            cdf = np.cumsum(pdf)
            if cdf[-1] <= 0:
                Y_sim[t] = 0.0; continue
            cdf = cdf / cdf[-1]
            Y_sim[t] = y_grid[np.searchsorted(cdf, np.random.rand())]
        sim_mean.append(Y_sim.mean())
        sim_max.append(Y_sim.max())
        sim_dry.append(np.mean(Y_sim < 40.0))
    stats = [('Mean (mm)', sim_mean, Y.mean()),
             ('Max (mm)', sim_max, Y.max()),
             ('P(Y<40mm)', sim_dry, np.mean(Y < 40.0))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (lab, sim, obs) in zip(axes, stats):
        ax.hist(sim, bins=25, color='steelblue', alpha=0.75,
                edgecolor='black', density=True)
        ax.axvline(obs, color='red', lw=2.5, ls='--',
                   label=f'Observed = {obs:.2f}')
        ax.set_xlabel(lab, fontsize=11); ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'PPC: {lab}', fontsize=11)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.suptitle('Posterior predictive checks', fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def generate_all_figures(Y, dates, samples, K):
    print("\n" + "=" * 78)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 78)
    jobs = [
        ("fig01_timeseries_regimes.png",
         lambda p: plot_timeseries_regimes(Y, dates, samples, K, p)),
        ("fig02_param_posteriors.png",
         lambda p: plot_parameter_posteriors(samples, K, p)),
        ("fig03_transition_matrix.png",
         lambda p: plot_transition_matrix(samples, K, p)),
        ("fig04_trace.png", lambda p: plot_trace(samples, K, p)),
        ("fig05_acf.png", lambda p: plot_acf(samples, K, p)),
        ("fig06_state_densities.png",
         lambda p: plot_state_densities(samples, Y, K, p)),
        ("fig07_monthly_heatmap.png",
         lambda p: plot_monthly_heatmap(samples, dates, K, p)),
        ("fig08_stationary.png", lambda p: plot_stationary(samples, K, p)),
        ("fig09_ppc.png", lambda p: plot_ppc(samples, Y, K, p)),
    ]
    for i, (name, fn) in enumerate(jobs, 1):
        print(f"  [{i:02d}/{len(jobs)}] {name} ...", end=" ", flush=True)
        fn(os.path.join(FIG_DIR, name))
        print("OK")
    print(f"\n  All figures saved to: {FIG_DIR}")


# ============================================================
# PART 9: CONVERGENCE DIAGNOSTICS
# ============================================================
def rhat_classic(chains_list):
    m, n = len(chains_list), len(chains_list[0])
    h = n // 2
    arr = np.zeros((2 * m, h))
    for i, c in enumerate(chains_list):
        arr[2 * i] = c[:h]
        arr[2 * i + 1] = c[h:2 * h]
    cm = arr.mean(axis=1); cv = arr.var(axis=1, ddof=1)
    W = cv.mean(); B = h * cm.var(ddof=1)
    var_hat = (h - 1) / h * W + B / h
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan


def rhat_rank_normalized(chains_list, folded=False):
    if folded:
        all_vals = np.concatenate(chains_list)
        med = np.median(all_vals)
        chains_list = [np.abs(c - med) for c in chains_list]
    all_vals = np.concatenate(chains_list)
    n_total = len(all_vals)
    ranks = rankdata(all_vals, method='average')
    z = norm.ppf((ranks - 3.0/8.0) / (n_total + 0.25))
    norm_chains, idx = [], 0
    for c in chains_list:
        nc = len(c)
        norm_chains.append(z[idx:idx+nc]); idx += nc
    return rhat_classic(norm_chains)


def ess_basic(chains_list):
    m, n = len(chains_list), len(chains_list[0])
    h = n // 2
    split = []
    for c in chains_list:
        split.append(c[:h]); split.append(c[h:2*h])
    m2, ns = len(split), h
    max_lag = min(ns - 2, 500)
    rho = np.zeros(max_lag + 1)
    for c in split:
        x = c - c.mean()
        var = np.sum(x * x)
        if var <= 0: continue
        for t in range(max_lag + 1):
            rho[t] += np.sum(x[:ns - t] * x[t:]) / var
    rho /= m2
    tau = 0.0
    for t in range(1, max_lag - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0: break
        tau += pair
    return max(m2 * ns / (1.0 + 2.0 * tau), 1.0)


def geweke_z(x, first=0.1, last=0.5):
    n = len(x); n1 = int(first * n); n2 = int(last * n)
    x1, x2 = x[:n1], x[-n2:]
    def s0(xx):
        nn = len(xx); xx = xx - xx.mean()
        v = np.sum(xx * xx) / nn
        L = max(int(np.sqrt(nn)), 1)
        for l in range(1, L):
            w = 1.0 - l / L
            v += 2.0 * w * np.sum(xx[:nn - l] * xx[l:]) / nn
        return max(v, 1e-12)
    return float((x1.mean() - x2.mean())
                 / np.sqrt(s0(x1)/n1 + s0(x2)/n2))


def compute_all_diagnostics(chains, K):
    results = {}
    for pname in ['a', 'b', 'q', 'pi']:
        for k in range(K):
            series = [ch[pname][:, k] for ch in chains]
            gz = [geweke_z(s) for s in series]
            results[f'{pname}[{k+1}]'] = {
                'Rhat': rhat_rank_normalized(series, False),
                'Rhat_folded': rhat_rank_normalized(series, True),
                'Rhat_classic': rhat_classic(series),
                'ESS': ess_basic(series),
                'Geweke_mean': float(np.mean(np.abs(gz))),
                'Geweke_max': float(np.max(np.abs(gz))),
            }
    return results


def print_diagnostics_table(diag):
    print("\n" + "=" * 78)
    print("MCMC CONVERGENCE DIAGNOSTICS")
    print("=" * 78)
    print(f"{'Parameter':<12}{'R-hat':>9}{'R-hat_fold':>13}"
          f"{'ESS':>10}{'|Geweke|max':>15}{'Status':>10}")
    print("-" * 78)
    n_bad = 0
    for pname, d in diag.items():
        ok = (d['Rhat'] < 1.01 and d['Rhat_folded'] < 1.01
              and d['ESS'] > 400.0 and d['Geweke_max'] < 1.96)
        status = "OK" if ok else "CHECK"
        if not ok: n_bad += 1
        print(f"{pname:<12}{d['Rhat']:>9.4f}"
              f"{d['Rhat_folded']:>13.4f}"
              f"{d['ESS']:>10.1f}"
              f"{d['Geweke_max']:>15.3f}"
              f"{status:>10}")
    print("-" * 78)
    if n_bad == 0:
        print("  All parameters converged (R-hat < 1.01, ESS > 400)")
    else:
        print(f"  WARNING: {n_bad} parameter(s) need more iterations.")
    return n_bad


def plot_convergence_diagnostics(chains, diag, K, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    param_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    pnames = ['a', 'b', 'q']
    x, w = np.arange(K), 0.30

    ax = axes[0, 0]
    for i, p in enumerate(pnames):
        vals = [diag[f'{p}[{k+1}]']['Rhat'] for k in range(K)]
        ax.bar(x + (i - 1) * w, vals, w, color=param_colors[i],
               edgecolor='black', alpha=0.88, label=f'R-hat ({p})')
    ax.axhline(1.01, color='red', ls='--', lw=1.6, label='Threshold 1.01')
    ax.set_xticks(x); ax.set_xticklabels([f'S{k+1}' for k in range(K)])
    ax.set_ylabel('Rank-normalized R-hat')
    ax.set_title('Split-R-hat', fontsize=11)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3, axis='y')

    ax = axes[0, 1]
    for i, p in enumerate(pnames):
        vals = [diag[f'{p}[{k+1}]']['Rhat_folded'] for k in range(K)]
        ax.bar(x + (i - 1) * w, vals, w, color=param_colors[i],
               edgecolor='black', alpha=0.88, label=f'Folded {p}')
    ax.axhline(1.01, color='red', ls='--', lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels([f'S{k+1}' for k in range(K)])
    ax.set_ylabel('Folded R-hat'); ax.set_title('Folded R-hat', fontsize=11)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 0]
    for i, p in enumerate(pnames):
        vals = [diag[f'{p}[{k+1}]']['ESS'] for k in range(K)]
        ax.bar(x + (i - 1) * w, vals, w, color=param_colors[i],
               edgecolor='black', alpha=0.88, label=f'ESS ({p})')
    ax.axhline(400, color='orange', ls='--', lw=1.4, label='Min 400')
    ax.axhline(1000, color='green', ls=':', lw=1.4, label='Good 1000')
    ax.set_xticks(x); ax.set_xticklabels([f'S{k+1}' for k in range(K)])
    ax.set_ylabel('Effective sample size')
    ax.set_title('ESS', fontsize=11)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 1]
    all_z, labels = [], []
    for p in pnames:
        for k in range(K):
            zs = [geweke_z(ch[p][:, k]) for ch in chains]
            all_z.append(float(np.mean(zs)))
            labels.append(f'{p}[{k+1}]')
    cols = ['#2ca02c' if abs(z) < 1.96 else '#d62728' for z in all_z]
    ax.scatter(range(len(all_z)), all_z, c=cols, s=90,
               edgecolor='black', lw=1.2, zorder=5)
    ax.axhline(1.96, color='red', ls='--', lw=1.5, alpha=0.75)
    ax.axhline(-1.96, color='red', ls='--', lw=1.5, alpha=0.75)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(range(len(all_z)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Geweke z (mean)')
    ax.set_title('Geweke diagnostic', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'MCMC convergence (K = {K}, {len(chains)} chains)',
                 fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_multi_chain_traces(chains, K, save_path):
    fig, axes = plt.subplots(3, K, figsize=(3.8 * K, 9), sharex=True)
    if K == 1: axes = axes.reshape(3, 1)
    ch_colors = plt.cm.tab10(np.linspace(0.05, 0.9, len(chains)))
    for i, p in enumerate(['a', 'b', 'q']):
        for k in range(K):
            ax = axes[i, k]
            for c, ch in enumerate(chains):
                ax.plot(ch[p][:, k], color=ch_colors[c],
                        lw=0.55, alpha=0.8, label=f'Chain {c+1}')
            ax.set_ylabel(p, fontsize=11)
            if i == 0: ax.set_title(f'State {k+1}', fontsize=11)
            if i == 2: ax.set_xlabel('Post-burn iteration', fontsize=10)
            if i == 0 and k == 0:
                ax.legend(fontsize=8, ncol=2, loc='upper right')
            ax.grid(alpha=0.3)
    plt.suptitle(f'Multi-chain traces ({len(chains)} chains)',
                 fontsize=13, y=1.005)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


# ============================================================
# PART 10: EXTENDED SUMMARY (numerically stabilised)
# ============================================================
def compute_extended_summary(final_chains, Y, K,
                             duration_cap=DURATION_CAP,
                             mad_k=MAD_K):
    """
    Comprehensive publication table. Includes:
      - state parameters a, b, q
      - initial probabilities pi
      - transition matrix Gamma
      - stationary distribution pi*
      - expected duration (clipped at duration_cap)
      - E[Y|S=k] in log-space with MAD-based clipping
      - Median[Y|S=k] by numerical CDF inversion
    """
    all_a  = np.vstack([ch['a']     for ch in final_chains])
    all_b  = np.vstack([ch['b']     for ch in final_chains])
    all_q  = np.vstack([ch['q']     for ch in final_chains])
    all_pi = np.vstack([ch['pi']    for ch in final_chains])
    all_G  = np.vstack([ch['Gamma'] for ch in final_chains])
    n_post = all_a.shape[0]

    rows = []
    def add_row(parameter, samples, note=""):
        samples = np.asarray(samples, dtype=float)
        finite = samples[np.isfinite(samples)]
        if len(finite) == 0:
            rows.append({'Parameter': parameter,
                         'Mean': np.nan, 'SD': np.nan,
                         '2.5%': np.nan, 'Median': np.nan,
                         '97.5%': np.nan, 'Note': note})
            return
        rows.append({
            'Parameter': parameter,
            'Mean':   round(float(np.mean(finite)), 4),
            'SD':     round(float(np.std(finite)), 4),
            '2.5%':   round(float(np.percentile(finite, 2.5)), 4),
            'Median': round(float(np.median(finite)), 4),
            '97.5%':  round(float(np.percentile(finite, 97.5)), 4),
            'Note':   note,
        })

    # ---- 1. State parameters ----
    for k in range(K): add_row(f'a[{k+1}]', all_a[:, k])
    for k in range(K): add_row(f'b[{k+1}] (mm)', all_b[:, k])
    for k in range(K): add_row(f'q[{k+1}]', all_q[:, k])

    # ---- 2. Initial probabilities ----
    for k in range(K): add_row(f'pi[{k+1}] (initial)', all_pi[:, k])

    # ---- 3. Transition matrix ----
    for i in range(K):
        for j in range(K):
            add_row(f'Gamma[{i+1},{j+1}]', all_G[:, i, j])

    # ---- 4. Stationary distribution ----
    stat = np.zeros((n_post, K))
    for i in range(n_post):
        G = all_G[i]
        A = np.vstack([G.T - np.eye(K), np.ones(K)])
        b_vec = np.concatenate([np.zeros(K), [1.0]])
        sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
        sol = np.maximum(sol, 0)
        stat[i] = sol / sol.sum() if sol.sum() > 0 else np.ones(K)/K
    for k in range(K):
        add_row(f'pi*[{k+1}] (stationary)', stat[:, k])

    # ---- 5. Duration (clipped) ----
    for k in range(K):
        raw = 1.0 / np.maximum(1.0 - all_G[:, k, k], 1e-6)
        dur = np.clip(raw, 0.0, duration_cap)
        add_row(f'Duration[{k+1}] (months)', dur,
                note=f'clipped at {duration_cap}')

    # ---- 6. E[Y|S=k] — log-space with MAD-based clipping ----
    print("\n  [E[Y|S=k]] log-space computation ...")
    E_Y = np.full((n_post, K), np.nan)
    for i in range(n_post):
        for k in range(K):
            a_ = all_a[i, k]; b_ = all_b[i, k]; q_ = all_q[i, k]
            if q_ * a_ > 1.01:   # moment defined
                log_mean = (np.log(b_)
                            + gammaln(1.0 + 1.0 / a_)
                            + gammaln(q_ - 1.0 / a_)
                            - gammaln(q_))
                E_Y[i, k] = np.exp(log_mean)

    for k in range(K):
        col = E_Y[:, k]
        finite = col[np.isfinite(col)]
        n_nan_before = np.isnan(col).sum()
        if len(finite) > 0:
            med = np.median(finite)
            mad = np.median(np.abs(finite - med))
            if mad > 0:
                lo = med - mad_k * mad
                hi = med + mad_k * mad
                col = np.clip(col, lo, hi)
                print(f"    E[Y|S={k+1}]: MAD-clipped to "
                      f"[{lo:.2f}, {hi:.2f}] (median={med:.2f}, "
                      f"MAD={mad:.2f})")
            # NaN replacement with posterior median
            mask = ~np.isfinite(col)
            if mask.any():
                col[mask] = med
                print(f"    E[Y|S={k+1}]: {mask.sum()} NaN replaced "
                      f"by median ({med:.2f} mm)")
            E_Y[:, k] = col

    for k in range(K):
        add_row(f'E[Y|S={k+1}] (mm)', E_Y[:, k])

    # ---- 7. Median[Y|S=k] via CDF inversion ----
    print("  [Median[Y|S=k]] numerical CDF inversion ...")
    y_grid = np.linspace(0.01, Y.max() * 1.5, 2000)
    med_Y = np.full((n_post, K), np.nan)
    for i in range(n_post):
        for k in range(K):
            a_ = all_a[i, k]; b_ = all_b[i, k]; q_ = all_q[i, k]
            pdf = _sm_pdf(y_grid, a_, b_, q_)
            cdf = np.cumsum(pdf)
            if cdf[-1] <= 0:
                continue
            cdf /= cdf[-1]
            med_Y[i, k] = y_grid[np.searchsorted(cdf, 0.5)]
    for k in range(K):
        add_row(f'Median[Y|S={k+1}] (mm)', med_Y[:, k])

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---- STEP 0 ----
    download_missing_years()

    # ---- STEP 1 ----
    Y_daily, dates_daily = load_daily_data()

    # ---- STEP 2 ----
    Y, dates = aggregate_to_monthly(Y_daily, dates_daily, agg="total")
    if len(Y) < 100:
        print(f"WARNING: only {len(Y)} monthly points — too few")
        exit()

    Y_sm = Y.copy()
    Y_sm[Y_sm <= 0] = 0.1
    print(f"  Zeros replaced with 0.1 mm")

    # ============================================================
    # STEP 3: AUTOMATIC K-SELECTION
    # ============================================================
    results_K, df_K, best_K = run_k_selection(
        Y_sm, K_VALUES,
        n_iter=K_SEL_ITER, burn=K_SEL_BURN, seed=SEED,
        parsimony_tol=PARSIMONY_TOL,
    )

    df_K.to_csv(os.path.join(BASE_DIR, "tables", "k_selection.csv"),
                index=False)
    print(f"\n  Saved: k_selection.csv")

    plot_k_selection(df_K, best_K,
                     os.path.join(FIG_DIR, "k_selection.png"))

    # Free memory
    for K_test in list(results_K.keys()):
        results_K[K_test].pop('samples', None)

    # ============================================================
    # STEP 4: FOCUSED INFERENCE on selected K
    # ============================================================
    print("\n" + "=" * 78)
    print(f"FINAL MODEL (selected K = {best_K}, {N_CHAINS} chains)")
    print("=" * 78)

    final_chains = run_multiple_chains(
        Y_sm, K=best_K,
        n_chains=N_CHAINS,
        n_iter=N_ITER_FINAL,
        burn=BURN_FINAL,
        base_seed=SEED,
    )
    final_samples = final_chains[0]

    # Save
    save_dict = {'Y': Y, 'dates': dates.values.astype(str), 'K': best_K}
    for c, ch in enumerate(final_chains):
        for key in ['a', 'b', 'q', 'pi', 'Gamma', 'S']:
            save_dict[f'chain{c}_{key}'] = ch[key]
    np.savez(os.path.join(BASE_DIR, "hmm_sm_final.npz"), **save_dict)
    print(f"  Saved: hmm_sm_final.npz")

    # ---- Compact summary ----
    rows = []
    for name in ['a', 'b', 'q']:
        for k in range(best_K):
            x = np.concatenate([ch[name][:, k] for ch in final_chains])
            rows.append({
                'Parameter': f'{name}[{k+1}]',
                'Mean':   round(x.mean(), 4),
                'SD':     round(x.std(), 4),
                '2.5%':   round(np.percentile(x, 2.5), 4),
                '97.5%':  round(np.percentile(x, 97.5), 4),
            })
    for k in range(best_K):
        x = np.concatenate([ch['pi'][:, k] for ch in final_chains])
        rows.append({
            'Parameter': f'pi[{k+1}]',
            'Mean': round(x.mean(), 4), 'SD': round(x.std(), 4),
            '2.5%': round(np.percentile(x, 2.5), 4),
            '97.5%': round(np.percentile(x, 97.5), 4),
        })
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"FINAL SUMMARY (K = {best_K})")
    print("=" * 78)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(BASE_DIR, "tables", "final_summary.csv"),
                   index=False)
    print(f"\n  Saved: final_summary.csv")

    # ---- Extended summary (with stabilisation) ----
    print("\n" + "=" * 78)
    print(f"EXTENDED SUMMARY (K = {best_K})")
    print("=" * 78)
    extended = compute_extended_summary(final_chains, Y, best_K)
    print(extended.to_string(index=False))

    # Save both versions
    extended.to_csv(
        os.path.join(BASE_DIR, "tables", "extended_summary_clean.csv"),
        index=False)
    extended.drop(columns=['Note']).to_csv(
        os.path.join(BASE_DIR, "tables", "extended_summary.csv"),
        index=False)
    print(f"\n  Saved: extended_summary_clean.csv")
    print(f"  Saved: extended_summary.csv (publication-ready)")

    # ---- Figures ----
    generate_all_figures(Y, dates, final_samples, best_K)

    # ---- Diagnostics ----
    print("\n" + "=" * 78)
    print("MULTI-CHAIN CONVERGENCE DIAGNOSTICS")
    print("=" * 78)
    diag = compute_all_diagnostics(final_chains, best_K)
    n_bad = print_diagnostics_table(diag)

    rows_diag = []
    for pname, d in diag.items():
        rows_diag.append({
            'Parameter': pname,
            'R-hat': round(d['Rhat'], 4),
            'R-hat_folded': round(d['Rhat_folded'], 4),
            'R-hat_classic': round(d['Rhat_classic'], 4),
            'ESS': round(d['ESS'], 1),
            'Geweke_mean_abs': round(d['Geweke_mean'], 3),
            'Geweke_max_abs': round(d['Geweke_max'], 3),
        })
    pd.DataFrame(rows_diag).to_csv(
        os.path.join(BASE_DIR, "tables", "mcmc_diagnostics.csv"),
        index=False)
    print(f"  Saved: mcmc_diagnostics.csv")

    plot_convergence_diagnostics(
        final_chains, diag, best_K,
        os.path.join(FIG_DIR, "fig10_convergence_diagnostics.png"))
    plot_multi_chain_traces(
        final_chains, best_K,
        os.path.join(FIG_DIR, "fig11_multi_chain_traces.png"))
    print("  Saved: fig10, fig11")

    # ---- Model fit on selected K ----
    waic, dic, loo, lppd, p_waic = compute_criteria(
        final_samples, Y_sm, best_K)
    print("\n" + "=" * 78)
    print(f"MODEL FIT (selected K = {best_K})")
    print("=" * 78)
    print(f"  WAIC   = {waic:.2f}")
    print(f"  DIC    = {dic:.2f}")
    print(f"  LOO    = {loo:.2f}")
    print(f"  lppd   = {lppd:.2f}")
    print(f"  p_waic = {p_waic:.2f}")

    print("\n" + "=" * 78)
    print("ALL DONE ✅")
    print("=" * 78)
    print(f"Selected K        : {best_K}")
    print(f"Convergence       : "
          f"{'PASS' if n_bad == 0 else f'{n_bad} parameter(s) need attention'}")
    print(f"Outputs in        : {BASE_DIR}")
    print(f"  figures/  → k_selection.png, fig01..fig11 (300 dpi)")
    print(f"  tables/   → k_selection.csv, final_summary.csv,")
    print(f"              extended_summary.csv, extended_summary_clean.csv,")
    print(f"              mcmc_diagnostics.csv")
    print(f"  hmm_sm_final.npz ({N_CHAINS} chains)")