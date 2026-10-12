# ============================================================
# COMPETING MODELS: HMM-Lognormal and HMM-Gamma
# Runs on the saved data from hmm_sm_final.npz
# Output: tables/competing_models.csv + console report
# Time: ~20 minutes total
#
# FIXED: scipy.special.loggamma -> math.lgamma (Numba-compatible)
# ============================================================
import os
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from math import lgamma
from numba import njit
import time

BASE_DIR = r"D:\فكرة بحث 2027\irish_rainfall"
NPZ_PATH = os.path.join(BASE_DIR, "hmm_sm_final.npz")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================
# CORE FUNCTIONS
# ============================================================
@njit(fastmath=True)
def forward_filter(log_em, pi, Gamma):
    T, K = log_em.shape
    log_pi = np.log(pi)
    log_G = np.log(Gamma)
    log_alpha = np.empty((T, K))
    for k in range(K):
        log_alpha[0, k] = log_pi[k] + log_em[0, k]
    for t in range(1, T):
        for k in range(K):
            m = -np.inf
            for j in range(K):
                v = log_alpha[t-1, j] + log_G[j, k]
                if v > m:
                    m = v
            s = 0.0
            for j in range(K):
                s += np.exp(log_alpha[t-1, j] + log_G[j, k] - m)
            log_alpha[t, k] = m + np.log(s) + log_em[t, k]
    return log_alpha


def ffbs(log_em, pi, Gamma, rng):
    T, K = log_em.shape
    log_alpha = forward_filter(log_em, pi, Gamma)
    log_G = np.log(Gamma)
    S = np.empty(T, dtype=np.int64)
    probs = np.exp(log_alpha[-1] - log_alpha[-1].max())
    probs /= probs.sum()
    S[-1] = rng.choice(K, p=probs)
    for t in range(T - 2, -1, -1):
        log_w = log_alpha[t] + log_G[:, S[t + 1]]
        probs = np.exp(log_w - log_w.max())
        probs /= probs.sum()
        S[t] = rng.choice(K, p=probs)
    return S


# ============================================================
# MODEL 1: HMM-LOGNORMAL
# ============================================================
@njit(fastmath=True)
def lognormal_logpdf(y, mu, sigma):
    """LogNormal(mu, sigma) density at y>0."""
    z = (np.log(y) - mu) / sigma
    return -np.log(y) - np.log(sigma) - 0.9189385332046727 - 0.5 * z * z


def run_mcmc_lognormal(Y, K, n_iter, burn, seed, verbose=False):
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    T = len(Y)
    log_Y = np.log(Y)
    t0 = time.time()

    # Initialization
    mu = np.percentile(log_Y, np.linspace(30, 70, K))
    sigma = np.full(K, 0.5)

    pi = np.ones(K) / K
    Gamma = np.full((K, K), 1.0 / K) + 0.1 * np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.ones(K) * 2.0

    prop_cov = [np.eye(2) * 0.05 ** 2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)
    samples = {k: [] for k in ['mu', 'sigma', 'pi', 'Gamma', 'S']}

    for it in range(n_iter):
        # ===== 1. Emission matrix =====
        log_em = np.empty((T, K))
        for t in range(T):
            for k in range(K):
                log_em[t, k] = lognormal_logpdf(Y[t], mu[k], sigma[k])

        # ===== 2. FFBS =====
        S = ffbs(log_em, pi, Gamma, rng)

        # ===== 3. Conjugate pi, Gamma =====
        counts = np.bincount(S[:1], minlength=K).astype(float)
        pi = rng.dirichlet(pi_prior + counts)
        for i in range(K):
            n_ij = np.zeros(K)
            mask = (S[:-1] == i)
            for j in range(K):
                n_ij[j] = np.sum(mask & (S[1:] == j))
            Gamma[i] = rng.dirichlet(np.ones(K) + n_ij)

        # ===== 4. Metropolis for (log sigma, mu) =====
        for k in range(K):
            y_k = log_Y[S == k]
            if len(y_k) < 2:
                continue
            log_curr = np.array([np.log(sigma[k]), mu[k]])
            L = np.linalg.cholesky(prop_cov[k])
            log_prop = log_curr + L @ rng.standard_normal(2)

            def lp(par, y_k=y_k):
                s = np.exp(par[0])
                m = par[1]
                if s < 1e-3 or s > 5.0:
                    return -1e15
                # Half-Cauchy(0, 2.5) on log sigma; Normal(0, 10) on mu
                return (-np.log(1.0 + (par[0] / 2.5) ** 2)
                        - 0.5 * (m / 10.0) ** 2
                        + par[0]
                        - len(y_k) * np.log(s)
                        - 0.5 * np.sum((y_k - m) ** 2) / s ** 2)

            ltc = lp(log_curr)
            ltp = lp(log_prop)
            n_prop[k] += 1
            if np.log(rng.uniform()) < ltp - ltc:
                sigma[k] = np.exp(log_prop[0])
                mu[k] = log_prop[1]
                n_acc[k] += 1
                hist[k].append(log_prop)
            else:
                hist[k].append(log_curr)

            if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist[k]) >= 200:
                h = np.array(hist[k][-500:])
                prop_cov[k] = (2.38 ** 2 / 2.0) * np.cov(h.T) + 1e-5 * np.eye(2)

        if it >= burn:
            samples['mu'].append(mu.copy())
            samples['sigma'].append(sigma.copy())
            samples['pi'].append(pi.copy())
            samples['Gamma'].append(Gamma.copy())
            samples['S'].append(S.copy())

        if verbose and (it + 1) % 3000 == 0:
            acc = n_acc / np.maximum(n_prop, 1)
            print(f"      iter {it+1}/{n_iter} | {time.time()-t0:.0f}s | "
                  f"acc=[" + " ".join(f"{x:.2f}" for x in acc) + "]")

    for k in samples:
        samples[k] = np.array(samples[k])
    return samples


# ============================================================
# MODEL 2: HMM-GAMMA (FIXED: uses math.lgamma)
# ============================================================
@njit(fastmath=True)
def gamma_logpdf(y, shape, rate):
    """Gamma(shape, rate) density at y>0. Uses math.lgamma."""
    return (shape * np.log(rate)
            - lgamma(shape)
            + (shape - 1.0) * np.log(y)
            - rate * y)


def run_mcmc_gamma(Y, K, n_iter, burn, seed, verbose=False):
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    T = len(Y)
    t0 = time.time()

    # Initialization: shape=2, rate chosen so that mean = shape/rate
    # matches data percentile
    shape = np.full(K, 2.0)
    rate = shape / np.percentile(Y, np.linspace(40, 80, K))

    pi = np.ones(K) / K
    Gamma = np.full((K, K), 1.0 / K) + 0.1 * np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.ones(K) * 2.0

    prop_cov = [np.eye(2) * 0.05 ** 2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)
    samples = {k: [] for k in ['shape', 'rate', 'pi', 'Gamma', 'S']}

    for it in range(n_iter):
        # ===== 1. Emission matrix =====
        log_em = np.empty((T, K))
        for t in range(T):
            for k in range(K):
                log_em[t, k] = gamma_logpdf(Y[t], shape[k], rate[k])

        # ===== 2. FFBS =====
        S = ffbs(log_em, pi, Gamma, rng)

        # ===== 3. Conjugate pi, Gamma =====
        counts = np.bincount(S[:1], minlength=K).astype(float)
        pi = rng.dirichlet(pi_prior + counts)
        for i in range(K):
            n_ij = np.zeros(K)
            mask = (S[:-1] == i)
            for j in range(K):
                n_ij[j] = np.sum(mask & (S[1:] == j))
            Gamma[i] = rng.dirichlet(np.ones(K) + n_ij)

        # ===== 4. Metropolis for (log shape, log rate) =====
        for k in range(K):
            y_k = Y[S == k]
            if len(y_k) < 2:
                continue
            log_curr = np.array([np.log(shape[k]), np.log(rate[k])])
            L = np.linalg.cholesky(prop_cov[k])
            log_prop = log_curr + L @ rng.standard_normal(2)

            def lp(par, y_k=y_k):
                sh = np.exp(par[0])
                rt = np.exp(par[1])
                if sh < 1e-3 or sh > 50.0 or rt < 1e-6 or rt > 1.0:
                    return -1e15
                # Half-Cauchy(0, 2.5) on log shape and log rate
                return (-np.log(1.0 + (par[0] / 2.5) ** 2)
                        - np.log(1.0 + (par[1] / 2.5) ** 2)
                        + par[0] + par[1]
                        + len(y_k) * (sh * np.log(rt) - lgamma(sh))
                        + (sh - 1.0) * np.sum(np.log(y_k))
                        - rt * np.sum(y_k))

            ltc = lp(log_curr)
            ltp = lp(log_prop)
            n_prop[k] += 1
            if np.log(rng.uniform()) < ltp - ltc:
                shape[k] = np.exp(log_prop[0])
                rate[k] = np.exp(log_prop[1])
                n_acc[k] += 1
                hist[k].append(log_prop)
            else:
                hist[k].append(log_curr)

            if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist[k]) >= 200:
                h = np.array(hist[k][-500:])
                prop_cov[k] = (2.38 ** 2 / 2.0) * np.cov(h.T) + 1e-5 * np.eye(2)

        if it >= burn:
            samples['shape'].append(shape.copy())
            samples['rate'].append(rate.copy())
            samples['pi'].append(pi.copy())
            samples['Gamma'].append(Gamma.copy())
            samples['S'].append(S.copy())

        if verbose and (it + 1) % 3000 == 0:
            acc = n_acc / np.maximum(n_prop, 1)
            print(f"      iter {it+1}/{n_iter} | {time.time()-t0:.0f}s | "
                  f"acc=[" + " ".join(f"{x:.2f}" for x in acc) + "]")

    for k in samples:
        samples[k] = np.array(samples[k])
    return samples


# ============================================================
# WAIC / DIC / LOO (generic given per-model emission)
# ============================================================
def compute_criteria_generic(samples, Y, K, emission_fn, n_sub=300):
    """Generic WAIC/DIC/LOO given a log-density function."""
    T = len(Y)
    n_post = list(samples.values())[0].shape[0]
    idx = np.random.choice(n_post, size=min(n_sub, n_post), replace=False)
    log_lik = np.zeros((len(idx), T))

    for i, ix in enumerate(idx):
        log_em = emission_fn(samples, ix, Y, K)
        pi_k = samples['pi'][ix]
        G_k = samples['Gamma'][ix]
        log_pi = np.log(pi_k)
        log_G = np.log(G_k)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_em[0]
        for t in range(1, T):
            m = log_alpha[t - 1][:, None] + log_G
            log_alpha[t] = logsumexp(m, axis=0) + log_em[t]
        log_p_y = np.zeros(T)
        log_p_y[0] = logsumexp(log_alpha[0])
        for t in range(1, T):
            log_p_y[t] = logsumexp(log_alpha[t]) - logsumexp(log_alpha[t - 1])
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
        log_p_loo = np.log(S_n) - logsumexp(-log_lik, axis=0)
        loo = -2.0 * log_p_loo.sum()
    except Exception:
        loo = np.nan
    return waic, dic, loo


def emission_lognormal(samples, ix, Y, K):
    """Emission log-density matrix for HMM-LogNormal."""
    mu_k = samples['mu'][ix]
    sg_k = samples['sigma'][ix]
    log_Y = np.log(Y)
    log_em = np.empty((len(Y), K))
    for k in range(K):
        z = (log_Y - mu_k[k]) / sg_k[k]
        log_em[:, k] = -log_Y - np.log(sg_k[k]) - 0.9189385332046727 - 0.5 * z * z
    return log_em


def emission_gamma(samples, ix, Y, K):
    """Emission log-density matrix for HMM-Gamma."""
    sh_k = samples['shape'][ix]
    rt_k = samples['rate'][ix]
    log_Y = np.log(Y)
    log_em = np.empty((len(Y), K))
    for k in range(K):
        log_em[:, k] = (sh_k[k] * np.log(rt_k[k])
                        - lgamma(sh_k[k])
                        + (sh_k[k] - 1.0) * log_Y
                        - rt_k[k] * Y)
    return log_em


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("COMPETING MODELS: HMM-LogNormal and HMM-Gamma")
    print("=" * 70)

    # ---- Load data ----
    data = np.load(NPZ_PATH, allow_pickle=True)
    Y = data['Y'].astype(float)
    K = 2
    print(f"\n  Data: T = {len(Y)} months, K = {K}")
    print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}] mm")
    print(f"  Y mean: {Y.mean():.2f} mm\n")

    # ============================================================
    # HMM-LogNormal
    # ============================================================
    print("-" * 70)
    print("  [1/2] HMM-LogNormal (30,000 iters, burn 15,000)")
    print("-" * 70)
    s_ln = run_mcmc_lognormal(Y, K, n_iter=30000, burn=15000,
                               seed=42, verbose=True)
    waic_ln, dic_ln, loo_ln = compute_criteria_generic(
        s_ln, Y, K, emission_lognormal)
    print(f"\n  HMM-LogNormal: WAIC={waic_ln:.2f}  DIC={dic_ln:.2f}  "
          f"LOO={loo_ln:.2f}")
    print(f"    mu    = {s_ln['mu'].mean(axis=0).round(4)}")
    print(f"    sigma = {s_ln['sigma'].mean(axis=0).round(4)}")

    # ============================================================
    # HMM-Gamma
    # ============================================================
    print("\n" + "-" * 70)
    print("  [2/2] HMM-Gamma (30,000 iters, burn 15,000)")
    print("-" * 70)
    s_gm = run_mcmc_gamma(Y, K, n_iter=30000, burn=15000,
                           seed=42, verbose=True)
    waic_gm, dic_gm, loo_gm = compute_criteria_generic(
        s_gm, Y, K, emission_gamma)
    print(f"\n  HMM-Gamma: WAIC={waic_gm:.2f}  DIC={dic_gm:.2f}  "
          f"LOO={loo_gm:.2f}")
    print(f"    shape = {s_gm['shape'].mean(axis=0).round(4)}")
    print(f"    rate  = {s_gm['rate'].mean(axis=0).round(4)}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    df = pd.DataFrame([
        {'Model': 'HMM-LogNormal', 'WAIC': round(waic_ln, 2),
         'DIC': round(dic_ln, 2), 'LOO': round(loo_ln, 2)},
        {'Model': 'HMM-Gamma', 'WAIC': round(waic_gm, 2),
         'DIC': round(dic_gm, 2), 'LOO': round(loo_gm, 2)},
    ])
    out = os.path.join(TABLES_DIR, "competing_models.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")

    # ============================================================
    # SUMMARY FOR TABLE 6
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR TABLE 6 (insert into LaTeX)")
    print("=" * 70)
    print(f"  HMM-Lognormal & 14 & {waic_ln:.2f} & {dic_ln:.2f} & "
          f"{loo_ln:.2f} & 3 \\\\")
    print(f"  HMM-Gamma     & 14 & {waic_gm:.2f} & {dic_gm:.2f} & "
          f"{loo_gm:.2f} & 4 \\\\")
    print("=" * 70)
    print("DONE")
    print("=" * 70)