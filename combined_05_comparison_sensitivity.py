# ============================================================
# combined_05_comparison_sensitivity.py — FINAL VERSION
# Section 1: Model comparison (6 models, Bayesian criteria only)
# Section 2: Prior sensitivity (CORRECTED - 6 independent prior params)
# ============================================================
import os, sys, time
sys.path.insert(0, r"D:\فكرة بحث 2027")
from output_helper import (get_folder, save_figure, save_table,
                            save_npz, save_summary)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from numba import njit
from hmm_core_extended import (simulate_hmm, run_chain, compute_waic_dic,
                                pointwise_ll_numpy, emission_matrix,
                                forward_filter, ffbs)

FOLDER = "05_comparison_sensitivity"
get_folder(FOLDER)
print("=" * 78)
print(f"[{FOLDER}] MODEL COMPARISON + PRIOR SENSITIVITY")
print("=" * 78)

# Simulated data
T = 800
K = 3
Y, S_true, truth = simulate_hmm(T=T, K=K, model_type=0, seed=1)
print(f"\nData: T={T}, K={K} (simulated from HMM-SM)")


# ============================================================
# HELPER: Bayesian criteria (WAIC + DIC + LOO)
# ============================================================
def compute_all_bayesian_criteria(log_lik_matrix):
    """Compute WAIC, DIC, LOO from a precomputed (S, T) log-lik matrix."""
    log_lik = np.clip(log_lik_matrix, -1e10, 100.0)
    S, T = log_lik.shape

    lppd = (logsumexp(log_lik, axis=0) - np.log(S)).sum()
    p_waic = log_lik.var(axis=0).sum()
    waic = -2 * (lppd - p_waic)

    D_bar = -2 * log_lik.mean(axis=0).sum()
    p_D = D_bar - (-2 * lppd)
    dic = D_bar + p_D

    log_p_loo_t = np.log(S) - logsumexp(-log_lik, axis=0)
    loo = -2 * log_p_loo_t.sum()

    if waic < 0 or not np.isfinite(waic):
        waic = dic = loo = np.nan

    return waic, dic, loo, lppd, p_waic


# ============================================================
# HELPER: GMM
# ============================================================
def gmm_logpdf(Y, w, mu, sigma):
    Y_ = Y[:, None]
    log_c = (-0.5 * np.log(2 * np.pi) - np.log(sigma)[None, :]
             - 0.5 * ((Y_ - mu[None, :]) / sigma[None, :]) ** 2)
    return logsumexp(np.log(w)[None, :] + log_c, axis=1)


def run_gmm_chain(Y, K=3, n_iter=5000, burn=2000, seed=0):
    rng = np.random.default_rng(seed)
    w = np.ones(K) / K
    mu = np.percentile(Y, np.linspace(20, 80, K))
    sigma = np.full(K, Y.std() / 2)
    log_w = np.log(w)
    log_sigma = np.log(sigma)
    n_dim = (K - 1) + K + K
    prop_cov = np.eye(n_dim) * 0.3 ** 2
    hist = []
    samples = {'w': [], 'mu': [], 'sigma': []}
    y_mean, y_std = Y.mean(), Y.std()

    for it in range(n_iter):
        theta = np.concatenate([log_w[:-1], mu, log_sigma])
        L = np.linalg.cholesky(prop_cov)
        theta_p = theta + L @ rng.standard_normal(n_dim)

        def lt(th):
            lw = np.concatenate([th[:K-1], [0.0]])
            m = th[K-1:2*K-1]
            ls = th[2*K-1:]
            ww = np.exp(lw - logsumexp(lw))
            ss = np.exp(ls)
            lp = -0.5 * np.sum(((m - y_mean) / (2 * y_std)) ** 2)
            lp += -0.5 * np.sum((ss / y_std) ** 2) + np.sum(ls)
            return lp + gmm_logpdf(Y, ww, m, ss).sum()

        if np.log(rng.uniform()) < lt(theta_p) - lt(theta):
            theta = theta_p
            log_w = np.concatenate([theta[:K-1], [0.0]])
            mu = theta[K-1:2*K-1]
            log_sigma = theta[2*K-1:]
        hist.append(theta.copy())

        if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist) >= 200:
            h = np.array(hist[-500:])
            prop_cov = (2.38 ** 2 / n_dim) * np.cov(h.T) + 1e-6 * np.eye(n_dim)

        if it >= burn:
            samples['w'].append(np.exp(log_w - logsumexp(log_w)))
            samples['mu'].append(mu.copy())
            samples['sigma'].append(np.exp(log_sigma))

    for k in samples:
        samples[k] = np.array(samples[k])
    return samples


def gmm_pointwise_ll(samples, Y, n_sub=500):
    n = samples['mu'].shape[0]
    idx = np.random.choice(n, size=min(n_sub, n), replace=False)
    T = len(Y)
    log_lik = np.zeros((len(idx), T))
    for i, ix in enumerate(idx):
        log_lik[i] = gmm_logpdf(Y, samples['w'][ix],
                                samples['mu'][ix], samples['sigma'][ix])
    return log_lik


# ============================================================
# HELPER: GEV
# ============================================================
def gev_logpdf(Y, mu, sigma, xi):
    if sigma <= 0:
        return np.full_like(Y, -np.inf)
    if abs(xi) < 1e-4:
        z = (Y - mu) / sigma
        return -np.log(sigma) - z - np.exp(-z)
    t = 1.0 + xi * (Y - mu) / sigma
    if np.any(t <= 0):
        return np.full_like(Y, -np.inf)
    return -np.log(sigma) - (1 + 1/xi) * np.log(t) - t ** (-1/xi)


def run_gev_chain(Y, n_iter=5000, burn=2000, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.array([Y.mean(), np.log(Y.std()), 0.1])
    prop_cov = np.eye(3) * 0.3 ** 2
    hist = []
    samples = {'mu': [], 'sigma': [], 'xi': []}
    y_mean, y_std = Y.mean(), Y.std()

    for it in range(n_iter):
        L = np.linalg.cholesky(prop_cov)
        theta_p = theta + L @ rng.standard_normal(3)

        def lt(th):
            m, ls, x = th
            s = np.exp(ls)
            lp = (-0.5 * ((m - y_mean) / (2 * y_std)) ** 2
                  - 0.5 * (s / y_std) ** 2 + ls
                  - 0.5 * ((x - 0.1) / 0.3) ** 2)
            ll = gev_logpdf(Y, m, s, x)
            if not np.all(np.isfinite(ll)):
                return -np.inf
            return lp + ll.sum()

        if np.log(rng.uniform()) < lt(theta_p) - lt(theta):
            theta = theta_p
        hist.append(theta.copy())

        if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist) >= 200:
            h = np.array(hist[-500:])
            prop_cov = (2.38 ** 2 / 3) * np.cov(h.T) + 1e-6 * np.eye(3)

        if it >= burn:
            samples['mu'].append(theta[0])
            samples['sigma'].append(np.exp(theta[1]))
            samples['xi'].append(theta[2])

    for k in samples:
        samples[k] = np.array(samples[k])
    return samples


def gev_pointwise_ll(samples, Y, n_sub=500):
    n = len(samples['mu'])
    idx = np.random.choice(n, size=min(n_sub, n), replace=False)
    T = len(Y)
    log_lik = np.zeros((len(idx), T))
    valid = np.ones(len(idx), dtype=bool)
    for i, ix in enumerate(idx):
        ll = gev_logpdf(Y, samples['mu'][ix], samples['sigma'][ix],
                        samples['xi'][ix])
        if not np.all(np.isfinite(ll)):
            valid[i] = False
        log_lik[i] = ll
    log_lik = log_lik[valid]
    if log_lik.shape[0] < 10:
        return None
    return log_lik


# ============================================================
# SECTION 1: MODEL COMPARISON (6 models)
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 1] Model Comparison (6 models, Bayesian criteria)")
print("-" * 78)
print("  Models: HMM-SM, HMM-Weibull, HMM-GPD, HMM-Gaussian, GMM, GEV")

results = {}

hmm_configs = [
    ("HMM-SM", 0, 3),
    ("HMM-Weibull", 1, 2),
    ("HMM-GPD", 4, 2),
    ("HMM-Gaussian", 5, 2),
]

for name, mt, params_per_state in hmm_configs:
    print(f"\n[{name}]")
    t0 = time.time()
    s = run_chain(Y, K=K, model_type=mt, n_iter=5000, burn=2000,
                  seed=0, chain_id=0)

    n_post = s['a'].shape[0]
    idx = np.random.choice(n_post, size=min(500, n_post), replace=False)
    log_lik = np.zeros((len(idx), T))
    for i, ix in enumerate(idx):
        log_lik[i] = pointwise_ll_numpy(
            Y, s['pi'][ix], s['Gamma'][ix],
            s['a'][ix], s['b'][ix], s['q'][ix], model_type=mt)

    waic, dic, loo, lppd, p_waic = compute_all_bayesian_criteria(log_lik)
    n_params = K * (K - 1) + (K - 1) + params_per_state * K
    results[name] = {
        'waic': waic, 'dic': dic, 'loo': loo,
        'lppd': lppd, 'p_waic': p_waic,
        'n_params': n_params, 'time': time.time() - t0,
    }
    print(f"  WAIC={waic:.2f}  DIC={dic:.2f}  LOO={loo:.2f}  "
          f"time={results[name]['time']:.1f}s")

# ---- GMM ----
print(f"\n[GMM]")
t0 = time.time()
s_gmm = run_gmm_chain(Y, K=3, n_iter=5000, burn=2000, seed=0)
log_lik_gmm = gmm_pointwise_ll(s_gmm, Y, n_sub=500)
waic, dic, loo, lppd, p_waic = compute_all_bayesian_criteria(log_lik_gmm)
n_params = 3 * (1 + 1 + 1) - 1
results['GMM'] = {
    'waic': waic, 'dic': dic, 'loo': loo,
    'lppd': lppd, 'p_waic': p_waic,
    'n_params': n_params, 'time': time.time() - t0,
}
print(f"  WAIC={waic:.2f}  DIC={dic:.2f}  LOO={loo:.2f}  "
      f"time={results['GMM']['time']:.1f}s")

# ---- GEV ----
print(f"\n[GEV]")
t0 = time.time()
s_gev = run_gev_chain(Y, n_iter=5000, burn=2000, seed=0)
log_lik_gev = gev_pointwise_ll(s_gev, Y, n_sub=500)
if log_lik_gev is not None:
    waic, dic, loo, lppd, p_waic = compute_all_bayesian_criteria(log_lik_gev)
else:
    waic = dic = loo = lppd = p_waic = np.nan
n_params = 3
results['GEV'] = {
    'waic': waic, 'dic': dic, 'loo': loo,
    'lppd': lppd, 'p_waic': p_waic,
    'n_params': n_params, 'time': time.time() - t0,
}
print(f"  WAIC={waic:.2f}  DIC={dic:.2f}  LOO={loo:.2f}  "
      f"time={results['GEV']['time']:.1f}s")

# ---- Summary ----
print("\n" + "=" * 90)
print("SUMMARY: Model Comparison")
print("=" * 90)

rows = []
for m, r in results.items():
    rows.append({
        "Model": m,
        "# params": r['n_params'],
        "WAIC": round(r['waic'], 2) if np.isfinite(r['waic']) else "N/A",
        "DIC": round(r['dic'], 2) if np.isfinite(r['dic']) else "N/A",
        "LOO": round(r['loo'], 2) if np.isfinite(r['loo']) else "N/A",
        "p_waic": round(r['p_waic'], 2) if np.isfinite(r['p_waic']) else "N/A",
        "Time (s)": round(r['time'], 1),
    })

df_cmp = pd.DataFrame(rows)
waic_vals = {m: r['waic'] for m, r in results.items() if np.isfinite(r['waic'])}
sorted_m = sorted(waic_vals, key=waic_vals.get)
df_cmp["Rank"] = [sorted_m.index(m) + 1 if m in sorted_m else "-"
                  for m in df_cmp["Model"]]
save_table(df_cmp, FOLDER, "S1_model_comparison", also_tex=True)
print(df_cmp.to_string(index=False))

# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_list = list(results.keys())
colors = plt.cm.Set2(np.linspace(0, 1, len(models_list)))

for ax, key, title in zip(axes, ['waic', 'dic', 'loo'],
                           ['WAIC', 'DIC', 'LOO']):
    vals = [results[m][key] if np.isfinite(results[m][key]) else np.nan
            for m in models_list]
    plot_vals = [v if np.isfinite(v) else 0 for v in vals]
    bars = ax.barh(models_list, plot_vals, color=colors)
    ax.set_title(f'{title} (lower = better)', fontsize=12)
    ax.set_xlabel(title)
    valid = [i for i, v in enumerate(vals) if np.isfinite(v)]
    if valid:
        best = min(valid, key=lambda i: vals[i])
        bars[best].set_edgecolor('red')
        bars[best].set_linewidth(3)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            ax.text(1, i, "N/A", va='center', fontsize=10, color='gray')
    ax.grid(alpha=0.3, axis='x')

plt.suptitle("Bayesian Model Comparison (6 models)", fontsize=14)
plt.tight_layout()
save_figure(fig, FOLDER, "S1_model_comparison.png")
plt.close(fig)


# ============================================================
# SECTION 2: PRIOR SENSITIVITY (CORRECTED)
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 2] Prior Sensitivity Analysis (CORRECTED)")
print("-" * 78)
print("  Re-running MCMC with 3 different prior scenarios...")


# ---- Numba helper: SM log-density ----
@njit(fastmath=True)
def _sm_logpdf_nb(y, a_v, b_v, q_v):
    log_yb = np.log(y) - np.log(b_v)
    a_log_yb = a_v * log_yb
    if a_log_yb > 30.0:
        l1p = a_log_yb
    elif a_log_yb < -30.0:
        l1p = 0.0
    else:
        l1p = np.log1p(np.exp(a_log_yb))
    return (np.log(a_v) + np.log(q_v) - np.log(b_v)
            + (a_v - 1.0) * log_yb - (q_v + 1.0) * l1p)


# ---- Numba helper: log target with 6 prior parameters ----
@njit(fastmath=True)
def _log_target_custom(log_abc, y_state,
                        aa, ba,       # Gamma for a
                        ab, bb,       # InvGamma for b
                        aq, bq):      # Gamma for q
    """
    Log target with 6 SEPARATE prior parameters.
    a ~ Gamma(aa, ba)
    b ~ InvGamma(ab, bb)
    q ~ Gamma(aq, bq)
    """
    a_v = np.exp(log_abc[0])
    b_v = np.exp(log_abc[1])
    q_v = np.exp(log_abc[2])

    log_prior = ((aa - 1.0) * np.log(a_v) - ba * a_v
                 - (ab + 1.0) * np.log(b_v) - bb / b_v
                 + (aq - 1.0) * np.log(q_v) - bq * q_v)

    log_lik = 0.0
    for i in range(len(y_state)):
        log_lik += _sm_logpdf_nb(y_state[i], a_v, b_v, q_v)

    return log_prior + log_lik + log_abc.sum()


def run_chain_with_priors(Y_in, K=3, n_iter=5000, burn=2000,
                          seed=0, priors=None):
    """
    Run HMM-SM MCMC with custom prior hyperparameters.
    priors = {
        'a': (alpha, beta),   # Gamma
        'b': (alpha, beta),   # InvGamma
        'q': (alpha, beta),   # Gamma
    }
    """
    if priors is None:
        priors = {'a': (2.0, 1.0), 'b': (2.0, 2.0), 'q': (2.0, 1.0)}

    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y_in, dtype=np.float64)
    T = len(Y)

    a = np.linspace(3.0, 1.5, K)
    b = np.percentile(Y, np.linspace(30, 90, K))
    q = np.linspace(4.0, 1.5, K)
    pi = np.ones(K) / K
    Gamma = np.full((K, K), 1.0 / K) + 0.05 * np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.ones(K) * 2.0

    aa, ba = priors['a']
    ab, bb = priors['b']
    aq, bq = priors['q']

    prop_cov = [np.eye(3) * 0.3 ** 2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)

    samples = {key: [] for key in ['a', 'b', 'q', 'pi', 'Gamma']}

    t0 = time.time()
    for it in range(n_iter):
        log_em = emission_matrix(Y, a, b, q, 0)
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

            # Correct 6-parameter call
            ltc = _log_target_custom(log_curr, y_k,
                                     aa, ba, ab, bb, aq, bq)
            ltp = _log_target_custom(log_prop, y_k,
                                     aa, ba, ab, bb, aq, bq)
            n_prop[k] += 1

            if np.log(rng.uniform()) < ltp - ltc:
                a[k] = np.exp(log_prop[0])
                b[k] = np.exp(log_prop[1])
                q[k] = np.exp(log_prop[2])
                n_acc[k] += 1
                hist[k].append(log_prop)
            else:
                hist[k].append(log_curr)

            if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist[k]) >= 200:
                h = np.array(hist[k][-500:])
                prop_cov[k] = (2.38 ** 2 / 3.0) * np.cov(h.T) + 1e-5 * np.eye(3)

        if it >= burn:
            samples['a'].append(a.copy())
            samples['b'].append(b.copy())
            samples['q'].append(q.copy())
            samples['pi'].append(pi.copy())
            samples['Gamma'].append(Gamma.copy())

        if (it + 1) % 1000 == 0:
            acc = n_acc / np.maximum(n_prop, 1)
            print(f"    iter {it+1}/{n_iter} | "
                  f"{time.time()-t0:.0f}s | "
                  f"acc=[{acc[0]:.2f} {acc[1]:.2f} {acc[2]:.2f}]")

    for key in samples:
        samples[key] = np.array(samples[key])
    return samples


# ---- Prior scenarios (CORRECTED - truly non-informative Weak) ----
prior_scenarios = {
    'Weak': {
        'a': (0.01, 0.01),   # ~ non-informative Gamma
        'b': (0.01, 0.01),   # ~ non-informative InvGamma
        'q': (0.01, 0.01),   # ~ non-informative Gamma
    },
    'Default': {
        'a': (2.0, 1.0),
        'b': (2.0, 2.0),
        'q': (2.0, 1.0),
    },
    'Informative': {
        'a': (10.0, 3.0),
        'b': (5.0, 4.0),
        'q': (8.0, 2.0),
    },
}

samples_by_prior = {}
for pname, priors in prior_scenarios.items():
    print(f"\n[Prior: {pname}]")
    print(f"  a ~ Gamma{priors['a']}")
    print(f"  b ~ InvGamma{priors['b']}")
    print(f"  q ~ Gamma{priors['q']}")
    t0 = time.time()
    s = run_chain_with_priors(Y, K=K, n_iter=5000, burn=2000,
                               seed=0, priors=priors)
    samples_by_prior[pname] = s
    print(f"  ✅ Done in {time.time() - t0:.1f}s")

# ---- Summary ----
print("\n" + "=" * 78)
print("Prior Sensitivity: Posterior Means")
print("=" * 78)

rows = []
for name in ['a', 'b', 'q']:
    for k in range(K):
        row = {"Parameter": f"{name}[{k+1}]",
               "Truth": round(truth[name][k], 4)}
        for pname in ['Weak', 'Default', 'Informative']:
            row[pname] = round(
                samples_by_prior[pname][name][:, k].mean(), 4)
        means = [samples_by_prior[p][name][:, k].mean()
                 for p in ['Weak', 'Default', 'Informative']]
        row["Range"] = round(max(means) - min(means), 4)
        rows.append(row)

df_ps = pd.DataFrame(rows)
save_table(df_ps, FOLDER, "S2_prior_sensitivity", also_tex=True)
print(df_ps.to_string(index=False))

# ---- Robustness ----
print("\n" + "=" * 78)
print("Robustness summary (max change across priors)")
print("=" * 78)

robustness_rows = []
for name in ['a', 'b', 'q']:
    max_range = max(
        max(samples_by_prior[p][name][:, k].mean()
            for p in ['Weak', 'Default', 'Informative']) -
        min(samples_by_prior[p][name][:, k].mean()
            for p in ['Weak', 'Default', 'Informative'])
        for k in range(K)
    )
    rel = 100 * max_range / np.abs(truth[name]).mean()
    status = "✅ Robust" if rel < 10 else "⚠️ Sensitive"
    robustness_rows.append({
        "Parameter": name,
        "Max change": round(max_range, 4),
        "Relative (%)": round(rel, 2),
        "Status": status,
    })
    print(f"  {name}: {max_range:.4f} ({rel:.1f}% of truth)  {status}")

df_rob = pd.DataFrame(robustness_rows)
save_table(df_rob, FOLDER, "S2_robustness_summary", also_tex=True)

# ---- Figure: Prior Sensitivity ----
fig, axes = plt.subplots(3, 3, figsize=(14, 9))
colors_p = {'Weak': 'steelblue', 'Default': 'green', 'Informative': 'coral'}
for i, name in enumerate(['a', 'b', 'q']):
    for k in range(3):
        ax = axes[i, k]
        for p in ['Weak', 'Default', 'Informative']:
            x = samples_by_prior[p][name][:, k]
            ax.hist(x, bins=40, density=True, alpha=0.4,
                    color=colors_p[p], label=p)
        ax.axvline(truth[name][k], color='red', ls='--', lw=2,
                   label=f'Truth={truth[name][k]:.2f}')
        ax.set_title(f'{name}[{k+1}]')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
plt.suptitle("Prior Sensitivity: Weak vs Default vs Informative",
             fontsize=13)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_prior_sensitivity.png")
plt.close(fig)

# ---- Figure: Robustness barchart ----
fig, ax = plt.subplots(figsize=(9, 5))
params = df_rob["Parameter"].tolist()
rel_vals = df_rob["Relative (%)"].tolist()
colors_r = ['green' if v < 10 else 'orange' for v in rel_vals]
bars = ax.bar(params, rel_vals, color=colors_r, alpha=0.85)
ax.axhline(10, color='red', ls='--', lw=2, alpha=0.6,
           label='10% threshold')
ax.set_ylabel('Relative change (%)', fontsize=12)
ax.set_title('Prior Sensitivity: Relative change across 3 priors',
             fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')
for bar, v in zip(bars, rel_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 0.5, f'{v:.1f}%', ha='center', fontsize=10,
            fontweight='bold')
plt.tight_layout()
save_figure(fig, FOLDER, "S2_robustness_barchart.png")
plt.close(fig)


# ============================================================
# SAVE SUMMARY
# ============================================================
summary = f"""MODEL COMPARISON + PRIOR SENSITIVITY
{"=" * 60}

Note: All criteria are fully Bayesian (WAIC, DIC, LOO).
AIC and BIC are excluded.

Note: HMM-Gamma and HMM-Lognormal excluded due to 
tail incompatibility with SM data. HMM-Gaussian included
as a standard baseline in the HMM literature.

[SECTION 1] Model Comparison (6 models)
{df_cmp.to_string(index=False)}

[SECTION 2] Prior Sensitivity
{df_ps.to_string(index=False)}

Robustness summary:
{df_rob.to_string(index=False)}
"""
save_summary(FOLDER, summary)

save_npz({
    "model_comparison": df_cmp.to_records(index=False),
    "prior_sensitivity": df_ps.to_records(index=False),
    "robustness": df_rob.to_records(index=False),
}, FOLDER, "comparison_sensitivity")

print("\n" + "=" * 78)
print("✅ DONE — combined_05_comparison_sensitivity.py")
print("=" * 78)