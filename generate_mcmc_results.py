# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:30:36 2026

@author: ROYAL2
"""

# ============================================================
# generate_mcmc_results.py — FINAL CLEAN VERSION
# Generates: D:\فكرة بحث 2027\hmm_sm_T5000_2chains.npz
#
# Two modes:
#   QUICK_TEST = True  → ~30 seconds (for testing)
#   QUICK_TEST = False → ~45 minutes (full analysis)
# ============================================================
import os
import sys
import time
import numpy as np
from numba import njit

# ============================================================
# CONFIGURATION — CHANGE ONLY THIS
# ============================================================
QUICK_TEST = False      # ⬅️ TRUE first time (30 sec), then FALSE for full run
OUTPUT_DIR = r"D:\فكرة بحث 2027"

# ============================================================
# NUMBA FUNCTIONS
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
    T = Y.shape[0]
    K = a.shape[0]
    log_em = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            log_em[t, k] = sm_logpdf(Y[t], a[k], b[k], q[k])
    return log_em


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


@njit(fastmath=True)
def log_target(log_abc, y_state):
    a = np.exp(log_abc[0])
    b = np.exp(log_abc[1])
    q = np.exp(log_abc[2])
    log_prior = ((2.0 - 1.0) * np.log(a) - 1.0 * a
                 - (2.0 + 1.0) * np.log(b) - 2.0 / b
                 + (2.0 - 1.0) * np.log(q) - 1.0 * q)
    log_lik = 0.0
    for i in range(len(y_state)):
        log_lik += sm_logpdf(y_state[i], a, b, q)
    return log_prior + log_lik + log_abc.sum()


# ============================================================
# FFBS
# ============================================================
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


# ============================================================
# SIMULATION
# ============================================================
def simulate_hmm_sm(T, K=3, seed=42):
    rng = np.random.default_rng(seed)
    a_true = np.array([3.0, 2.0, 1.5])
    b_true = np.array([2.0, 8.0, 20.0])
    q_true = np.array([4.0, 2.5, 1.5])
    Gamma_true = np.array([[0.90, 0.08, 0.02],
                           [0.10, 0.85, 0.05],
                           [0.05, 0.15, 0.80]])
    pi_true = np.array([0.6, 0.3, 0.1])

    S = np.zeros(T, dtype=np.int64)
    S[0] = rng.choice(K, p=pi_true)
    for t in range(1, T):
        S[t] = rng.choice(K, p=Gamma_true[S[t-1]])

    u = rng.uniform(size=T)
    Y = np.zeros(T)
    for k in range(K):
        idx = (S == k)
        if idx.sum() > 0:
            Y[idx] = b_true[k] * ((1-u[idx])**(-1.0/q_true[k]) - 1.0)**(1.0/a_true[k])

    return Y, S, dict(a=a_true, b=b_true, q=q_true,
                      Gamma=Gamma_true, pi=pi_true)


# ============================================================
# MCMC CHAIN
# ============================================================
def run_chain(Y_in, K=3, n_iter=10000, burn=4000, seed=0, chain_id=0):
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y_in, dtype=np.float64)
    T = len(Y)

    inits = [
        (np.array([3.0, 2.0, 1.5]),
         np.array([2.0, 8.0, 20.0]),
         np.array([4.0, 2.5, 1.5]),
         np.array([0.5, 0.3, 0.2])),
        (np.array([3.5, 1.7, 1.3]),
         np.array([2.5, 9.0, 22.0]),
         np.array([4.5, 2.2, 1.2]),
         np.array([0.6, 0.3, 0.1])),
    ]
    if chain_id < len(inits):
        a, b, q, pi = [x.copy() for x in inits[chain_id]]
    else:
        a = rng.uniform(1.0, 3.5, K)
        b = np.percentile(Y, [30, 60, 90])
        q = rng.uniform(1.0, 4.0, K)
        pi = np.ones(K) / K

    Gamma = np.full((K, K), 1.0/K) + 0.05*np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.array([4.0, 2.0, 1.0])

    prop_cov = [np.eye(3) * 0.3**2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)

    samples = {key: [] for key in ['a', 'b', 'q', 'pi', 'Gamma', 'S']}

    t0 = time.time()
    prog_interval = max(500, n_iter // 10)

    for it in range(n_iter):
        print("iter=",it)
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

        if (it + 1) % prog_interval == 0:
            elapsed = time.time() - t0
            pct = 100 * (it + 1) / n_iter
            acc = n_acc / np.maximum(n_prop, 1)
            speed = (it + 1) / elapsed
            eta = (n_iter - it - 1) / speed if speed > 0 else 0
            print(f"    Ch{chain_id} | {it+1:5d}/{n_iter} ({pct:5.1f}%) "
                  f"| {elapsed:6.0f}s, ETA {eta:5.0f}s "
                  f"| acc=[{acc[0]:.2f} {acc[1]:.2f} {acc[2]:.2f}]")

    for key in samples:
        samples[key] = np.array(samples[key])
    return samples


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 78)
    print("HMM-SM: Generate MCMC results")
    print("=" * 78)
    print(f"Mode: {'QUICK TEST (30 sec)' if QUICK_TEST else 'FULL RUN (~45 min)'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "hmm_sm_T5000_2chains.npz")
    print(f"Output: {output_path}")

    # ---- Config ----
    if QUICK_TEST:
        T = 500
        n_iter, burn = 500, 200
        seeds = [1]
    else:
        T = 5000
        n_iter, burn = 10000, 4000
        seeds = [1, 2]

    K = 3

    # ---- Warm-up ----
    print("\n[Step 1] Warming up Numba...")
    t_c = time.time()
    _y = np.array([1.0, 2.0, 3.0])
    _a = np.array([2.0, 2.0, 2.0])
    _b = np.array([1.0, 1.0, 1.0])
    _q = np.array([2.0, 2.0, 2.0])
    _pi = np.array([0.5, 0.3, 0.2])
    _G = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    _ = forward_filter(emission_matrix(_y, _a, _b, _q), _pi, _G)
    _ = log_target(np.array([0.0, 0.0, 0.0]), _y)
    print(f"  ✅ Numba warm-up done in {time.time()-t_c:.1f}s")

    # ---- Simulate ----
    print(f"\n[Step 2] Simulating data (T={T}, K={K})...")
    Y, S_true, truth = simulate_hmm_sm(T=T, K=K, seed=1)
    print(f"  ✅ Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  ✅ State counts: {np.bincount(S_true, minlength=K)}")

    # ---- Run ----
    print(f"\n[Step 3] Running {len(seeds)} chain(s)...")
    t_total = time.time()
    chains = []
    for i, seed in enumerate(seeds):
        print(f"\n  --- Chain {i+1}/{len(seeds)} (seed={seed}) ---")
        t0 = time.time()
        s = run_chain(Y, K=K, n_iter=n_iter, burn=burn, seed=seed, chain_id=i)
        print(f"  ✅ Done in {(time.time()-t0)/60:.2f} min")
        chains.append(s)

    elapsed = time.time() - t_total
    print(f"\n✅ All chains finished in {elapsed/60:.2f} min")

    # ---- Combine ----
    print("\n[Step 4] Combining chains...")
    combined = {key: np.concatenate([c[key] for c in chains], axis=0)
                for key in chains[0].keys()}
    print(f"  Total post-burn-in samples: {combined['a'].shape[0]}")

    # ---- Sort by b ----
    b_means = combined['b'].mean(axis=0)
    order = np.argsort(b_means)
    for key in ['a', 'b', 'q']:
        combined[key] = combined[key][:, order]
    combined['pi'] = combined['pi'][:, order]
    combined['Gamma'] = combined['Gamma'][:, order][:, :, order]
    truth_sorted = {k: truth[k][order] for k in ['a', 'b', 'q']}
    pi_true_sorted = truth['pi'][order]
    Gamma_true_sorted = truth['Gamma'][order][:, order]

    S_post = np.array([np.bincount(combined['S'][:, t], minlength=K).argmax()
                       for t in range(combined['S'].shape[1])])
    accuracy = (S_post == S_true).mean()
    print(f"  State accuracy: {accuracy*100:.2f}%")

    # ---- Save ----
    print("\n[Step 5] Saving results...")
    np.savez(
        output_path,
        Y=Y,
        S_true=S_true,
        S_post=S_post,
        a_samples=combined['a'],
        b_samples=combined['b'],
        q_samples=combined['q'],
        pi_samples=combined['pi'],
        Gamma_samples=combined['Gamma'],
        S_samples=combined['S'],
        a_true=truth_sorted['a'],
        b_true=truth_sorted['b'],
        q_true=truth_sorted['q'],
        pi_true=pi_true_sorted,
        Gamma_true=Gamma_true_sorted,
    )
    print(f"  ✅ Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path)/1024:.1f} KB")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'param':6s} {'state':5s} {'mean':>10s} {'truth':>10s} "
          f"{'error %':>10s}")
    print("-" * 50)
    for name in ['a', 'b', 'q']:
        for k in range(K):
            x = combined[name][:, k]
            t = truth_sorted[name][k]
            err = 100 * (x.mean() - t) / t
            print(f"{name:6s} {k+1:5d} {x.mean():10.4f} {t:10.4f} "
                  f"{err:+10.2f}")

    print("\n" + "=" * 78)
    print("✅ DONE")
    print(f"   Now run: combined_01_core_properties.py")
    print("=" * 78)