# ============================================================
# HMM-SM: T=5000 with 2 chains (10,000 iters each, 4,000 burn-in)
# Balanced: fast + accurate emissions + stable pi
# ============================================================
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

# ============================================================
# 1) NUMBA FUNCTIONS
# ============================================================
@njit(fastmath=True)
def sm_logpdf_numba(y, a, b, q):
    log_yb = np.log(y) - np.log(b)
    a_log_yb = a * log_yb
    if a_log_yb > 30.0:
        log_1p_ya = a_log_yb
    elif a_log_yb < -30.0:
        log_1p_ya = 0.0
    else:
        log_1p_ya = np.log1p(np.exp(a_log_yb))
    return (np.log(a) + np.log(q) - np.log(b)
            + (a - 1.0) * log_yb
            - (q + 1.0) * log_1p_ya)


@njit(fastmath=True)
def emission_log_matrix_numba(Y, a, b, q):
    T = Y.shape[0]
    K = a.shape[0]
    log_em = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            log_em[t, k] = sm_logpdf_numba(Y[t], a[k], b[k], q[k])
    return log_em


@njit(fastmath=True)
def forward_filter_numba(log_em, pi, Gamma):
    T, K = log_em.shape
    log_pi = np.log(pi)
    log_G = np.log(Gamma)
    log_alpha = np.empty((T, K))
    for k in range(K):
        log_alpha[0, k] = log_pi[k] + log_em[0, k]
    for t in range(1, T):
        for k in range(K):
            m_max = -np.inf
            for j in range(K):
                v = log_alpha[t-1, j] + log_G[j, k]
                if v > m_max:
                    m_max = v
            s = 0.0
            for j in range(K):
                s += np.exp(log_alpha[t-1, j] + log_G[j, k] - m_max)
            log_alpha[t, k] = m_max + np.log(s) + log_em[t, k]
    return log_alpha


@njit(fastmath=True)
def log_target_numba(log_abc, y_state):
    a = np.exp(log_abc[0])
    b = np.exp(log_abc[1])
    q = np.exp(log_abc[2])
    alpha_a, beta_a = 2.0, 1.0
    alpha_b, beta_b = 2.0, 2.0
    alpha_q, beta_q = 2.0, 1.0
    log_prior = ((alpha_a - 1.0) * np.log(a) - beta_a * a
                 - (alpha_b + 1.0) * np.log(b) - beta_b / b
                 + (alpha_q - 1.0) * np.log(q) - beta_q * q)
    log_lik = 0.0
    for i in range(len(y_state)):
        log_lik += sm_logpdf_numba(y_state[i], a, b, q)
    log_jac = log_abc[0] + log_abc[1] + log_abc[2]
    return log_prior + log_lik + log_jac


# ============================================================
# 2) SIMULATION
# ============================================================
def simulate_hmm_sm(T=5000, K=3, seed=42):
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
                      Gamma=Gamma_true, pi=pi_true, S=S)


# ============================================================
# 3) FFBS
# ============================================================
def ffbs_hybrid(log_em, pi, Gamma, rng):
    T, K = log_em.shape
    log_alpha = forward_filter_numba(log_em, pi, Gamma)
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
# 4) MCMC CHAIN
# ============================================================
def run_chain(Y_in, K=3, n_iter=10000, burn=4000, seed=0,
              chain_id=0, verbose=True):
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y_in, dtype=np.float64)
    T = len(Y)

    # Different init per chain
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

    Gamma = np.full((K, K), 1.0 / K) + 0.05 * np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)

    # Informative prior on pi
    pi_prior = np.array([4.0, 2.0, 1.0])

    prop_cov = [np.eye(3) * 0.3**2 for _ in range(K)]
    hist_log_abc = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)

    samples = {key: [] for key in ['a', 'b', 'q', 'pi', 'Gamma', 'S']}

    t0 = time.time()
    prog_interval = max(1000, n_iter // 10)

    for it in range(n_iter):
        print(it)

        log_em = emission_log_matrix_numba(Y, a, b, q)
        S = ffbs_hybrid(log_em, pi, Gamma, rng)

        # pi with prior
        counts_pi = np.bincount(S[:1], minlength=K).astype(float)
        pi = rng.dirichlet(pi_prior + counts_pi)

        # Gamma rows
        for i in range(K):
            n_ij = np.zeros(K)
            mask = (S[:-1] == i)
            for j in range(K):
                n_ij[j] = np.sum(mask & (S[1:] == j))
            Gamma[i] = rng.dirichlet(np.ones(K) + n_ij)

        # Emissions
        for k in range(K):
            y_k = Y[S == k]
            if len(y_k) < 2:
                continue
            y_k = np.ascontiguousarray(y_k, dtype=np.float64)
            log_abc_curr = np.array([np.log(a[k]), np.log(b[k]), np.log(q[k])])
            L = np.linalg.cholesky(prop_cov[k])
            z = rng.standard_normal(3)
            log_abc_prop = log_abc_curr + L @ z

            log_t_curr = log_target_numba(log_abc_curr, y_k)
            log_t_prop = log_target_numba(log_abc_prop, y_k)
            n_prop[k] += 1

            if np.log(rng.uniform()) < log_t_prop - log_t_curr:
                a[k] = np.exp(log_abc_prop[0])
                b[k] = np.exp(log_abc_prop[1])
                q[k] = np.exp(log_abc_prop[2])
                n_acc[k] += 1
                hist_log_abc[k].append(log_abc_prop)
            else:
                hist_log_abc[k].append(log_abc_curr)

            if it < 0.6 * burn and (it + 1) % 100 == 0 and len(hist_log_abc[k]) >= 200:
                hist = np.array(hist_log_abc[k][-500:])
                emp_cov = np.cov(hist.T)
                scale = 2.38**2 / 3.0
                prop_cov[k] = scale * emp_cov + 1e-5 * np.eye(3)

        if it >= burn:
            samples['a'].append(a.copy())
            samples['b'].append(b.copy())
            samples['q'].append(q.copy())
            samples['pi'].append(pi.copy())
            samples['Gamma'].append(Gamma.copy())
            samples['S'].append(S.copy())

        if verbose and (it + 1) % prog_interval == 0:
            elapsed = time.time() - t0
            pct = 100.0 * (it + 1) / n_iter
            acc = n_acc / np.maximum(n_prop, 1)
            speed = (it + 1) / elapsed
            eta = (n_iter - it - 1) / speed if speed > 0 else 0
            print(f"    Ch{chain_id} | {it+1:5d}/{n_iter} "
                  f"({pct:5.1f}%) | {elapsed:6.0f}s, ETA {eta:5.0f}s | "
                  f"acc=[{acc[0]:.2f} {acc[1]:.2f} {acc[2]:.2f}]")

    for key in samples:
        samples[key] = np.array(samples[key])
    return samples


# ============================================================
# 5) DIAGNOSTICS
# ============================================================
def compute_ess(x):
    n = len(x)
    x = x - x.mean()
    if (x**2).mean() == 0:
        return n
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    cutoff = 1
    for i in range(1, n):
        if acf[i] < 0.05:
            cutoff = i
            break
    return n / (1 + 2*acf[1:cutoff].sum())


def split_rhat(chains_list, param_name, idx):
    x = np.array([c[param_name][:, idx] for c in chains_list])
    m, n = x.shape
    if m < 2:
        return np.nan
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    return np.sqrt(((n-1)/n * W + B/n) / W)


# ============================================================
# 6) MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 78)
    print("HMM-SM: T=5000 + 2 chains (10,000 iters each)")
    print("=" * 78)

    # Warm-up Numba
    print("\n[Step 1] Warming up Numba...")
    t_c = time.time()
    _y = np.array([1.0, 2.0, 3.0])
    _a = np.array([2.0, 2.0, 2.0])
    _b = np.array([1.0, 1.0, 1.0])
    _q = np.array([2.0, 2.0, 2.0])
    _pi = np.array([0.5, 0.3, 0.2])
    _G = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    _ = forward_filter_numba(emission_log_matrix_numba(_y, _a, _b, _q), _pi, _G)
    _ = log_target_numba(np.array([0.0, 0.0, 0.0]), _y)
    print(f"  Done in {time.time()-t_c:.1f}s")

    # Simulate
    print("\n[Step 2] Simulating data (T=5000)...")
    T, K = 5000, 3
    Y, S_true, truth = simulate_hmm_sm(T=T, K=K, seed=1)
    print(f"  T = {T}, K = {K}")
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  State counts: {np.bincount(S_true, minlength=K)}")
    print(f"  True a : {truth['a']}")
    print(f"  True b : {truth['b']}")
    print(f"  True q : {truth['q']}")

    # Run 2 chains
    print(f"\n[Step 3] Running 2 chains (10,000 iters, 4,000 burn-in)...")
    print(f"Expected time: ~15-20 minutes\n")

    t0_total = time.time()
    chains = []
    seeds = [1, 2]
    for i, seed in enumerate(seeds):
        print(f"\n--- Chain {i+1}/2 (seed={seed}) ---")
        t0 = time.time()
        s = run_chain(Y, K=K, n_iter=10000, burn=4000,
                      seed=seed, chain_id=i, verbose=True)
        elapsed = time.time() - t0
        print(f"  ✅ Chain {i+1} finished in {elapsed/60:.2f} min")
        chains.append(s)

    total_time = time.time() - t0_total
    print(f"\n✅ All 2 chains finished in {total_time/60:.1f} min")

    # Combine
    print("\n[Step 4] Combining chains...")
    combined = {key: np.concatenate([c[key] for c in chains], axis=0)
                for key in chains[0].keys()}
    print(f"  Total post-burn-in samples: {combined['a'].shape[0]}")

    # Sort by b
    b_means = combined['b'].mean(axis=0)
    order = np.argsort(b_means)
    print(f"  State reordering (by increasing b): {order}")

    for key in ['a', 'b', 'q']:
        combined[key] = combined[key][:, order]
    combined['pi'] = combined['pi'][:, order]
    combined['Gamma'] = combined['Gamma'][:, order][:, :, order]
    truth_sorted = {k: truth[k][order] for k in ['a', 'b', 'q']}
    truth_pi_sorted = truth['pi'][order]
    truth_G_sorted = truth['Gamma'][order][:, order]

    # Report emissions
    print("\n" + "=" * 78)
    print("Posterior summary — Emissions (T=5000, 2 chains combined)")
    print("=" * 78)
    print(f"{'param':6s} {'st':3s} {'mean':>9s} {'sd':>8s} "
          f"{'q2.5':>9s} {'q97.5':>9s} {'truth':>9s} "
          f"{'err%':>7s} {'ESS':>7s} {'Rhat':>6s}")
    print("-" * 84)
    for name in ['a', 'b', 'q']:
        for k in range(K):
            x = combined[name][:, k]
            err = 100.0*(x.mean()-truth_sorted[name][k])/truth_sorted[name][k]
            rhat = split_rhat(chains, name, order[k])
            print(f"{name:6s} {k+1:3d} {x.mean():9.4f} {x.std():8.4f} "
                  f"{np.percentile(x,2.5):9.4f} "
                  f"{np.percentile(x,97.5):9.4f} "
                  f"{truth_sorted[name][k]:9.4f} "
                  f"{err:+7.1f} {compute_ess(x):7.0f} {rhat:6.4f}")

    # Report pi
    print("\n" + "=" * 78)
    print("Posterior summary — pi (T=5000, 2 chains combined)")
    print("=" * 78)
    print(f"{'param':6s} {'st':3s} {'mean':>9s} {'sd':>8s} "
          f"{'truth':>9s} {'err%':>7s} {'ESS':>7s} {'Rhat':>6s}")
    print("-" * 65)
    for k in range(K):
        x = combined['pi'][:, k]
        err = 100.0*(x.mean()-truth_pi_sorted[k])/truth_pi_sorted[k]
        rhat = split_rhat(chains, 'pi', order[k])
        print(f"{'pi':6s} {k+1:3d} {x.mean():9.4f} {x.std():8.4f} "
              f"{truth_pi_sorted[k]:9.4f} {err:+7.1f} "
              f"{compute_ess(x):7.0f} {rhat:6.4f}")

    # Report Gamma
    print("\n" + "=" * 78)
    print("Posterior summary — Gamma (T=5000, 2 chains combined)")
    print("=" * 78)
    print(f"{'from':>4s} {'to':>4s} {'mean':>9s} {'sd':>8s} "
          f"{'truth':>9s} {'err%':>7s}")
    print("-" * 55)
    for i in range(K):
        for j in range(K):
            x = combined['Gamma'][:, i, j]
            err = 100.0*(x.mean()-truth_G_sorted[i,j])/truth_G_sorted[i,j]
            print(f"{i+1:>4d} {j+1:>4d} {x.mean():9.4f} {x.std():8.4f} "
                  f"{truth_G_sorted[i,j]:9.4f} {err:+7.1f}")

    # State recovery
    print("\n" + "=" * 78)
    print("Hidden state recovery")
    print("=" * 78)
    S_post = np.array([np.bincount(combined['S'][:, t], minlength=K).argmax()
                       for t in range(combined['S'].shape[1])])
    acc = (S_post == S_true).mean()
    print(f"  State accuracy: {acc*100:.2f}%")
    print(f"  True state counts: {np.bincount(S_true, minlength=K)}")
    print(f"  Post state counts: {np.bincount(S_post, minlength=K)}")

    # Save
    np.savez("hmm_sm_T5000_2chains.npz",
             Y=Y, S_true=S_true, S_post=S_post,
             a_samples=combined['a'], b_samples=combined['b'],
             q_samples=combined['q'], pi_samples=combined['pi'],
             Gamma_samples=combined['Gamma'],
             a_true=truth_sorted['a'], b_true=truth_sorted['b'],
             q_true=truth_sorted['q'],
             pi_true=truth_pi_sorted, Gamma_true=truth_G_sorted)
    print("\n✅ Saved: hmm_sm_T5000_2chains.npz")

    # Plots
    print("\n[Step 5] Generating plots...")
    colors = ['steelblue', 'coral']

    fig, axes = plt.subplots(4, 3, figsize=(15, 11))
    for i, name in enumerate(['a', 'b', 'q', 'pi']):
        for k in range(3):
            ax = axes[i, k]
            for c, chain in enumerate(chains):
                x = chain[name][:, order[k]]
                ax.plot(x, lw=0.3, alpha=0.6, color=colors[c],
                        label=f'chain {c+1}' if k == 0 else None)
            tv = truth_sorted[name][k] if name != 'pi' else truth_pi_sorted[k]
            ax.axhline(tv, color='red', ls='--', lw=1.5,
                       label=f'truth={tv:.2f}')
            ax.set_title(f"{name}[state {k+1}]")
            ax.legend(fontsize=6)
    plt.suptitle("Trace plots — T=5000, 2 chains")
    plt.tight_layout()
    plt.savefig("hmm_sm_T5000_2chains_traces.png", dpi=100)
    plt.show(block=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for i, name in enumerate(['a', 'b', 'q']):
        for k in range(3):
            ax = axes[i, k]
            x = combined[name][:, k]
            ax.hist(x, bins=50, density=True, alpha=0.7, color='steelblue')
            ax.axvline(truth_sorted[name][k], color='red', ls='--',
                       lw=2, label=f'truth={truth_sorted[name][k]:.2f}')
            ax.axvline(x.mean(), color='black', ls='-', lw=1,
                       label=f'mean={x.mean():.2f}')
            ax.set_title(f"{name}[state {k+1}]")
            ax.legend(fontsize=8)
    plt.suptitle("Posterior densities — T=5000, 2 chains")
    plt.tight_layout()
    plt.savefig("hmm_sm_T5000_2chains_posteriors.png", dpi=100)
    plt.show(block=False)

    print("  ✅ Saved: hmm_sm_T5000_2chains_traces.png")
    print("  ✅ Saved: hmm_sm_T5000_2chains_posteriors.png")

    print("\n" + "=" * 78)
    print("✅ DONE")
    print("=" * 78)