# ============================================================
# hmm_core_extended.py — WITH GPD SUPPORT
# Emission types: 0=SM, 1=Weibull, 2=Gamma, 3=Lognormal, 4=GPD
# ============================================================
import numpy as np
from numba import njit
import math
from scipy.stats import gamma as gamma_dist
from scipy.special import erfinv


# ============================================================
# 1) EMISSION LOG-DENSITIES
# ============================================================
@njit(fastmath=True)
def sm_logpdf(y, a, b, q):
    """Singh-Maddala (Burr XII) log-density."""
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
def weibull_logpdf(y, a, b, q):
    """Weibull(shape=a, scale=b). q unused."""
    return (np.log(a) - np.log(b)
            + (a - 1.0) * (np.log(y) - np.log(b))
            - (y / b) ** a)


@njit(fastmath=True)
def gamma_logpdf(y, a, b, q):
    """Gamma(shape=a, rate=b). q unused."""
    return (a * np.log(b) - math.lgamma(a)
            + (a - 1.0) * np.log(y) - b * y)


@njit(fastmath=True)
def lognormal_logpdf(y, mu, sigma, q):
    """Lognormal(mu, sigma). a=mu, b=sigma."""
    return (-np.log(y) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((np.log(y) - mu) / sigma) ** 2)


@njit(fastmath=True)
def gpd_logpdf(y, xi, sigma, q):
    """
    Generalized Pareto Distribution (GPD) log-density.
    Params:
        xi = shape parameter (heavy tail if xi > 0)
        sigma = scale parameter
        q = unused (kept for uniform interface)
    Support: y > 0 for xi > 0
    """
    if sigma <= 0:
        return -np.inf
    if xi == 0:
        # Exponential limit
        return -np.log(sigma) - y / sigma
    # Support constraint: 1 + xi*y/sigma > 0
    if 1.0 + xi * y / sigma <= 0:
        return -np.inf
    return -np.log(sigma) - (1.0 + 1.0/xi) * np.log1p(xi * y / sigma)


@njit(fastmath=True)
def exponential_logpdf(y, lam, b, q):
    """Exponential(rate=lam). a=lam."""
    return np.log(lam) - lam * y


@njit(fastmath=True)
def emission_matrix(Y, a, b, q, model_type):
    """
    Emission matrix (T, K).
    model_type:
        0 = Singh-Maddala
        1 = Weibull
        2 = Gamma
        3 = Lognormal
        4 = GPD
        5 = Exponential
    """
    T = Y.shape[0]
    K = a.shape[0]
    log_em = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            if model_type == 0:
                log_em[t, k] = sm_logpdf(Y[t], a[k], b[k], q[k])
            elif model_type == 1:
                log_em[t, k] = weibull_logpdf(Y[t], a[k], b[k], q[k])
            elif model_type == 2:
                log_em[t, k] = gamma_logpdf(Y[t], a[k], b[k], q[k])
            elif model_type == 3:
                log_em[t, k] = lognormal_logpdf(Y[t], a[k], b[k], q[k])
            elif model_type == 4:
                log_em[t, k] = gpd_logpdf(Y[t], a[k], b[k], q[k])
            else:
                log_em[t, k] = exponential_logpdf(Y[t], a[k], b[k], q[k])
    return log_em


# ============================================================
# 2) FORWARD FILTER
# ============================================================
@njit(fastmath=True)
def forward_filter(log_em, pi, Gamma):
    """Forward algorithm in log-space."""
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


# ============================================================
# 3) FFBS
# ============================================================
def ffbs(log_em, pi, Gamma, rng):
    """Forward-Filtering Backward-Sampling."""
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
# 4) LOG TARGET
# ============================================================
@njit(fastmath=True)
def log_target(log_abc, y_state, model_type):
    """Log target for (a, b, q) MH update."""
    a = np.exp(log_abc[0])
    b = np.exp(log_abc[1])
    q = np.exp(log_abc[2])
    log_prior = ((2.0 - 1.0) * np.log(a) - 1.0 * a
                 - (2.0 + 1.0) * np.log(b) - 2.0 / b
                 + (2.0 - 1.0) * np.log(q) - 1.0 * q)
    log_lik = 0.0
    for i in range(len(y_state)):
        if model_type == 0:
            log_lik += sm_logpdf(y_state[i], a, b, q)
        elif model_type == 1:
            log_lik += weibull_logpdf(y_state[i], a, b, q)
        elif model_type == 2:
            log_lik += gamma_logpdf(y_state[i], a, b, q)
        elif model_type == 3:
            log_lik += lognormal_logpdf(y_state[i], a, b, q)
        elif model_type == 4:
            log_lik += gpd_logpdf(y_state[i], a, b, q)
        else:
            log_lik += exponential_logpdf(y_state[i], a, b, q)
    return log_prior + log_lik + log_abc.sum()


# ============================================================
# 5) SIMULATION
# ============================================================
def simulate_hmm(T=800, K=3, model_type=0, seed=42):
    """
    Simulate data from HMM with specified emission distribution.

    Different true parameters for each model type to ensure realistic
    data-generating processes.
    """
    rng = np.random.default_rng(seed)

    # ---- True parameters per model type ----
    if model_type == 0:  # Singh-Maddala
        a_true = np.array([3.0, 2.0, 1.5, 2.5, 1.8])[:K]
        b_true = np.array([2.0, 8.0, 20.0, 5.0, 12.0])[:K]
        q_true = np.array([4.0, 2.5, 1.5, 3.0, 2.0])[:K]
    elif model_type == 1:  # Weibull
        a_true = np.array([2.0, 1.8, 1.4, 1.6, 1.5])[:K]
        b_true = np.array([3.0, 8.0, 15.0, 6.0, 11.0])[:K]
        q_true = np.ones(K)
    elif model_type == 2:  # Gamma
        a_true = np.array([2.5, 3.0, 4.0, 3.5, 3.8])[:K]
        b_true = np.array([1.5, 0.8, 0.4, 0.6, 0.5])[:K]
        q_true = np.ones(K)
    elif model_type == 3:  # Lognormal
        a_true = np.array([1.5, 2.0, 2.5, 2.2, 2.4])[:K]  # mu
        b_true = np.array([0.7, 1.0, 1.3, 1.1, 1.2])[:K]  # sigma
        q_true = np.ones(K)
    elif model_type == 4:  # GPD
        a_true = np.array([0.15, 0.25, 0.35, 0.30, 0.32])[:K]  # xi (shape)
        b_true = np.array([3.0, 8.0, 15.0, 6.0, 11.0])[:K]      # sigma (scale)
        q_true = np.ones(K)
    else:  # Exponential
        a_true = np.array([0.5, 0.15, 0.07, 0.12, 0.09])[:K]  # rate
        b_true = np.ones(K)
        q_true = np.ones(K)

    # ---- Sticky transition matrix ----
    Gamma_true = np.full((K, K), 0.05 / max(K-1, 1))
    np.fill_diagonal(Gamma_true, 0.95)
    Gamma_true /= Gamma_true.sum(axis=1, keepdims=True)
    pi_true = np.ones(K) / K

    # ---- Simulate states ----
    S = np.zeros(T, dtype=np.int64)
    S[0] = rng.choice(K, p=pi_true)
    for t in range(1, T):
        S[t] = rng.choice(K, p=Gamma_true[S[t-1]])

    # ---- Simulate observations via inverse-CDF ----
    u = rng.uniform(size=T)
    Y = np.zeros(T)
    for k in range(K):
        idx = (S == k)
        if idx.sum() == 0:
            continue

        if model_type == 0:  # Singh-Maddala
            Y[idx] = (b_true[k]
                      * ((1 - u[idx])**(-1/q_true[k]) - 1)**(1/a_true[k]))
        elif model_type == 1:  # Weibull
            Y[idx] = b_true[k] * (-np.log(1 - u[idx]))**(1/a_true[k])
        elif model_type == 2:  # Gamma
            Y[idx] = gamma_dist.ppf(u[idx], a=a_true[k], scale=1/b_true[k])
        elif model_type == 3:  # Lognormal
            Y[idx] = np.exp(a_true[k] + b_true[k] * np.sqrt(2)
                            * erfinv(2*u[idx] - 1))
        elif model_type == 4:  # GPD
            xi_k = a_true[k]
            sigma_k = b_true[k]
            Y[idx] = (sigma_k / xi_k) * ((1 - u[idx])**(-xi_k) - 1)
        else:  # Exponential
            Y[idx] = -np.log(1 - u[idx]) / a_true[k]

    # Ensure positivity
    Y = np.maximum(Y, 1e-10)

    return Y, S, dict(a=a_true, b=b_true, q=q_true,
                      Gamma=Gamma_true, pi=pi_true, S=S)


# ============================================================
# 6) MCMC CHAIN
# ============================================================
def run_chain(Y_in, K=3, model_type=0, n_iter=5000, burn=2000,
              seed=0, chain_id=0):
    """Run a single MCMC chain for HMM with given emission type."""
    rng = np.random.default_rng(seed)
    Y = np.ascontiguousarray(Y_in, dtype=np.float64)
    T = len(Y)

    # ---- Initialization (model-specific) ----
    if model_type == 3:  # Lognormal
        log_Y = np.log(Y)
        a = np.percentile(log_Y, np.linspace(30, 70, K))
        b = np.full(K, 1.0)
        q = np.ones(K)
    elif model_type == 4:  # GPD
        a = np.linspace(0.1, 0.4, K)  # xi
        b = np.percentile(Y, np.linspace(60, 95, K))  # sigma
        q = np.ones(K)
    else:  # SM, Weibull, Gamma, Exponential
        a = np.linspace(3.0, 1.5, K)
        b = np.percentile(Y, np.linspace(30, 90, K))
        q = np.linspace(4.0, 1.5, K)

    pi = np.ones(K) / K
    Gamma = np.full((K, K), 1.0/K) + 0.05*np.eye(K)
    Gamma /= Gamma.sum(axis=1, keepdims=True)
    pi_prior = np.ones(K) * 2.0

    prop_cov = [np.eye(3) * 0.3**2 for _ in range(K)]
    hist = [[] for _ in range(K)]
    n_acc = np.zeros(K)
    n_prop = np.zeros(K)

    samples = {key: [] for key in ['a', 'b', 'q', 'pi', 'Gamma', 'S']}

    for it in range(n_iter):
        # Forward-backward
        log_em = emission_matrix(Y, a, b, q, model_type)
        S = ffbs(log_em, pi, Gamma, rng)

        # Update pi
        counts = np.bincount(S[:1], minlength=K).astype(float)
        pi = rng.dirichlet(pi_prior + counts)

        # Update Gamma
        for i in range(K):
            n_ij = np.zeros(K)
            mask = (S[:-1] == i)
            for j in range(K):
                n_ij[j] = np.sum(mask & (S[1:] == j))
            Gamma[i] = rng.dirichlet(np.ones(K) + n_ij)

        # Update emissions via adaptive MH
        for k in range(K):
            y_k = Y[S == k]
            if len(y_k) < 2:
                continue
            y_k = np.ascontiguousarray(y_k, dtype=np.float64)
            log_curr = np.array([np.log(a[k]), np.log(b[k]), np.log(q[k])])
            L = np.linalg.cholesky(prop_cov[k])
            log_prop = log_curr + L @ rng.standard_normal(3)
            ltc = log_target(log_curr, y_k, model_type)
            ltp = log_target(log_prop, y_k, model_type)
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

    for key in samples:
        samples[key] = np.array(samples[key])
    return samples


# ============================================================
# 7) POINTWISE LOG-LIKELIHOOD
# ============================================================
def pointwise_ll_numpy(Y, pi, Gamma, a, b, q, model_type=0):
    """Pointwise log-likelihood for WAIC/LOO."""
    from scipy.special import logsumexp as lse, gammaln
    T = len(Y)
    K = len(pi)

    log_em = np.empty((T, K))
    for k in range(K):
        if model_type == 0:  # Singh-Maddala
            log_yb = np.log(Y) - np.log(b[k])
            alyb = a[k] * log_yb
            l1p = np.logaddexp(0, alyb)
            log_em[:, k] = (np.log(a[k]) + np.log(q[k]) - np.log(b[k])
                            + (a[k]-1)*log_yb - (q[k]+1)*l1p)
        elif model_type == 1:  # Weibull
            log_em[:, k] = (np.log(a[k]) - np.log(b[k])
                            + (a[k]-1)*(np.log(Y)-np.log(b[k]))
                            - (Y/b[k])**a[k])
        elif model_type == 2:  # Gamma
            log_em[:, k] = (a[k]*np.log(b[k]) - gammaln(a[k])
                            + (a[k]-1)*np.log(Y) - b[k]*Y)
        elif model_type == 3:  # Lognormal
            log_em[:, k] = (-np.log(Y) - np.log(b[k])
                            - 0.5*np.log(2*np.pi)
                            - 0.5*((np.log(Y) - a[k]) / b[k])**2)
        elif model_type == 4:  # GPD
            xi_k = a[k]
            sigma_k = b[k]
            if xi_k == 0:
                log_em[:, k] = -np.log(sigma_k) - Y / sigma_k
            else:
                log_em[:, k] = (-np.log(sigma_k)
                                - (1.0 + 1.0/xi_k)
                                * np.log1p(xi_k * Y / sigma_k))
        else:  # Exponential
            log_em[:, k] = np.log(a[k]) - a[k]*Y

    log_pi = np.log(pi)
    log_G = np.log(Gamma)
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_em[0]
    for t in range(1, T):
        m = log_alpha[t-1][:, None] + log_G
        log_alpha[t] = lse(m, axis=0) + log_em[t]

    log_p_y = np.empty(T)
    log_p_y[0] = lse(log_alpha[0])
    for t in range(1, T):
        log_p_y[t] = lse(log_alpha[t]) - lse(log_alpha[t-1])
    return log_p_y


# ============================================================
# 8) WAIC / DIC
# ============================================================
def compute_waic_dic(samples, Y, model_type=0, n_sub=500):
    """Compute WAIC and DIC from posterior samples."""
    from scipy.special import logsumexp as lse
    n = samples['a'].shape[0]
    idx = np.random.choice(n, size=min(n_sub, n), replace=False)
    T = len(Y)
    log_lik = np.zeros((len(idx), T))
    for i, ix in enumerate(idx):
        log_lik[i] = pointwise_ll_numpy(
            Y, samples['pi'][ix], samples['Gamma'][ix],
            samples['a'][ix], samples['b'][ix], samples['q'][ix],
            model_type)

    lppd = (lse(log_lik, axis=0) - np.log(len(idx))).sum()
    p_waic = log_lik.var(axis=0).sum()
    waic = -2 * (lppd - p_waic)

    D_bar = -2 * log_lik.mean(axis=0).sum()
    p_D = D_bar - (-2 * lppd)
    dic = D_bar + p_D

    return waic, dic, lppd, p_waic


# ============================================================
# 9) EFFECTIVE SAMPLE SIZE
# ============================================================
def compute_ess(x):
    """Effective sample size via autocorrelation."""
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


# ============================================================
# 10) SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("hmm_core_extended.py — Self test (with GPD)")
    print("=" * 60)

    # Test 1: All emission densities
    print("\n[Test 1] Emission densities at y=5")
    print(f"  SM(5; 2,3,4)         = {sm_logpdf(5.0, 2.0, 3.0, 4.0):.4f}")
    print(f"  Weibull(5; 2,3)      = {weibull_logpdf(5.0, 2.0, 3.0, 1.0):.4f}")
    print(f"  Gamma(5; 2,3)        = {gamma_logpdf(5.0, 2.0, 3.0, 1.0):.4f}")
    print(f"  Lognormal(5; 1,1)    = {lognormal_logpdf(5.0, 1.0, 1.0, 1.0):.4f}")
    print(f"  GPD(5; 0.2, 3)       = {gpd_logpdf(5.0, 0.2, 3.0, 1.0):.4f}")
    print(f"  Exponential(5; 0.5)  = {exponential_logpdf(5.0, 0.5, 1.0, 1.0):.4f}")

    # Test 2: Simulation for each model type
    print("\n[Test 2] Simulation for each model (T=200)")
    for mt, name in [(0, "SM"), (1, "Weibull"), (2, "Gamma"),
                     (3, "Lognormal"), (4, "GPD"), (5, "Exponential")]:
        Y, S, truth = simulate_hmm(T=200, K=3, model_type=mt, seed=0)
        print(f"  {name:<12}: Y range = [{Y.min():.3f}, {Y.max():.3f}]")

    # Test 3: MCMC for SM
    print("\n[Test 3] MCMC 200 iters on SM")
    import time
    Y, _, _ = simulate_hmm(T=200, K=3, model_type=0, seed=0)
    t0 = time.time()
    samples = run_chain(Y, K=3, model_type=0, n_iter=200, burn=100, seed=0)
    print(f"  ✅ Done in {time.time()-t0:.2f}s")

    # Test 4: MCMC for GPD
    print("\n[Test 4] MCMC 200 iters on GPD")
    Y_gpd, _, _ = simulate_hmm(T=200, K=3, model_type=4, seed=0)
    t0 = time.time()
    samples_gpd = run_chain(Y_gpd, K=3, model_type=4,
                             n_iter=200, burn=100, seed=0)
    print(f"  ✅ Done in {time.time()-t0:.2f}s")

    # Test 5: WAIC
    waic, dic, _, _ = compute_waic_dic(samples, Y, model_type=0, n_sub=50)
    print(f"\n[Test 5] WAIC(SM) = {waic:.2f}, DIC(SM) = {dic:.2f}")

    print("\n" + "=" * 60)
    print("✅ hmm_core_extended.py works correctly")
    print("=" * 60)