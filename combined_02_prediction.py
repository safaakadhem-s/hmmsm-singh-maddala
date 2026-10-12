# ============================================================
# combined_02_prediction.py — IMPROVED VERSION
# Proper conditional forecasting using Forward Filter
# Priority: 1 (fast, ~2 min)
# ============================================================
import os, sys
sys.path.insert(0, r"D:\فكرة بحث 2027")
from output_helper import (get_folder, save_figure, save_table,
                            save_npz, save_summary, load_idata)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn, logsumexp

FOLDER = "02_prediction"
get_folder(FOLDER)
print("=" * 78)
print(f"[{FOLDER}] PREDICTION: RETURN PERIODS + PROPER FORECASTING")
print("=" * 78)

# ---- Load ----
data = load_idata()
Y = data['Y']
a_s, b_s, q_s = data['a_samples'], data['b_samples'], data['q_samples']
G_s = data['Gamma_samples']
pi_s = data['pi_samples']
K = a_s.shape[1]
T = len(Y)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def stationary(G):
    """Stationary distribution of Markov chain."""
    ev, evec = np.linalg.eig(G.T)
    j = np.argmin(np.abs(ev - 1))
    p = np.abs(np.real(evec[:, j]))
    return p / p.sum()


def sm_cdf(y, a, b, q):
    """Singh-Maddala CDF."""
    return 1 - (1 + (y/b)**a)**(-q)


def mix_cdf(y, pi, a, b, q):
    """Mixture CDF."""
    return sum(pi[k] * sm_cdf(y, a[k], b[k], q[k]) for k in range(K))


def mix_quantile(p, pi, a, b, q):
    """Quantile of the mixture distribution."""
    return brentq(lambda y: mix_cdf(y, pi, a, b, q) - p,
                  1e-6, 1e9, xtol=1e-8)


def sm_state_mean(a, b, q):
    """Mean of SM distribution (inf if a*q <= 1)."""
    if a * q > 1:
        return b * gamma_fn(1 + 1/a) * gamma_fn(q - 1/a) / gamma_fn(q)
    return np.inf


def sm_logpdf_np(y, a, b, q):
    """Singh-Maddala log-density (numpy)."""
    log_yb = np.log(y) - np.log(b)
    a_log_yb = a * log_yb
    log_1p = np.logaddexp(0, a_log_yb)
    return (np.log(a) + np.log(q) - np.log(b)
            + (a - 1) * log_yb - (q + 1) * log_1p)


def forward_filter_np(Y_obs, pi, Gamma, a, b, q, K):
    """
    Forward filter for observed data Y_obs.
    Returns log_alpha (T, K) matrix.
    """
    T_obs = len(Y_obs)
    log_em = np.zeros((T_obs, K))
    for k in range(K):
        log_em[:, k] = sm_logpdf_np(Y_obs, a[k], b[k], q[k])

    log_pi = np.log(pi)
    log_G = np.log(Gamma)

    log_alpha = np.zeros((T_obs, K))
    log_alpha[0] = log_pi + log_em[0]
    for t in range(1, T_obs):
        m = log_alpha[t-1][:, None] + log_G
        log_alpha[t] = logsumexp(m, axis=0) + log_em[t]
    return log_alpha


def predictive_state_probs(Y_train, pi, Gamma, a, b, q, K, h):
    """
    Compute P(S_{T+h} = k | Y_{1:T}) via:
    1. Forward filter on training data
    2. Propagate h steps through Gamma^h
    """
    # Forward filter
    log_alpha = forward_filter_np(Y_train, pi, Gamma, a, b, q, K)
    # Filtered state distribution at time T
    log_filtered = log_alpha[-1] - logsumexp(log_alpha[-1])
    filtered = np.exp(log_filtered)
    # Propagate h steps
    Gamma_h = np.linalg.matrix_power(Gamma, h)
    pred_state = filtered @ Gamma_h
    return pred_state


# ============================================================
# SECTION 1: RETURN PERIODS
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 1] Return Period Analysis")
print("-" * 78)

np.random.seed(0)
idx = np.random.choice(a_s.shape[0], size=min(1000, a_s.shape[0]),
                        replace=False)
periods = [2, 5, 10, 25, 50, 100, 200, 500, 1000]

rows = []
for T_rp in periods:
    p = 1 - 1/T_rp
    levels = np.array([
        mix_quantile(p, stationary(G_s[ix]),
                     a_s[ix], b_s[ix], q_s[ix])
        for ix in idx
    ])
    rows.append({
        "T (years)": T_rp,
        "Mean": round(levels.mean(), 3),
        "Median": round(np.median(levels), 3),
        "2.5%": round(np.percentile(levels, 2.5), 3),
        "97.5%": round(np.percentile(levels, 97.5), 3),
    })

df_rp = pd.DataFrame(rows)
save_table(df_rp, FOLDER, "S1_return_levels", also_tex=True)
print(df_rp.to_string(index=False))

# Figure
fig, ax = plt.subplots(figsize=(9, 6))
Ts = np.array(periods)
means = np.array([r["Mean"] for r in rows])
lows = np.array([r["2.5%"] for r in rows])
highs = np.array([r["97.5%"] for r in rows])
ax.loglog(Ts, means, 'o-', lw=2, color='steelblue', label='Posterior mean')
ax.fill_between(Ts, lows, highs, alpha=0.3, color='steelblue',
                label='95% CI')
ax.set_xlabel('Return period T (years)')
ax.set_ylabel('Return level $y_T$')
ax.set_title('Return Level Curve under HMM-SM')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
save_figure(fig, FOLDER, "S1_return_level_curve.png")
plt.close(fig)

# Tail probabilities
print("\n  Tail probabilities P(Y > y):")
rows_tail = []
for y_thr in [10, 20, 50, 100, 200, 500]:
    probs = []
    for ix in idx:
        pi = stationary(G_s[ix])
        p = sum(pi[k] * (1 + (y_thr/b_s[ix, k])**a_s[ix, k])**(-q_s[ix, k])
                for k in range(K))
        probs.append(p)
    p_mean = np.mean(probs)
    rp = 1/p_mean if p_mean > 0 else np.inf
    rows_tail.append({"Threshold": y_thr,
                      "P(Y>y)": f"{p_mean:.4e}",
                      "Return period": f"{rp:.2f}"})
save_table(pd.DataFrame(rows_tail), FOLDER,
           "S1_tail_probabilities", also_tex=True)

# ============================================================
# SECTION 2: OUT-OF-SAMPLE FORECASTING (PROPER)
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 2] Out-of-Sample Forecasting (Conditional)")
print("-" * 78)

# Train/test split
n_train = int(0.8 * T)
Y_train = Y[:n_train]
Y_test = Y[n_train:]
print(f"  Train: {n_train}, Test: {len(Y_test)}")

# Use posterior mean parameters
a_hat = a_s.mean(axis=0)
b_hat = b_s.mean(axis=0)
q_hat = q_s.mean(axis=0)
pi_hat = pi_s.mean(axis=0)
G_hat = G_s.mean(axis=0)

# Compute filtered state distribution at T_train using ALL training data
log_alpha = forward_filter_np(Y_train, pi_hat, G_hat,
                               a_hat, b_hat, q_hat, K)
log_filtered = log_alpha[-1] - logsumexp(log_alpha[-1])
state_dist = np.exp(log_filtered)
print(f"\n  Filtered state dist at end of training: {state_dist.round(4)}")

# State means
mu_k = np.array([sm_state_mean(a_hat[k], b_hat[k], q_hat[k])
                 for k in range(K)])
print(f"  State means: {mu_k.round(4)}")

# ============================================================
# 1-step-ahead forecasting
# ============================================================
preds_mean = np.zeros(len(Y_test))
preds_low = np.zeros(len(Y_test))
preds_high = np.zeros(len(Y_test))

# For each test point, use ALL available data (rolling window)
Y_available = Y_train.copy()
current_state_dist = state_dist.copy()

for h in range(len(Y_test)):
    # Predictive state distribution: P(S_{t+1} | Y_{1:t})
    pred_state = current_state_dist @ G_hat

    # Predictive mean: E[Y_{t+1} | Y_{1:t}]
    pred_mean = np.sum(pred_state * mu_k)
    preds_mean[h] = pred_mean

    # Predictive quantiles (approximate via grid)
    # For each state, we have SM distribution.
    # Predictive distribution = Σ_k pred_state[k] * SM(a_k, b_k, q_k)
    # Compute 2.5% and 97.5% quantiles numerically
    def pred_cdf(y):
        return sum(pred_state[k] * sm_cdf(y, a_hat[k], b_hat[k], q_hat[k])
                   for k in range(K))

    # Compute quantiles via brentq
    try:
        q_lo = brentq(lambda y: pred_cdf(y) - 0.025, 1e-6, 1e8)
        q_hi = brentq(lambda y: pred_cdf(y) - 0.975, 1e-6, 1e8)
    except Exception:
        q_lo, q_hi = 0.0, np.inf
    preds_low[h] = q_lo
    preds_high[h] = q_hi

    # Update state distribution using ACTUAL observation
    y_true = Y_test[h]
    # Emission likelihood for each state
    log_em = np.array([sm_logpdf_np(y_true, a_hat[k], b_hat[k], q_hat[k])
                       for k in range(K)])
    # Posterior state distribution AFTER observing y_true
    log_post = np.log(pred_state) + log_em
    log_post_norm = log_post - logsumexp(log_post)
    current_state_dist = np.exp(log_post_norm)

# ---- Metrics ----
actuals = Y_test
errors = preds_mean - actuals
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))

# MAPE (with cutoff to avoid division by zero)
cutoff = 0.5  # only for values > cutoff
mask = actuals > cutoff
if mask.sum() > 0:
    mape = 100 * np.mean(np.abs(errors[mask] / actuals[mask]))
else:
    mape = np.nan

# Coverage of 95% PI
in_interval = (actuals >= preds_low) & (actuals <= preds_high)
coverage = 100 * in_interval.mean()

# Bias
bias = errors.mean()

print(f"\n  ✅ Forecast metrics (proper conditional):")
print(f"     RMSE                = {rmse:.4f}")
print(f"     MAE                 = {mae:.4f}")
print(f"     MAPE (y > 0.5 mm)   = {mape:.2f}%")
print(f"     Bias                = {bias:+.4f}")
print(f"     95% PI coverage     = {coverage:.2f}%")

df_metrics = pd.DataFrame([
    {"Metric": "RMSE", "Value": round(rmse, 4)},
    {"Metric": "MAE", "Value": round(mae, 4)},
    {"Metric": "MAPE (%)", "Value": round(mape, 2)},
    {"Metric": "Bias", "Value": round(bias, 4)},
    {"Metric": "95% PI coverage (%)", "Value": round(coverage, 2)},
    {"Metric": "Train size", "Value": n_train},
    {"Metric": "Test size", "Value": len(Y_test)},
])
save_table(df_metrics, FOLDER, "S2_forecast_metrics", also_tex=True)
print("\n" + df_metrics.to_string(index=False))

# ============================================================
# Figure: Forecast vs Actual
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Full view
ax = axes[0]
ax.plot(range(len(actuals)), actuals, 'k-', lw=0.8,
        label='Actual', alpha=0.7)
ax.plot(range(len(preds_mean)), preds_mean, 'r--', lw=1.2,
        label='Forecast mean')
ax.fill_between(range(len(preds_mean)),
                preds_low, preds_high,
                alpha=0.2, color='red', label='95% PI')
ax.set_xlabel('Test index')
ax.set_ylabel('Y')
ax.set_title(f'Out-of-sample forecasting — full view '
             f'(RMSE = {rmse:.3f}, Coverage = {coverage:.1f}%)')
ax.legend()
ax.grid(alpha=0.3)

# Zoom (first 100 points)
ax = axes[1]
n_show = min(100, len(actuals))
ax.plot(range(n_show), actuals[:n_show], 'k-', lw=1, marker='o',
        ms=3, label='Actual')
ax.plot(range(n_show), preds_mean[:n_show], 'r--', lw=1.5,
        label='Forecast mean')
ax.fill_between(range(n_show),
                preds_low[:n_show], preds_high[:n_show],
                alpha=0.2, color='red', label='95% PI')
ax.set_xlabel('Test index (first 100 points)')
ax.set_ylabel('Y')
ax.set_title('Zoom: first 100 test points')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
save_figure(fig, FOLDER, "S2_forecast_vs_actual.png")
plt.close(fig)

# ============================================================
# Figure: Prediction interval coverage
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
# Sort by predicted value
sorted_idx = np.argsort(preds_mean)
actual_sorted = actuals[sorted_idx]
mean_sorted = preds_mean[sorted_idx]
low_sorted = preds_low[sorted_idx]
high_sorted = preds_high[sorted_idx]

x = np.arange(len(sorted_idx))
ax.scatter(x, actual_sorted, s=3, alpha=0.5, color='black', label='Actual')
ax.plot(x, mean_sorted, 'r-', lw=1.5, label='Forecast mean')
ax.fill_between(x, low_sorted, high_sorted, alpha=0.25,
                color='red', label='95% PI')
ax.set_xlabel('Test index (sorted by forecast)')
ax.set_ylabel('Y')
ax.set_title(f'Prediction intervals (coverage = {coverage:.1f}%)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_prediction_intervals.png")
plt.close(fig)

# ============================================================
# SAVE SUMMARY
# ============================================================
summary = f"""PREDICTION ANALYSIS (IMPROVED)
{"="*50}

[SECTION 1] Return Periods
{df_rp.to_string(index=False)}

[SECTION 2] Forecasting (conditional)
{df_metrics.to_string(index=False)}

Notes:
- Forecasting uses proper conditional predictive distribution
  p(Y_{{t+1}} | Y_{{1:t}}) via forward filter + state propagation.
- MAPE computed only for y > 0.5 mm (avoids division by near-zero).
- 95% PI coverage should be close to 95% for well-calibrated model.
"""
save_summary(FOLDER, summary)

save_npz({
    "periods": np.array(periods),
    "return_means": means,
    "return_lows": lows,
    "return_highs": highs,
    "pred_mean": preds_mean,
    "pred_low": preds_low,
    "pred_high": preds_high,
    "actual": actuals,
}, FOLDER, "prediction")

print("\n" + "=" * 78)
print("✅ DONE — combined_02_prediction.py (IMPROVED)")
print("=" * 78)