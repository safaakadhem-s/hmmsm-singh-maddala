# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 08:26:15 2026

@author: ROYAL2
"""

# ============================================================
# NUMERICAL VERIFICATION OF HMMSM CLAIMS
#   - Hazard functions for the two regimes
#   - P(Y > threshold | S=k)
#   - QQ plots per regime
#   - Tail index verification
# Runtime: ~10 seconds
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln

BASE_DIR = r"D:\فكرة بحث 2027\irish_rainfall"
NPZ = os.path.join(BASE_DIR, "hmm_sm_final.npz")
FIG_DIR = os.path.join(BASE_DIR, "figures")
TAB_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================
# SM FUNCTIONS
# ============================================================
def sm_pdf(y, a, b, q):
    y = np.asarray(y, float)
    out = np.zeros_like(y)
    m = y > 0
    yy = y[m]
    log_yb = np.log(yy) - np.log(b)
    alyb = a * log_yb
    l1p = np.logaddexp(0.0, alyb)
    out[m] = np.exp(np.log(a) + np.log(q) - np.log(b)
                    + (a - 1.0) * log_yb - (q + 1.0) * l1p)
    return out

def sm_survival(y, a, b, q):
    """S(y) = 1 - F(y) = [1 + (y/b)^a]^{-q}"""
    return np.power(1.0 + (y / b) ** a, -q)

def sm_hazard(y, a, b, q):
    """h(y) = f(y) / S(y)"""
    f = sm_pdf(y, a, b, q)
    S = sm_survival(y, a, b, q)
    return f / np.maximum(S, 1e-15)

def sm_median(a, b, q):
    return b * (2.0 ** (1.0 / q) - 1.0) ** (1.0 / a)

# ============================================================
# LOAD POSTERIOR
# ============================================================
data = np.load(NPZ, allow_pickle=True)
Y = data['Y'].astype(float)
K = int(data['K'])

chains = []
c = 0
while f'chain{c}_a' in data.files:
    chains.append({k: data[f'chain{c}_{k}']
                   for k in ['a', 'b', 'q', 'pi', 'Gamma', 'S']})
    c += 1

all_a = np.vstack([ch['a'] for ch in chains])
all_b = np.vstack([ch['b'] for ch in chains])
all_q = np.vstack([ch['q'] for ch in chains])
S_last = chains[0]['S'][-1]

a_mean = all_a.mean(axis=0)
b_mean = all_b.mean(axis=0)
q_mean = all_q.mean(axis=0)
alpha_mean = a_mean * q_mean

print("=" * 70)
print("HMMSM VERIFICATION")
print("=" * 70)
print(f"  T = {len(Y)}, K = {K}")
for k in range(K):
    print(f"  State {k+1}: a={a_mean[k]:.3f}  b={b_mean[k]:.3f}  "
          f"q={q_mean[k]:.3f}  alpha={alpha_mean[k]:.3f}")

# ============================================================
# 1. HAZARD FUNCTIONS
# ============================================================
y_grid = np.linspace(1.0, 350.0, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hazard
ax = axes[0]
for k in range(K):
    h_k = sm_hazard(y_grid, a_mean[k], b_mean[k], q_mean[k])
    ax.plot(y_grid, h_k, lw=2.5,
            label=f'State {k+1} (a={a_mean[k]:.2f}, α={alpha_mean[k]:.2f})')
ax.set_xlabel('Monthly rainfall y (mm)', fontsize=12)
ax.set_ylabel('Hazard rate h(y)', fontsize=12)
ax.set_title('Hazard functions for the two regimes', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# PDF for reference
ax = axes[1]
for k in range(K):
    f_k = sm_pdf(y_grid, a_mean[k], b_mean[k], q_mean[k])
    ax.plot(y_grid, f_k, lw=2.5,
            label=f'State {k+1} (b={b_mean[k]:.1f})')
ax.set_xlabel('Monthly rainfall y (mm)', fontsize=12)
ax.set_ylabel('Density f(y)', fontsize=12)
ax.set_title('State-specific densities', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_verify_hazard_and_density.png"),
            dpi=200)
plt.close()
print("\n  Saved: fig_verify_hazard_and_density.png")

# ============================================================
# 2. P(Y > threshold | S = k)
# ============================================================
thresholds = [100, 150, 200, 250, 300]

rows = []
for k in range(K):
    # Posterior samples of S(y|theta_k) for each threshold
    for thr in thresholds:
        S_k = sm_survival(thr, all_a[:, k], all_b[:, k], all_q[:, k])
        rows.append({
            'State': k + 1,
            'Threshold (mm)': thr,
            'P(Y > thr | S=k)': S_k.mean(),
            'SD': S_k.std(),
            '2.5%': np.percentile(S_k, 2.5),
            '97.5%': np.percentile(S_k, 97.5),
        })

df_pexceed = pd.DataFrame(rows)
print("\n" + "=" * 70)
print("P(Y > threshold | S = k)")
print("=" * 70)
print(df_pexceed.round(4).to_string(index=False))
df_pexceed.to_csv(os.path.join(TAB_DIR, "prob_exceedance.csv"), index=False)

# ============================================================
# 3. QQ PLOTS PER REGIME
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for k in range(K):
    y_k = Y[S_last == k]
    y_k_sorted = np.sort(y_k)
    n = len(y_k)
    # Theoretical quantiles under the fitted SM
    # Numerical CDF inversion on a fine grid
    y_fine = np.linspace(0.01, Y.max() * 1.5, 5000)
    # For each posterior draw compute median CDF (posterior-averaged)
    F_avg = np.zeros_like(y_fine)
    n_sub = min(500, all_a.shape[0])
    idx = np.random.choice(all_a.shape[0], n_sub, replace=False)
    for i in idx:
        F_k = 1.0 - sm_survival(y_fine, all_a[i, k], all_b[i, k],
                                 all_q[i, k])
        F_avg += F_k / n_sub
    # Invert
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_q = np.interp(probs, F_avg, y_fine)

    axes[k].scatter(theo_q, y_k_sorted, s=18, alpha=0.7,
                    edgecolor='black', lw=0.3)
    # Diagonal
    lim = max(theo_q.max(), y_k_sorted.max()) * 1.05
    axes[k].plot([0, lim], [0, lim], 'r--', lw=1.2, label='y = x')
    axes[k].set_xlabel('Theoretical quantiles (mm)', fontsize=11)
    axes[k].set_ylabel('Empirical quantiles (mm)', fontsize=11)
    axes[k].set_title(f'State {k+1} (n = {n})', fontsize=12)
    axes[k].grid(alpha=0.3)
    axes[k].legend(fontsize=9)
    axes[k].set_xlim(0, lim)
    axes[k].set_ylim(0, lim)

plt.suptitle('QQ plots against the fitted HMMSM per regime', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_verify_qq.png"), dpi=200)
plt.close()
print("\n  Saved: fig_verify_qq.png")

# ============================================================
# 4. TAIL COMPARISON (numerical)
# ============================================================
print("\n" + "=" * 70)
print("TAIL COMPARISON (numerical integration)")
print("=" * 70)
print(f"{'Threshold':>10} | {'P(Y>t|S=1)':>12} | {'P(Y>t|S=2)':>12} "
      f"| {'Ratio S2/S1':>12}")
print("-" * 70)
for thr in [100, 150, 200, 250, 300]:
    p1 = sm_survival(thr, a_mean[0], b_mean[0], q_mean[0])
    p2 = sm_survival(thr, a_mean[1], b_mean[1], q_mean[1])
    ratio = p2 / max(p1, 1e-15)
    print(f"{thr:>10} | {p1:>12.5f} | {p2:>12.5f} | {ratio:>12.3f}")

# ============================================================
# 5. SELF-CRITIQUE TABLE
# ============================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

# Verify α_1 < α_2 (lighter tail in state 2)
if alpha_mean[1] > alpha_mean[0]:
    print(f"  [OK] alpha_2 ({alpha_mean[1]:.3f}) > alpha_1 ({alpha_mean[0]:.3f})")
    print(f"       State 2 has a LIGHTER tail than State 1.")
else:
    print(f"  [CHECK] alpha_2 <= alpha_1")

# Verify b_2 > b_1
if b_mean[1] > b_mean[0]:
    print(f"  [OK] b_2 ({b_mean[1]:.1f}) > b_1 ({b_mean[0]:.1f})")
    print(f"       State 2 has larger SCALE than State 1.")

# Verify P(Y>200|S=2) > P(Y>200|S=1)
p200_1 = sm_survival(200, a_mean[0], b_mean[0], q_mean[0])
p200_2 = sm_survival(200, a_mean[1], b_mean[1], q_mean[1])
if p200_2 > p200_1:
    print(f"  [OK] P(Y>200|S=2) = {p200_2:.4f} > P(Y>200|S=1) = {p200_1:.4f}")
    print(f"       State 2 produces more extremes DESPITE lighter tail —")
    print(f"       because its scale is much larger.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)