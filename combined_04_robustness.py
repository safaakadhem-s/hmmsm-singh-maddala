# ============================================================
# combined_04_robustness.py — FINAL VERSION
# Section 1: Misspecification (HMM-SM vs Weibull vs GPD vs Gaussian)
# Section 2: K-Selection (K=2,3,4,5)
# Bayesian criteria only: WAIC, DIC, LOO
# ============================================================
import os, sys, time
sys.path.insert(0, r"D:\فكرة بحث 2027")
from output_helper import (get_folder, save_figure, save_table,
                            save_npz, save_summary)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from hmm_core_extended import (simulate_hmm, run_chain, compute_waic_dic,
                                pointwise_ll_numpy)

FOLDER = "04_robustness"
get_folder(FOLDER)
print("=" * 78)
print(f"[{FOLDER}] ROBUSTNESS (BAYESIAN)")
print("=" * 78)


# ============================================================
# HELPER: WAIC + DIC + LOO
# ============================================================
def compute_all_bayesian_criteria(samples, Y, model_type=0, n_sub=500):
    """
    Compute WAIC, DIC, and LOO (harmonic mean).
    All three are fully Bayesian criteria.
    """
    n = samples['a'].shape[0]
    idx = np.random.choice(n, size=min(n_sub, n), replace=False)
    T = len(Y)

    log_lik = np.zeros((len(idx), T))
    for i, ix in enumerate(idx):
        log_lik[i] = pointwise_ll_numpy(
            Y, samples['pi'][ix], samples['Gamma'][ix],
            samples['a'][ix], samples['b'][ix], samples['q'][ix],
            model_type)

    # Numerical safety
    log_lik = np.clip(log_lik, -1e10, 100.0)

    # ---- WAIC ----
    lppd = (logsumexp(log_lik, axis=0) - np.log(len(idx))).sum()
    p_waic = log_lik.var(axis=0).sum()
    waic = -2 * (lppd - p_waic)

    # ---- DIC ----
    D_bar = -2 * log_lik.mean(axis=0).sum()
    p_D = D_bar - (-2 * lppd)
    dic = D_bar + p_D

    # ---- LOO via harmonic mean ----
    try:
        S = log_lik.shape[0]
        log_p_loo_t = np.log(S) - logsumexp(-log_lik, axis=0)
        loo = -2 * log_p_loo_t.sum()
    except Exception:
        loo = np.nan

    # Sanity check
    if waic < 0:
        print(f"  ⚠️  WARNING: WAIC < 0. Setting to NaN.")
        waic = np.nan
        dic = np.nan
        loo = np.nan

    return waic, dic, loo, lppd, p_waic


# ============================================================
# SECTION 1: MISSPECIFICATION STUDY
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 1] Misspecification Study")
print("-" * 78)
print("  Bayesian criteria only: WAIC, DIC, LOO")
print("  Models: HMM-SM, HMM-Weibull, HMM-GPD, HMM-Gaussian")

configs = [
    ("HMM-SM", 0),
    ("HMM-Weibull", 1),
    ("HMM-GPD", 4),
    ("HMM-Gaussian", 5),
]

rows = []
for name, mt in configs:
    print(f"\n[{name}]")
    Y, _, _ = simulate_hmm(T=800, K=3, model_type=mt, seed=1)
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    s = run_chain(Y, K=3, model_type=0, n_iter=3000, burn=1000, seed=1)

    waic, dic, loo, lppd, p_waic = compute_all_bayesian_criteria(
        s, Y, model_type=0, n_sub=500)

    rows.append({
        "True model": name,
        "WAIC": round(waic, 2) if np.isfinite(waic) else "N/A",
        "DIC": round(dic, 2) if np.isfinite(dic) else "N/A",
        "LOO": round(loo, 2) if np.isfinite(loo) else "N/A",
        "lppd": round(lppd, 2),
        "p_waic": round(p_waic, 2),
    })
    print(f"  WAIC = {waic:.2f}")
    print(f"  DIC  = {dic:.2f}")
    print(f"  LOO  = {loo:.2f}")

df_mis = pd.DataFrame(rows)
save_table(df_mis, FOLDER, "S1_misspecification", also_tex=True)
print("\n" + df_mis.to_string(index=False))

# ---- Figure: WAIC comparison ----
fig, ax = plt.subplots(figsize=(11, 6))
models_list = df_mis["True model"].tolist()
waic_vals = [float(v) if v != "N/A" else np.nan for v in df_mis["WAIC"]]
colors = ['green', 'steelblue', 'purple', 'coral']

bars = ax.bar(models_list, waic_vals, color=colors, alpha=0.85)
valid_idx = [i for i, v in enumerate(waic_vals) if np.isfinite(v)]
if valid_idx:
    best = min(valid_idx, key=lambda i: waic_vals[i])
    bars[best].set_edgecolor('red')
    bars[best].set_linewidth(3)

# Annotate values
for bar, v in zip(bars, waic_vals):
    if np.isfinite(v):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + max(waic_vals)*0.02,
                f'{v:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('WAIC', fontsize=12)
ax.set_title('Misspecification Study: WAIC when fitting HMM-SM\n'
             'to data generated from different models',
             fontsize=12)
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
save_figure(fig, FOLDER, "S1_misspecification.png")
plt.close(fig)


# ============================================================
# SECTION 2: K-SELECTION
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 2] K-Selection Study (Bayesian)")
print("-" * 78)

T = 800
K_true = 3
Y, S_true, _ = simulate_hmm(T=T, K=K_true, model_type=0, seed=1)
print(f"  Data: T={T}, K_true={K_true}")

K_values = [2, 3, 4, 5]
rows = []
for K_ in K_values:
    print(f"\n[K = {K_}]")
    t0 = time.time()
    s = run_chain(Y, K=K_, model_type=0, n_iter=5000, burn=2000,
                  seed=0, chain_id=0)

    waic, dic, loo, lppd, p_waic = compute_all_bayesian_criteria(
        s, Y, model_type=0, n_sub=500)

    n_params = K_ * (K_ - 1) + (K_ - 1) + 3 * K_

    rows.append({
        "K": K_,
        "# params": n_params,
        "WAIC": round(waic, 2) if np.isfinite(waic) else "N/A",
        "DIC": round(dic, 2) if np.isfinite(dic) else "N/A",
        "LOO": round(loo, 2) if np.isfinite(loo) else "N/A",
        "p_waic": round(p_waic, 2),
        "Time (s)": round(time.time() - t0, 1),
    })
    print(f"  WAIC={waic:.2f}  DIC={dic:.2f}  LOO={loo:.2f}")

df_k = pd.DataFrame(rows)
save_table(df_k, FOLDER, "S2_k_selection", also_tex=True)
print("\n" + df_k.to_string(index=False))

# ---- Find best K ----
waic_vals = [float(v) if v != "N/A" else np.inf for v in df_k["WAIC"]]
best_K_waic = K_values[int(np.argmin(waic_vals))]
dic_vals = [float(v) if v != "N/A" else np.inf for v in df_k["DIC"]]
best_K_dic = K_values[int(np.argmin(dic_vals))]
loo_vals = [float(v) if v != "N/A" else np.inf for v in df_k["LOO"]]
best_K_loo = K_values[int(np.argmin(loo_vals))]

print(f"\n  ✅ Best K by WAIC: {best_K_waic}")
print(f"  ✅ Best K by DIC : {best_K_dic}")
print(f"  ✅ Best K by LOO : {best_K_loo}")
print(f"  ✅ True K        : {K_true}")

# ---- Figure: WAIC and DIC vs K ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
waic_plot = [v if np.isfinite(v) else np.nan for v in waic_vals]
ax.plot(K_values, waic_plot, 'o-', lw=2, ms=10, color='steelblue')
ax.scatter([best_K_waic], [min(waic_plot)], color='red',
           s=300, zorder=5, marker='*', label=f'Best K={best_K_waic}')
ax.axvline(K_true, color='green', ls='--', lw=2, alpha=0.6,
           label=f'True K={K_true}')
ax.set_xlabel('K (number of states)', fontsize=12)
ax.set_ylabel('WAIC', fontsize=12)
ax.set_title('WAIC vs K (lower = better)', fontsize=12)
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
dic_plot = [v if np.isfinite(v) else np.nan for v in dic_vals]
ax.plot(K_values, dic_plot, 's-', lw=2, ms=10, color='coral')
ax.scatter([best_K_dic], [min(dic_plot)], color='red',
           s=300, zorder=5, marker='*', label=f'Best K={best_K_dic}')
ax.axvline(K_true, color='green', ls='--', lw=2, alpha=0.6,
           label=f'True K={K_true}')
ax.set_xlabel('K (number of states)', fontsize=12)
ax.set_ylabel('DIC', fontsize=12)
ax.set_title('DIC vs K (lower = better)', fontsize=12)
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle("Bayesian K-Selection: WAIC and DIC", fontsize=14)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_k_selection.png")
plt.close(fig)

# ---- Figure: LOO vs K ----
fig, ax = plt.subplots(figsize=(8, 5))
loo_plot = [v if np.isfinite(v) else np.nan for v in loo_vals]
ax.plot(K_values, loo_plot, '^-', lw=2, ms=10, color='purple')
ax.scatter([best_K_loo], [min(loo_plot)], color='red',
           s=300, zorder=5, marker='*', label=f'Best K={best_K_loo}')
ax.axvline(K_true, color='green', ls='--', lw=2, alpha=0.6,
           label=f'True K={K_true}')
ax.set_xlabel('K (number of states)', fontsize=12)
ax.set_ylabel('LOO (PSIS)', fontsize=12)
ax.set_title('LOO vs K (lower = better)', fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_k_selection_loo.png")
plt.close(fig)


# ============================================================
# SAVE SUMMARY
# ============================================================
summary = f"""ROBUSTNESS ANALYSIS (BAYESIAN)
{"=" * 60}

NOTE: AIC and BIC excluded (frequentist criteria).
All criteria here (WAIC, DIC, LOO) are fully Bayesian.

NOTE: HMM-Gamma and HMM-Lognormal excluded due to
tail incompatibility with SM data. HMM-GPD and 
HMM-Gaussian included as alternative benchmarks.

[SECTION 1] Misspecification Study
{df_mis.to_string(index=False)}

[SECTION 2] K-Selection Study
{df_k.to_string(index=False)}

Best K by WAIC : {best_K_waic}
Best K by DIC  : {best_K_dic}
Best K by LOO  : {best_K_loo}
True K         : {K_true}
"""
save_summary(FOLDER, summary)

save_npz({
    "misspecification": df_mis.to_records(index=False),
    "k_selection": df_k.to_records(index=False),
    "best_K_waic": best_K_waic,
    "best_K_dic": best_K_dic,
    "best_K_loo": best_K_loo,
}, FOLDER, "robustness")

print("\n" + "=" * 78)
print("✅ DONE — combined_04_robustness.py (Bayesian)")
print("=" * 78)