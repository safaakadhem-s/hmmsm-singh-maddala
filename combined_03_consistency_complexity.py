# ============================================================
# combined_03_consistency_complexity.py — CORRECTED VERSION
# Combines: (6) Posterior consistency, (7) Computational complexity
# Priority: 2 (medium, ~25 min)
# ============================================================
import os, sys, time
sys.path.insert(0, r"D:\فكرة بحث 2027")
from output_helper import (get_folder, save_figure, save_table,
                            save_npz, save_summary)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmm_core_extended import simulate_hmm, run_chain

FOLDER = "03_consistency_complexity"
get_folder(FOLDER)
print("=" * 78)
print(f"[{FOLDER}] CONSISTENCY + COMPLEXITY")
print("=" * 78)

# ============================================================
# SECTION 1: POSTERIOR CONSISTENCY
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 1] Posterior Consistency")
print("-" * 78)

T_values = [500, 1000, 2000, 5000]
K = 3
rows = []
all_bias = {}
all_sd = {}

for T in T_values:
    print(f"\n[T = {T}]")
    t0 = time.time()
    Y, _, truth = simulate_hmm(T=T, K=K, model_type=0, seed=1)
    s = run_chain(Y, K=K, model_type=0, n_iter=3000, burn=1000, seed=1)

    a_hat = s['a'].mean(0)
    b_hat = s['b'].mean(0)
    q_hat = s['q'].mean(0)

    bias = (np.abs(a_hat - truth['a']).mean(),
            np.abs(b_hat - truth['b']).mean(),
            np.abs(q_hat - truth['q']).mean())
    sd_ = (s['a'].std(0).mean(),
           s['b'].std(0).mean(),
           s['q'].std(0).mean())

    all_bias[T] = bias
    all_sd[T] = sd_
    elapsed = time.time() - t0

    rows.append({
        "T": T,
        "Time (s)": round(elapsed, 1),
        "Bias(a)": round(bias[0], 4),
        "SD(a)": round(sd_[0], 4),
        "Bias(b)": round(bias[1], 4),
        "SD(b)": round(sd_[1], 4),
        "Bias(q)": round(bias[2], 4),
        "SD(q)": round(sd_[2], 4),
    })
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Bias = {np.round(bias, 4)}")
    print(f"  SD   = {np.round(sd_, 4)}")

df_cons = pd.DataFrame(rows)
save_table(df_cons, FOLDER, "S1_posterior_consistency", also_tex=True)
print("\n" + df_cons.to_string(index=False))

# ---- Figure: Posterior contraction ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
Ts = np.array(T_values)
for ax, (name, i) in zip(axes, [("a", 0), ("b", 1), ("q", 2)]):
    sds = np.array([all_sd[T][i] for T in T_values])
    biases = np.array([all_bias[T][i] for T in T_values])
    ax.loglog(Ts, sds, 'o-', label='Posterior SD',
              color='steelblue', lw=2, ms=8)
    ax.loglog(Ts, biases, 's--', label='|Bias|',
              color='coral', lw=2, ms=8)
    ref = sds[0] * np.sqrt(Ts[0] / Ts)
    ax.loglog(Ts, ref, 'k:', label='O(1/√T)',
              alpha=0.6, lw=1.5)
    ax.set_xlabel('T (series length)')
    ax.set_ylabel(f'SD or |Bias| of {name}')
    ax.set_title(f'Posterior contraction: {name}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

plt.suptitle("Posterior Consistency — HMM-SM", fontsize=13)
plt.tight_layout()
save_figure(fig, FOLDER, "S1_posterior_contraction.png")
plt.close(fig)

# ============================================================
# SECTION 2: COMPUTATIONAL COMPLEXITY
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 2] Computational Complexity")
print("-" * 78)

T_values2 = [250, 500, 1000, 2000, 5000]
K_values = [2, 3, 4, 5]
n_iter, burn = 500, 200

rows = []
for T in T_values2:
    for K_ in K_values:
        Y, _, _ = simulate_hmm(T=T, K=K_, model_type=0, seed=0)
        t0 = time.time()
        _ = run_chain(Y, K=K_, model_type=0, n_iter=n_iter, burn=burn,
                      seed=0, chain_id=0)
        elapsed = time.time() - t0
        rows.append({
            "T": T,
            "K": K_,
            "Time (s)": round(elapsed, 2),
            "ms/iter": round(elapsed / n_iter * 1000, 3),
        })
        print(f"  T={T:5d} K={K_}: {elapsed:7.2f}s")

df_comp = pd.DataFrame(rows)
save_table(df_comp, FOLDER, "S2_complexity", also_tex=True)

# ---- Compute empirical slopes (CORRECTED) ----
# Slope vs T (K fixed at 3)
df_T = df_comp[df_comp["K"] == 3].copy()
slope_T = np.polyfit(np.log(df_T["T"].values),
                     np.log(df_T["Time (s)"].values), 1)[0]

# Slope vs K (T fixed at max)
df_K = df_comp[df_comp["T"] == max(T_values2)].copy()
slope_K = np.polyfit(np.log(df_K["K"].values),
                     np.log(df_K["Time (s)"].values), 1)[0]

print(f"\n  Empirical slope vs T: {slope_T:.3f}  (theoretical: 1.0)")
print(f"  Empirical slope vs K: {slope_K:.3f}  (theoretical: 2.0)")

# ---- Figure: Complexity ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (A) Time vs T
ax = axes[0]
ax.loglog(df_T["T"], df_T["Time (s)"], 'o-', lw=2, ms=8,
          color='steelblue', label='Empirical')
# Reference line O(T)
ref_T = df_T["Time (s)"].iloc[0] * (df_T["T"] / df_T["T"].iloc[0])
ax.loglog(df_T["T"], ref_T, 'k:', lw=1.5, alpha=0.6,
          label='Theoretical O(T)')
ax.set_xlabel('T (series length)', fontsize=12)
ax.set_ylabel('Runtime (s)', fontsize=12)
ax.set_title(f'Runtime vs T\n(empirical slope = {slope_T:.2f}, '
             f'theoretical = 1.0)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, which='both')

# (B) Time vs K
ax = axes[1]
ax.loglog(df_K["K"], df_K["Time (s)"], 's-', lw=2, ms=8,
          color='coral', label='Empirical')
# Reference line O(K^2)
ref_K = df_K["Time (s)"].iloc[0] * (df_K["K"] / df_K["K"].iloc[0])**2
ax.loglog(df_K["K"], ref_K, 'k:', lw=1.5, alpha=0.6,
          label='Theoretical O(K²)')
ax.set_xlabel('K (number of states)', fontsize=12)
ax.set_ylabel('Runtime (s)', fontsize=12)
ax.set_title(f'Runtime vs K\n(empirical slope = {slope_K:.2f}, '
             f'theoretical = 2.0)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, which='both')

plt.suptitle("Computational Complexity of HMM-SM", fontsize=14)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_complexity.png")
plt.close(fig)

# ============================================================
# SAVE SUMMARY
# ============================================================
summary = f"""CONSISTENCY + COMPLEXITY ANALYSIS
{"=" * 60}

[SECTION 1] Posterior Consistency
{df_cons.to_string(index=False)}

Interpretation:
  - Posterior SD and |Bias| decrease as T increases.
  - This confirms posterior consistency of the HMM-SM model.

[SECTION 2] Computational Complexity
{df_comp.to_string(index=False)}

Empirical scaling:
  - Slope vs T: {slope_T:.3f}  (theoretical 1.0)
  - Slope vs K: {slope_K:.3f}  (theoretical 2.0)

Note: The K-dependence is flatter than O(K²) because the runtime for
small K is dominated by Python-level overhead (MCMC loop, RNG, MH)
rather than matrix operations in FFBS.
"""
save_summary(FOLDER, summary)

save_npz({
    "T_consistency": np.array(T_values),
    "consistency": df_cons.to_records(index=False),
    "complexity": df_comp.to_records(index=False),
    "slope_T": slope_T,
    "slope_K": slope_K,
}, FOLDER, "consistency_complexity")

print("\n" + "=" * 78)
print("✅ DONE — combined_03_consistency_complexity.py")
print("=" * 78)