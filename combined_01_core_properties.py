# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:21:46 2026

@author: ROYAL2
"""

# ============================================================
# combined_01_core_properties.py
# Combines: (1) Model properties, (2) Parameter dependence,
#           (3) Identifiability, (4) Label switching
# Priority: 1 (fast, ~2 min)
# ============================================================
import os, sys
sys.path.insert(0, r"D:\فكرة بحث 2027")
from output_helper import (get_folder, save_figure, save_table,
                            save_npz, save_summary, load_idata)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn
from scipy.stats import pearsonr, spearmanr

FOLDER = "01_core_properties"
get_folder(FOLDER)
print("=" * 78)
print(f"[{FOLDER}] CORE PROPERTIES OF HMM-SM")
print("=" * 78)

# ---- Load data ----
data = load_idata()
Y = data['Y']
a_true, b_true, q_true = data['a_true'], data['b_true'], data['q_true']
Gamma_true = data['Gamma_true']
a_s, b_s, q_s = data['a_samples'], data['b_samples'], data['q_samples']
G_s = data['Gamma_samples']
K = len(a_true)

# ============================================================
# SECTION 1: MODEL PROPERTIES
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 1] Model Properties")
print("-" * 78)

def stationary(Gamma):
    ev, evec = np.linalg.eig(Gamma.T)
    j = np.argmin(np.abs(ev - 1))
    p = np.abs(np.real(evec[:, j]))
    return p / p.sum()

pi_star = stationary(Gamma_true)
pi_star_post = np.array([stationary(G) for G in G_s])

# Dwell times
mean_dur = 1.0 / (1 - np.diag(Gamma_true))
var_dur = np.diag(Gamma_true) / (1 - np.diag(Gamma_true))**2

# Marginal moments
def sm_moment(a, b, q, r):
    if r >= a*q: return np.inf
    return b**r * gamma_fn(1 + r/a) * gamma_fn(q - r/a) / gamma_fn(q)

moments = {r: sum(pi_star[k] * sm_moment(a_true[k], b_true[k], q_true[k], r)
                  for k in range(K)) for r in range(1, 5)}
mean = moments[1]
var = moments[2] - mean**2
sd = np.sqrt(var)
skew_val = (moments[3] - 3*mean*var - mean**3)/sd**3 if np.isfinite(moments[3]) else np.inf
kurt_val = (moments[4] - 4*mean*moments[3] + 6*mean**2*moments[2] - 3*mean**4)/var**2 if np.isfinite(moments[4]) else np.inf

df_stationary = pd.DataFrame({
    "State": [f"State {k+1}" for k in range(K)],
    "π* (theoretical)": pi_star.round(4),
    "π* (posterior mean)": pi_star_post.mean(axis=0).round(4),
    "π* (posterior SD)": pi_star_post.std(axis=0).round(4),
    "Mean dwell time": mean_dur.round(3),
    "SD dwell time": np.sqrt(var_dur).round(3),
})
save_table(df_stationary, FOLDER, "S1_stationary_dwell", also_tex=True)

df_moments = pd.DataFrame({
    "Moment": ["Mean", "Variance", "SD", "Skewness", "Kurtosis"],
    "Value": [f"{mean:.4f}", f"{var:.4f}", f"{sd:.4f}",
              f"{skew_val:.4f}", f"{kurt_val:.4f}"],
})
save_table(df_moments, FOLDER, "S1_marginal_moments", also_tex=True)

# Figures
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(K)
axes[0].bar(x, pi_star, color='steelblue', alpha=0.8)
axes[0].errorbar(x, pi_star_post.mean(axis=0),
                 yerr=1.96*pi_star_post.std(axis=0),
                 fmt='none', color='red', capsize=8, lw=2)
axes[0].set_xticks(x); axes[0].set_xticklabels([f'S{k+1}' for k in range(K)])
axes[0].set_ylabel('π*'); axes[0].set_title('Stationary distribution')
axes[0].grid(alpha=0.3, axis='y')

y_grid = np.linspace(0.01, np.percentile(Y, 99.5), 500)
pdf_marg = np.zeros_like(y_grid)
for k in range(K):
    log_yb = np.log(y_grid) - np.log(b_true[k])
    alyb = a_true[k] * log_yb
    log_f = (np.log(a_true[k]) + np.log(q_true[k]) - np.log(b_true[k])
             + (a_true[k]-1)*log_yb - (q_true[k]+1)*np.logaddexp(0, alyb))
    pdf_marg += pi_star[k] * np.exp(log_f)
axes[1].plot(y_grid, pdf_marg, 'k-', lw=2, label='Marginal')
axes[1].hist(Y, bins=60, density=True, alpha=0.2, color='gray', label='Observed')
axes[1].set_xlabel('Y'); axes[1].set_ylabel('Density')
axes[1].set_title('Marginal distribution of Y')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, FOLDER, "S1_properties.png")
plt.close(fig)

print(f"  ✅ π* = {pi_star.round(4)}")
print(f"  ✅ Mean = {mean:.4f}, SD = {sd:.4f}, Skew = {skew_val:.4f}")

# ============================================================
# SECTION 2: PARAMETER DEPENDENCE
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 2] Parameter Dependence")
print("-" * 78)

rows = []
fig, axes = plt.subplots(K, 3, figsize=(15, 4*K))
for k in range(K):
    a_k, b_k, q_k = a_s[:, k], b_s[:, k], q_s[:, k]
    r_ab, _ = pearsonr(a_k, b_k); r_aq, _ = pearsonr(a_k, q_k); r_bq, _ = pearsonr(b_k, q_k)
    s_ab, _ = spearmanr(a_k, b_k); s_aq, _ = spearmanr(a_k, q_k); s_bq, _ = spearmanr(b_k, q_k)
    rows.append({"State": f"S{k+1}",
                 "Pearson(a,b)": round(r_ab, 4), "Pearson(a,q)": round(r_aq, 4),
                 "Pearson(b,q)": round(r_bq, 4),
                 "Spearman(a,b)": round(s_ab, 4), "Spearman(a,q)": round(s_aq, 4),
                 "Spearman(b,q)": round(s_bq, 4)})
    for ax, (x_, y_, name, r_) in zip(axes[k],
        [(a_k, b_k, 'a-b', r_ab), (a_k, q_k, 'a-q', r_aq), (b_k, q_k, 'b-q', r_bq)]):
        ax.scatter(x_, y_, s=1, alpha=0.3, color='steelblue')
        ax.set_xlabel(name.split('-')[0]); ax.set_ylabel(name.split('-')[1])
        ax.set_title(f'S{k+1}: {name} (r={r_:+.3f})')
        ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, FOLDER, "S2_parameter_dependence.png")
plt.close(fig)

df = pd.DataFrame(rows)
save_table(df, FOLDER, "S2_parameter_correlations", also_tex=True)
print(df.to_string(index=False))

# ============================================================
# SECTION 3: IDENTIFIABILITY
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 3] Local Identifiability")
print("-" * 78)

rows = []
for k in range(K):
    X = np.column_stack([np.log(a_s[:, k]), np.log(b_s[:, k]), np.log(q_s[:, k])])
    cov = np.cov(X.T)
    cond = np.linalg.cond(cov)
    rank = np.linalg.matrix_rank(cov)
    rows.append({"State": f"S{k+1}", "Rank": rank,
                 "Condition number": f"{cond:.2e}",
                 "Status": "✅ OK" if (rank == 3 and cond < 1e6) else "⚠️ Check"})
df = pd.DataFrame(rows)
save_table(df, FOLDER, "S3_identifiability", also_tex=True)
print(df.to_string(index=False))

# ============================================================
# SECTION 4: LABEL SWITCHING
# ============================================================
print("\n" + "-" * 78)
print("[SECTION 4] Label Switching Diagnostic")
print("-" * 78)

orderings = np.argsort(b_s, axis=1)
unique, counts = np.unique(orderings, axis=0, return_counts=True)
sort_idx = np.argsort(-counts)
most_common_pct = counts[sort_idx[0]] / b_s.shape[0] * 100

rows = [{"Ordering": str(tuple(unique[i])), "Count": counts[i],
         "Percentage": f"{counts[i]/b_s.shape[0]*100:.1f}%"}
        for i in sort_idx[:5]]
df = pd.DataFrame(rows)
save_table(df, FOLDER, "S4_label_switch_orderings", also_tex=True)

status = ("✅ No label switching" if most_common_pct > 95 else
          "⚠️ Moderate" if most_common_pct > 80 else "❌ Severe")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot([b_s[:, k] for k in range(K)], labels=[f'S{k+1}' for k in range(K)])
axes[0].set_title('BEFORE relabeling'); axes[0].set_ylabel('b'); axes[0].grid(alpha=0.3)
b_relabel = np.sort(b_s, axis=1)
axes[1].boxplot([b_relabel[:, k] for k in range(K)], labels=[f'S{k+1}' for k in range(K)])
axes[1].set_title('AFTER relabeling by b'); axes[1].set_ylabel('b'); axes[1].grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, FOLDER, "S4_label_switching.png")
plt.close(fig)

# ---- Save summary ----
summary = f"""CORE PROPERTIES OF HMM-SM
{"="*50}

[SECTION 1] Model Properties
  π* = {pi_star.round(4)}
  Mean = {mean:.4f}, SD = {sd:.4f}
  Skew = {skew_val:.4f}, Kurt = {kurt_val:.4f}
  Dwell times: {mean_dur.round(2)}

[SECTION 2] Parameter Dependence
{df.to_string(index=False) if False else ''}

[SECTION 3] Identifiability
  Check figures/tables/S3_identifiability.csv

[SECTION 4] Label Switching
  Most common ordering: {tuple(unique[sort_idx[0]])}
  Frequency: {most_common_pct:.1f}%
  Status: {status}

All figures and tables saved to {FOLDER}/
"""
save_summary(FOLDER, summary)
save_npz({"pi_star": pi_star, "pi_star_post": pi_star_post,
          "mean_dur": mean_dur, "mean": mean, "var": var,
          "skew": skew_val, "kurt": kurt_val}, FOLDER, "core")

print("\n✅ DONE — combined_01_core_properties.py")