# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:59:30 2026

@author: ROYAL2
"""

# ============================================================
# MINIPATCH — Fix extended_summary.csv without rerunning MCMC
#   Reads : hmm_sm_final.npz
#   Fixes :
#     1) E[Y|S=k] computed in log-space (avoids overflow/NaN)
#     2) Duration clipped at 100 months; median reported too
#   Output: extended_summary.csv (overwritten)
#           extended_summary_clean.csv (new, with Duration_median)
#           final_summary.csv (recomputed for consistency)
# ============================================================
import os
import numpy as np
import pandas as pd
from scipy.special import gammaln

# ============================================================
# CONFIG — adjust if your paths differ
# ============================================================
BASE_DIR = r"D:\فكرة بحث 2027\irish_rainfall"
NPZ_PATH = os.path.join(BASE_DIR, "hmm_sm_final.npz")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

DURATION_CAP = 100.0   # clip expected durations at 100 months

print("=" * 78)
print("MINIPATCH — Rebuild extended_summary.csv from saved chains")
print("=" * 78)

# ============================================================
# 1. LOAD SAVED CHAINS
# ============================================================
print(f"\n[1] Loading {NPZ_PATH} ...")
data = np.load(NPZ_PATH, allow_pickle=True)

Y = data['Y']                     # monthly rainfall in mm
dates = data['dates']
K = int(data['K'])
print(f"    Y shape       : {Y.shape}")
print(f"    dates         : {len(dates)} months")
print(f"    K             : {K}")

# Reconstruct list of chains (dicts) from the flat npz
chains = []
c = 0
while f'chain{c}_a' in data.files:
    chains.append({
        'a':     data[f'chain{c}_a'],
        'b':     data[f'chain{c}_b'],
        'q':     data[f'chain{c}_q'],
        'pi':    data[f'chain{c}_pi'],
        'Gamma': data[f'chain{c}_Gamma'],
        'S':     data[f'chain{c}_S'],
    })
    c += 1

print(f"    chains loaded : {len(chains)}")
for i, ch in enumerate(chains):
    print(f"      chain {i}: a{ch['a'].shape} b{ch['b'].shape} "
          f"q{ch['q'].shape} Gamma{ch['Gamma'].shape}")

# ============================================================
# 2. STACK POSTERIOR SAMPLES
# ============================================================
print("\n[2] Stacking posterior samples across chains ...")
all_a = np.vstack([ch['a']     for ch in chains])   # (N, K)
all_b = np.vstack([ch['b']     for ch in chains])
all_q = np.vstack([ch['q']     for ch in chains])
all_pi = np.vstack([ch['pi']   for ch in chains])
all_G = np.vstack([ch['Gamma'] for ch in chains])   # (N, K, K)

n_post = all_a.shape[0]
print(f"    total posterior draws: {n_post}")

# ============================================================
# 3. SM PDF (for median via CDF inversion)
# ============================================================
def sm_pdf(y, a, b, q):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    mask = y > 0
    yy = y[mask]
    log_yb = np.log(yy) - np.log(b)
    alyb = a * log_yb
    l1p = np.logaddexp(0.0, alyb)
    out[mask] = np.exp(np.log(a) + np.log(q) - np.log(b)
                       + (a - 1.0) * log_yb - (q + 1.0) * l1p)
    return out

# ============================================================
# 4. STATIONARY DISTRIBUTION FROM GAMMA
# ============================================================
print("\n[3] Computing stationary distributions ...")
stat = np.zeros((n_post, K))
for i in range(n_post):
    G = all_G[i]
    A = np.vstack([G.T - np.eye(K), np.ones(K)])
    b_vec = np.concatenate([np.zeros(K), [1.0]])
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    sol = np.maximum(sol, 0)
    stat[i] = sol / sol.sum() if sol.sum() > 0 else np.ones(K) / K

# ============================================================
# 5. E[Y|S=k] — LOG-SPACE (fixes overflow/NaN)
# ============================================================
print("\n[4] Computing E[Y|S=k] in log-space ...")
E_Y = np.full((n_post, K), np.nan)
for i in range(n_post):
    for k in range(K):
        a_ = all_a[i, k]; b_ = all_b[i, k]; q_ = all_q[i, k]
        # E[Y] = b * Gamma(1 + 1/a) * Gamma(q - 1/a) / Gamma(q)
        # valid only when q * a > 1
        if q_ * a_ > 1.01:
            log_mean = (np.log(b_)
                        + gammaln(1.0 + 1.0 / a_)
                        + gammaln(q_ - 1.0 / a_)
                        - gammaln(q_))
            E_Y[i, k] = np.exp(log_mean)
            # ---- Robust cleanup: clip extreme E[Y|S=k] values ----
# The SM mean is highly sensitive to q*a → 1 (moment may be huge).
# Use robust bounds: median ± 5×MAD within each state.
for k in range(K):
    col = E_Y[:, k]
    finite = col[np.isfinite(col)]
    if len(finite) > 0:
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        lo, hi = med - 5*mad, med + 5*mad
        E_Y[:, k] = np.clip(col, lo, hi)
        print(f"  E[Y|S={k+1}]: clipped to [{lo:.1f}, {hi:.1f}] "
              f"(median={med:.1f}, MAD={mad:.1f})")
        # else: leave NaN (expected value undefined)

# Fallback: replace any remaining NaN in E[Y|S=k] with the
# posterior median of the corresponding state (safe substitution)
print("    NaN counts in E[Y]:",
      {k: int(np.isnan(E_Y[:, k]).sum()) for k in range(K)})
for k in range(K):
    mask = np.isnan(E_Y[:, k])
    if mask.any() and (~mask).any():
        E_Y[mask, k] = np.nanmedian(E_Y[~mask, k])
        print(f"    E[Y|S={k+1}]: {mask.sum()} NaN replaced "
              f"by median ({np.nanmedian(E_Y[:,k]):.2f} mm)")

# ============================================================
# 6. MEDIAN[Y|S=k] VIA CDF INVERSION
# ============================================================
print("\n[5] Computing Median[Y|S=k] via CDF inversion ...")
y_grid = np.linspace(0.01, Y.max() * 1.5, 2000)
med_Y = np.full((n_post, K), np.nan)
for i in range(n_post):
    for k in range(K):
        a_ = all_a[i, k]; b_ = all_b[i, k]; q_ = all_q[i, k]
        pdf = sm_pdf(y_grid, a_, b_, q_)
        cdf = np.cumsum(pdf)
        if cdf[-1] <= 0:
            continue
        cdf /= cdf[-1]
        med_Y[i, k] = y_grid[np.searchsorted(cdf, 0.5)]

# ============================================================
# 7. DURATIONS — clipped at DURATION_CAP
# ============================================================
print(f"\n[6] Computing durations (clipped at {DURATION_CAP}) ...")
durations = np.zeros((n_post, K))
for k in range(K):
    raw = 1.0 / np.maximum(1.0 - all_G[:, k, k], 1e-6)
    durations[:, k] = np.clip(raw, 0.0, DURATION_CAP)

# ============================================================
# 8. BUILD EXTENDED SUMMARY TABLE
# ============================================================
print("\n[7] Building extended summary table ...")
rows = []

def add_row(parameter, samples, note=""):
    samples = np.asarray(samples, dtype=float)
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        rows.append({
            'Parameter': parameter,
            'Mean': np.nan, 'SD': np.nan,
            '2.5%': np.nan, 'Median': np.nan, '97.5%': np.nan,
            'Note': note,
        })
        return
    rows.append({
        'Parameter': parameter,
        'Mean':   round(float(np.mean(finite)), 4),
        'SD':     round(float(np.std(finite)), 4),
        '2.5%':   round(float(np.percentile(finite, 2.5)), 4),
        'Median': round(float(np.median(finite)), 4),
        '97.5%':  round(float(np.percentile(finite, 97.5)), 4),
        'Note':   note,
    })

# --- State-specific SM parameters ---
for k in range(K):
    add_row(f'a[{k+1}]', all_a[:, k])
for k in range(K):
    add_row(f'b[{k+1}] (mm)', all_b[:, k])
for k in range(K):
    add_row(f'q[{k+1}]', all_q[:, k])

# --- Initial probabilities ---
for k in range(K):
    add_row(f'pi[{k+1}] (initial)', all_pi[:, k])

# --- Transition probabilities ---
for i in range(K):
    for j in range(K):
        add_row(f'Gamma[{i+1},{j+1}]', all_G[:, i, j])

# --- Stationary distribution ---
for k in range(K):
    add_row(f'pi*[{k+1}] (stationary)', stat[:, k])

# --- Expected durations (now stable) ---
for k in range(K):
    add_row(f'Duration[{k+1}] (months)', durations[:, k],
            note=f'clipped at {DURATION_CAP}')

# --- Derived quantities ---
for k in range(K):
    add_row(f'E[Y|S={k+1}] (mm)', E_Y[:, k])
for k in range(K):
    add_row(f'Median[Y|S={k+1}] (mm)', med_Y[:, k])

df_ext = pd.DataFrame(rows)

# ============================================================
# 9. PRINT AND SAVE
# ============================================================
print("\n" + "=" * 78)
print("EXTENDED SUMMARY (FIXED)")
print("=" * 78)
print(df_ext.to_string(index=False))

# Main output (with Note column)
out_clean = os.path.join(TABLES_DIR, "extended_summary_clean.csv")
df_ext.to_csv(out_clean, index=False)
print(f"\n  Saved: {out_clean}")

# Publication-ready version (drop Note column)
out_pub = os.path.join(TABLES_DIR, "extended_summary.csv")
df_ext.drop(columns=['Note']).to_csv(out_pub, index=False)
print(f"  Saved: {out_pub}  (publication-ready, no Note column)")

# ============================================================
# 10. RECOMPUTE COMPACT SUMMARY (for consistency)
# ============================================================
print("\n[8] Recomputing compact final_summary.csv ...")
rows_small = []
for name, arr in [('a', all_a), ('b', all_b), ('q', all_q)]:
    for k in range(K):
        x = arr[:, k]
        rows_small.append({
            'Parameter': f'{name}[{k+1}]',
            'Mean': round(x.mean(), 4),
            'SD':   round(x.std(), 4),
            '2.5%': round(np.percentile(x, 2.5), 4),
            '97.5%':round(np.percentile(x, 97.5), 4),
        })
for k in range(K):
    x = all_pi[:, k]
    rows_small.append({
        'Parameter': f'pi[{k+1}]',
        'Mean': round(x.mean(), 4),
        'SD':   round(x.std(), 4),
        '2.5%': round(np.percentile(x, 2.5), 4),
        '97.5%':round(np.percentile(x, 97.5), 4),
    })
df_small = pd.DataFrame(rows_small)
out_small = os.path.join(TABLES_DIR, "final_summary.csv")
df_small.to_csv(out_small, index=False)
print(f"  Saved: {out_small}")

# ============================================================
# 11. SANITY: VERIFY KEY RESULTS
# ============================================================
print("\n" + "=" * 78)
print("SANITY CHECK — key results")
print("=" * 78)
for k in range(K):
    b_m = all_b[:, k].mean()
    b_sd = all_b[:, k].std()
    e_m = np.nanmean(E_Y[:, k])
    md_m = np.nanmean(med_Y[:, k])
    dur_m = durations[:, k].mean()
    dur_md = np.median(durations[:, k])
    print(f"  State {k+1}:")
    print(f"    b          = {b_m:7.2f} ± {b_sd:5.2f} mm")
    print(f"    E[Y|S]     = {e_m:7.2f} mm")
    print(f"    Median[Y]  = {md_m:7.2f} mm")
    print(f"    Duration   = mean {dur_m:6.2f} | median {dur_md:6.2f} months")
    print(f"    pi*        = {stat[:, k].mean():.4f}")

print("\n" + "=" * 78)
print("MINIPATCH DONE ✅")
print("=" * 78)
print(f"Updated files in: {TABLES_DIR}")
print(f"  - extended_summary.csv        (publication-ready)")
print(f"  - extended_summary_clean.csv  (with Note column)")
print(f"  - final_summary.csv           (recomputed)")