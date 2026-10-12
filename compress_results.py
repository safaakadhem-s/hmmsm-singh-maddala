# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:21:45 2026

@author: ROYAL2
"""

# ============================================================
# compress_results.py — Reduce file size from 470 MB to ~60 MB
# ============================================================
import os
import numpy as np

BASE = r"D:\فكرة بحث 2027"
src = os.path.join(BASE, "hmm_sm_T5000_2chains.npz")
dst = os.path.join(BASE, "hmm_sm_T5000_2chains_compressed.npz")

print("Loading 470 MB file... (may take 30 sec)")
data = dict(np.load(src, allow_pickle=True))

print("Compressing...")
compressed = {}
for key, val in data.items():
    if key == "S_samples":
        # Convert int64 → int8 (3 states fit in 1 byte)
        compressed[key] = val.astype(np.int8)
        print(f"  {key}: int64 → int8  ({val.nbytes/1e6:.0f} MB → {val.nbytes/8/1e6:.0f} MB)")
    else:
        compressed[key] = val

print("Saving compressed file...")
np.savez_compressed(dst, **compressed)

old_size = os.path.getsize(src) / 1e6
new_size = os.path.getsize(dst) / 1e6
print(f"\n✅ Original:    {old_size:.1f} MB")
print(f"✅ Compressed:  {new_size:.1f} MB")
print(f"✅ Reduction:   {old_size/new_size:.1f}×")
print(f"\nNew file: {dst}")

# Replace original with compressed
os.remove(src)
os.rename(dst, src)
print(f"\n✅ Replaced: {src}")