# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:25:54 2026

@author: ROYAL2
"""

# ============================================================
# output_helper.py
# Utility library for saving figures, tables, and data
# with automatic folder structure
# ============================================================
import os
import numpy as np
import pandas as pd

# ============================================================
# BASE DIRECTORY
# ============================================================
BASE = r"D:\فكرة بحث 2027"


# ============================================================
# FOLDER MANAGEMENT
# ============================================================
def get_folder(name):
    """
    Return the folder path, creating it and its subfolders if needed.

    Structure created:
        BASE/name/
        BASE/name/figures/
        BASE/name/tables/
        BASE/name/data/
    """
    folder = os.path.join(BASE, name)
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    os.makedirs(os.path.join(folder, "tables"), exist_ok=True)
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    return folder


# ============================================================
# FIGURE SAVING
# ============================================================
def save_figure(fig, folder_name, filename, dpi=120):
    """
    Save a matplotlib figure to BASE/folder_name/figures/filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    folder_name : str
    filename : str (e.g., "plot.png")
    dpi : int

    Returns
    -------
    path : str
    """
    path = os.path.join(BASE, folder_name, "figures", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  ✅ Saved figure: {path}")
    return path


# ============================================================
# TABLE SAVING
# ============================================================
def save_table(df, folder_name, filename, also_tex=False):
    """
    Save a pandas DataFrame as CSV (and optionally LaTeX).

    Parameters
    ----------
    df : pandas.DataFrame
    folder_name : str
    filename : str (without extension)
    also_tex : bool — also save as .tex

    Returns
    -------
    csv_path : str
    """
    # CSV
    csv_path = os.path.join(BASE, folder_name, "tables", filename + ".csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  ✅ Saved table (CSV): {csv_path}")

    # LaTeX
    if also_tex:
        tex_path = os.path.join(BASE, folder_name, "tables", filename + ".tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=False,
                                float_format="%.4f"))
        print(f"  ✅ Saved table (TeX): {tex_path}")

    return csv_path


# ============================================================
# ARRAY SAVING
# ============================================================
def save_array(arr, folder_name, filename):
    """
    Save a numpy array as .npy.
    """
    path = os.path.join(BASE, folder_name, "data", filename + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    print(f"  ✅ Saved array: {path}")
    return path


def save_npz(dict_of_arrays, folder_name, filename):
    """
    Save multiple arrays as .npz (compressed numpy archive).

    Parameters
    ----------
    dict_of_arrays : dict {name: array}
    folder_name : str
    filename : str (without extension)
    """
    path = os.path.join(BASE, folder_name, "data", filename + ".npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Clean dict: convert lists to arrays
    clean = {}
    for k, v in dict_of_arrays.items():
        try:
            clean[k] = np.asarray(v)
        except Exception:
            clean[k] = np.array([v], dtype=object)
    np.savez(path, **clean)
    print(f"  ✅ Saved npz: {path}")
    return path


# ============================================================
# SUMMARY SAVING
# ============================================================
def save_summary(folder_name, text):
    """
    Save a summary text file at BASE/folder_name/summary.txt.
    """
    path = os.path.join(BASE, folder_name, "summary.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✅ Saved summary: {path}")
    return path


# ============================================================
# DATA LOADING
# ============================================================
def load_idata():
    """
    Load the main HMM-SM posterior samples.

    Searches in multiple candidate locations:
        1. BASE/hmm_sm_T5000_2chains.npz
        2. BASE/hmm_sm_T5000_2chains/hmm_sm_T5000_2chains.npz
        3. ./hmm_sm_T5000_2chains.npz (current dir)

    Returns
    -------
    data : dict with keys:
        Y, S_true, S_post, a_samples, b_samples, q_samples,
        pi_samples, Gamma_samples,
        a_true, b_true, q_true, pi_true, Gamma_true
    """
    candidates = [
        os.path.join(BASE, "hmm_sm_T5000_2chains.npz"),
        os.path.join(BASE, "hmm_sm_T5000_2chains",
                     "hmm_sm_T5000_2chains.npz"),
        os.path.join(BASE, "results", "hmm_sm_T5000_2chains.npz"),
        "hmm_sm_T5000_2chains.npz",
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  ✅ Loaded HMM-SM posterior from: {p}")
            return dict(np.load(p, allow_pickle=True))

    raise FileNotFoundError(
        f"❌ Could not find hmm_sm_T5000_2chains.npz.\n"
        f"   Searched in:\n" +
        "\n".join(f"     - {c}" for c in candidates) +
        f"\n\n   Please run the HMM-SM estimation first "
        f"(hmm_sm_T5000_2chains.py) and save the results."
    )


# ============================================================
# SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("output_helper.py — Self test")
    print("=" * 60)
    print(f"BASE directory: {BASE}")

    # Test folder creation
    test_folder = "_test_output_helper"
    get_folder(test_folder)
    print(f"  ✅ Created test folder: {test_folder}")

    # Test figure saving
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test")
        save_figure(fig, test_folder, "test_plot.png")
        plt.close(fig)
    except ImportError:
        print("  ⚠️  matplotlib not available — skipping figure test")

    # Test table saving
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    save_table(df, test_folder, "test_table", also_tex=True)

    # Test array saving
    save_npz({"arr1": np.array([1, 2, 3]),
              "arr2": np.array([4.0, 5.0])},
             test_folder, "test_data")

    # Test summary
    save_summary(test_folder, "This is a test summary.")

    # Clean up
    import shutil
    shutil.rmtree(os.path.join(BASE, test_folder))
    print(f"\n  ✅ Cleaned up test folder")

    print("\n" + "=" * 60)
    print("✅ output_helper.py works correctly")
    print("=" * 60)