import os
import re
import argparse
import math
import pickle
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def load_wordbank_aoa(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    word_to_aoa = {}
    for _, row in df.iterrows():
        word = str(row["item_definition"]).strip().lower()
        if not word:
            continue
        aoa = math.nan
        for col in aoa_cols_sorted:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if v >= 0.5:
                aoa = float(col)
                break
        if not math.isnan(aoa):
            word_to_aoa[word] = aoa
    return word_to_aoa


def load_wordbank_curves(csv_path: str) -> Tuple[List[int], Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    months = [int(c) for c in aoa_cols_sorted]
    word_to_curve: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        word = str(row["item_definition"]).strip().lower()
        if not word:
            continue
        vals = []
        for col in aoa_cols_sorted:
            v = row[col]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(math.nan)
        word_to_curve[word] = np.array(vals, dtype=float)
    return months, word_to_curve


def parse_step_from_fname(fname: str) -> Tuple[int, int, str]:
    m_idx = re.search(r"idx(\d+)", fname)
    if not m_idx:
        raise ValueError(f"Cannot parse checkpoint idx from {fname}")
    idx = int(m_idx.group(1))
    m_label = re.search(r"idx\d+_([A-Za-z]+)(\d+)\.pkl$", fname)
    if not m_label:
        raise ValueError(f"Cannot parse label from {fname}")
    label_type = m_label.group(1)
    step = int(m_label.group(2))
    return step, idx, label_type


def load_results_dir(results_dir: str) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]], str]:
    entries = []
    label_type_global = None
    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        step, idx, label_type = parse_step_from_fname(fname)
        if label_type_global is None:
            label_type_global = label_type
        path = os.path.join(results_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)
        word_surprisal = {}
        word_act = {}
        for w, info in data.items():
            word_surprisal[w] = float(info["avg_surprisal"])
            layer_attn = info.get("avg_layer_attn", None)
            if layer_attn is not None:
                layer_arr = np.array(layer_attn, dtype=float)
                word_act[w] = float(np.nanmean(layer_arr))
        entries.append((step, idx, word_surprisal, word_act))
    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    all_words_surpr: Set[str] = set()
    all_words_act: Set[str] = set()
    for _, _, ws, wa in entries:
        all_words_surpr.update(ws.keys())
        all_words_act.update(wa.keys())
    all_words = all_words_surpr | all_words_act
    word_to_surprisal: Dict[str, List[float]] = {}
    word_to_act: Dict[str, List[float]] = {}
    for w in all_words:
        s_series = []
        a_series = []
        for _, _, ws, wa in entries:
            s_series.append(float(ws.get(w, math.nan)))
            a_series.append(float(wa.get(w, math.nan)))
        word_to_surprisal[w] = s_series
        word_to_act[w] = a_series
    return steps, word_to_surprisal, word_to_act, (label_type_global or "")


def get_simple_ranking(word_aoa: Dict[str, float], available_words: Set[str], max_n: int) -> List[str]:
    items = [(w, aoa) for w, aoa in word_aoa.items() if w in available_words and not math.isnan(aoa)]
    items.sort(key=lambda x: (x[1], x[0]))
    return [w for w, _ in items[:max_n]]


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def build_llm_distributions(word_to_surpr: Dict[str, List[float]], steps: List[int], words: List[str], baseline_bits: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(steps)
    W = len(words)
    S = np.full((T, W), baseline_bits, dtype=float)
    for j, w in enumerate(words):
        s_series = np.array(word_to_surpr.get(w, []), dtype=float)
        if s_series.shape[0] != T:
            if s_series.shape[0] == 0:
                continue
            s_padded = np.full(T, baseline_bits, dtype=float)
            n = min(T, s_series.shape[0])
            s_padded[:n] = np.where(np.isfinite(s_series[:n]), s_series[:n], baseline_bits)
            s_series = s_padded
        else:
            s_series = np.where(np.isfinite(s_series), s_series, baseline_bits)
        S[:, j] = s_series
    logits = -S
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P = P / P.sum(axis=1, keepdims=True)
    return P, np.array(steps, dtype=float)


def build_activation_distributions(word_to_act: Dict[str, List[float]], steps: List[int], words: List[str], baseline_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    T = len(steps)
    W = len(words)
    A = np.full((T, W), baseline_val, dtype=float)
    for j, w in enumerate(words):
        a_series = np.array(word_to_act.get(w, []), dtype=float)
        if a_series.shape[0] != T:
            if a_series.shape[0] == 0:
                continue
            a_padded = np.full(T, baseline_val, dtype=float)
            n = min(T, a_series.shape[0])
            a_padded[:n] = np.nan_to_num(a_series[:n], nan=baseline_val, posinf=baseline_val, neginf=baseline_val)
            a_series = a_padded
        else:
            a_series = np.nan_to_num(a_series, nan=baseline_val, posinf=baseline_val, neginf=baseline_val)
        A[:, j] = a_series
    logits = A - A.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P = P / P.sum(axis=1, keepdims=True)
    return P, np.array(steps, dtype=float)


def build_child_distributions(months: List[int], word_to_curve: Dict[str, np.ndarray], words: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    M = len(months)
    W = len(words)
    C = np.zeros((M, W), dtype=float)
    for j, w in enumerate(words):
        curve = word_to_curve.get(w, None)
        if curve is None or curve.size == 0:
            continue
        if curve.shape[0] < M:
            vals = np.full(M, 0.0, dtype=float)
            n = curve.shape[0]
            vals[:n] = np.nan_to_num(curve[:n], nan=0.0, posinf=0.0, neginf=0.0)
            C[:, j] = vals
        else:
            vals = np.nan_to_num(curve[:M], nan=0.0, posinf=0.0, neginf=0.0)
            C[:, j] = vals
    return C, np.array(months, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_dir", type=str, default="kl_figs")
    parser.add_argument("--out_txt", type=str, default="kl_child_vs_checkpoint_results.txt")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--baseline_bits", type=float, default=14.9)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)

    steps_small, small_surpr, small_act, label_type_small = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, medium_act, label_type_medium = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)
    words_for_kl = [w for w in simple_ranking if w in word_to_curve]

    P_small_full, steps_small_arr_full = build_llm_distributions(small_surpr, steps_small, words_for_kl, args.baseline_bits)
    P_medium_full, steps_medium_arr_full = build_llm_distributions(medium_surpr, steps_medium, words_for_kl, args.baseline_bits)
    A_small_full, _ = build_activation_distributions(small_act, steps_small, words_for_kl)
    A_medium_full, _ = build_activation_distributions(medium_act, steps_medium, words_for_kl)
    C_child, months_arr = build_child_distributions(months, word_to_curve, words_for_kl)

    P_small = P_small_full[1:]
    steps_small_arr = steps_small_arr_full[1:]
    P_medium = P_medium_full[1:]
    steps_medium_arr = steps_medium_arr_full[1:]
    A_small = A_small_full[1:]
    A_medium = A_medium_full[1:]

    M = C_child.shape[0]
    T_small = P_small.shape[0]
    T_medium = P_medium.shape[0]

    KL_small = np.full((M, T_small), np.nan, dtype=float)
    KL_medium = np.full((M, T_medium), np.nan, dtype=float)

    for mi in range(M):
        child_vec = C_child[mi]
        if not np.isfinite(child_vec).any() or child_vec.sum() <= 0:
            continue
        for ti in range(T_small):
            KL_small[mi, ti] = kl_divergence(child_vec, P_small[ti])
        for ti in range(T_medium):
            KL_medium[mi, ti] = kl_divergence(child_vec, P_medium[ti])

    KL_sa_small = np.full((T_small, T_small), np.nan, dtype=float)
    KL_sa_medium = np.full((T_medium, T_medium), np.nan, dtype=float)

    for i in range(T_small):
        for j in range(T_small):
            d1 = kl_divergence(P_small[i], A_small[j])
            d2 = kl_divergence(A_small[j], P_small[i])
            KL_sa_small[i, j] = 0.5 * (d1 + d2)

    for i in range(T_medium):
        for j in range(T_medium):
            d1 = kl_divergence(P_medium[i], A_medium[j])
            d2 = kl_divergence(A_medium[j], P_medium[i])
            KL_sa_medium[i, j] = 0.5 * (d1 + d2)

    with open(args.out_txt, "w") as fout:
        fout.write(f"Words used for KL analyses (<= {args.max_simple}): {len(words_for_kl)}\n\n")

        min_small = np.nanmin(KL_small)
        mi_s, ti_s = np.where(KL_small == min_small)
        if mi_s.size > 0:
            fout.write("Small model minimum KL(child||LLM):\n")
            fout.write(f"  KL={min_small:.4f}, month_index={mi_s[0]}, month={months_arr[mi_s[0]]}, checkpoint_index={ti_s[0]+1}, step={steps_small_arr[ti_s[0]]}\n\n")

        min_medium = np.nanmin(KL_medium)
        mi_m, ti_m = np.where(KL_medium == min_medium)
        if mi_m.size > 0:
            fout.write("Medium model minimum KL(child||LLM):\n")
            fout.write(f"  KL={min_medium:.4f}, month_index={mi_m[0]}, month={months_arr[mi_m[0]]}, checkpoint_index={ti_m[0]+1}, step={steps_medium_arr[ti_m[0]]}\n\n")

        fout.write("Per-month argmin KL for small model:\n")
        for mi in range(M):
            row = KL_small[mi]
            if not np.isfinite(row).any():
                continue
            ti = int(np.nanargmin(row))
            fout.write(f"  month={months_arr[mi]:.1f}, best_checkpoint_index={ti+1}, step={steps_small_arr[ti]}, KL={row[ti]:.4f}\n")
        fout.write("\nPer-month argmin KL for medium model:\n")
        for mi in range(M):
            row = KL_medium[mi]
            if not np.isfinite(row).any():
                continue
            ti = int(np.nanargmin(row))
            fout.write(f"  month={months_arr[mi]:.1f}, best_checkpoint_index={ti+1}, step={steps_medium_arr[ti]}, KL={row[ti]:.4f}\n")

        fout.write("\nSymmetric KL between surprisal- and activation-based distributions (small):\n")
        fout.write(f"  min={np.nanmin(KL_sa_small):.4f}, max={np.nanmax(KL_sa_small):.4f}\n")
        fout.write("Symmetric KL between surprisal- and activation-based distributions (medium):\n")
        fout.write(f"  min={np.nanmin(KL_sa_medium):.4f}, max={np.nanmax(KL_sa_medium):.4f}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(KL_small, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("KL(child month || GPT2-small checkpoint)")
    ax.set_xlabel("Checkpoint index (starting at 1)")
    ax.set_ylabel("Month index")
    fig.colorbar(im, ax=ax, label="KL divergence")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "kl_child_vs_small_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(KL_medium, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("KL(child month || GPT2-medium checkpoint)")
    ax.set_xlabel("Checkpoint index (starting at 1)")
    ax.set_ylabel("Month index")
    fig.colorbar(im, ax=ax, label="KL divergence")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "kl_child_vs_medium_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(KL_sa_small, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("Symmetric KL: surprisal vs activation (GPT2-small)")
    ax.set_xlabel("Activation checkpoint index")
    ax.set_ylabel("Surprisal checkpoint index")
    fig.colorbar(im, ax=ax, label="Symmetric KL")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "kl_suprisal_vs_activation_small.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(KL_sa_medium, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("Symmetric KL: surprisal vs activation (GPT2-medium)")
    ax.set_xlabel("Activation checkpoint index")
    ax.set_ylabel("Surprisal checkpoint index")
    fig.colorbar(im, ax=ax, label="Symmetric KL")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "kl_suprisal_vs_activation_medium.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
