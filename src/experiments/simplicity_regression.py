import os
import argparse
import math
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plot import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--baseline_bits", type=float, default=15.6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)

    steps_small, small_surpr, _ = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, _ = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    months_arr = np.array(months, dtype=float)
    child_interp_aoa: Dict[str, float] = {}
    for w in simple_ranking:
        if w not in word_to_curve:
            continue
        curve = np.array(word_to_curve[w], dtype=float)
        if not np.isfinite(curve).any():
            continue
        mask = np.isfinite(curve)
        m = months_arr[mask]
        y = curve[mask]
        if m.size < 2:
            continue
        idx_cross = None
        for j in range(1, y.size):
            y_prev = y[j - 1]
            y_curr = y[j]
            if not (np.isfinite(y_prev) and np.isfinite(y_curr)):
                continue
            if (y_prev - 0.5) * (y_curr - 0.5) <= 0:
                idx_cross = j
                break
        if idx_cross is None:
            continue
        m1, m2 = m[idx_cross - 1], m[idx_cross]
        y1, y2 = y[idx_cross - 1], y[idx_cross]
        if not (np.isfinite(y1) and np.isfinite(y2)) or m2 == m1:
            continue
        if y2 == y1:
            aoa_month = m2
        else:
            aoa_month = m1 + (0.5 - y1) * (m2 - m1) / (y2 - y1)
        if np.isfinite(aoa_month) and aoa_month > 0:
            child_interp_aoa[w] = float(aoa_month)

    words_for_aoa = [w for w in simple_ranking if w in child_interp_aoa]

    aoa_small_log = compute_llm_aoa_steps(
        word_to_series=small_surpr,
        steps=steps_small,
        baseline_bits=args.baseline_bits,
        words=words_for_aoa,
    )
    aoa_medium_log = compute_llm_aoa_steps(
        word_to_series=medium_surpr,
        steps=steps_medium,
        baseline_bits=args.baseline_bits,
        words=words_for_aoa,
    )

    for model_name, aoa_log_dict, suffix in [
        ("gpt2-small", aoa_small_log, "small"),
        ("gpt2-medium", aoa_medium_log, "medium"),
    ]:
        x_vals = []
        y_vals = []
        for w in words_for_aoa:
            if w not in child_interp_aoa or w not in aoa_log_dict:
                continue
            child_a = child_interp_aoa[w]
            llm_a = aoa_log_dict[w]
            if not (np.isfinite(child_a) and np.isfinite(llm_a)) or child_a <= 0:
                continue
            x_vals.append(child_a)
            y_vals.append(float(llm_a))
        if not x_vals:
            continue
        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)
        if x_arr.size < 2:
            continue
        r, _ = pearsonr(x_arr, y_arr)
        r2 = r ** 2
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_arr, y_arr, alpha=0.2)
        ax.plot(x_line, y_line)
        ax.set_xlabel("Child AoA (months)")
        ax.set_ylabel("log10 LLM AoA (steps)")
        ax.set_title(f"Child vs LLM AoA ({model_name})")
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}, R$^2$ = {r2:.3f}, n = {x_arr.size}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        out_path_aoa = os.path.join(args.out_dir, f"log_child_vs_llm_aoa_{suffix}.png")
        fig.savefig(out_path_aoa, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
