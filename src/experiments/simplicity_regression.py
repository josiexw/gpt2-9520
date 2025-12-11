import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import *

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_dir", type=str, default="figs_paper")
    parser.add_argument("--max_simple", type=int, default=600)
    parser.add_argument("--baseline_bits", type=float, default=15.6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    steps_small, small_surpr, _ = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, _ = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)
    child_interp_aoa = compute_child_interp_aoa(args.wordbank_csv)
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

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plot_specs = [
        ("gpt2-small", aoa_small_log, axes[0]),
        ("gpt2-medium", aoa_medium_log, axes[1]),
    ]

    for model_name, aoa_log_dict, ax in plot_specs:
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
        
        model, r, pval = fit_regression_xy(x_arr, y_arr)
        if model is None:
            continue

        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.scatter(x_arr, y_arr, alpha=0.3)
        ax.plot(x_line, y_line)
        ax.set_xlabel("Child AoA (months)")
        ax.set_ylabel("LLM AoA (steps, log10)")
        ax.set_title(f"Child vs LLM AoA ({model_name})")
        ax.tick_params(axis="both", labelsize=14)
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}, p = {pval:.3f}, R^2 = {r**2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=16
        )

    fig.tight_layout()
    out_path_aoa = os.path.join(args.out_dir, "child_vs_llm_aoa_small_medium.png")
    fig.savefig(out_path_aoa, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()