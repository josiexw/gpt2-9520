import argparse 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
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
    parser = argparse.ArgumentParser(description="Plot regression between log-frequency and AoA for small/medium.")
    parser.add_argument("--fig_path", type=str, default="figs_paper/log_freq_vs_llm_aoa.png")
    parser.add_argument("--freq_dict_path", type=str, default="data/owt_model_frequency.pkl")
    parser.add_argument("--aoa_small_path", type=str, default="data/aoa_small.pkl")
    parser.add_argument("--aoa_medium_path", type=str, default="data/aoa_medium.pkl")
    args = parser.parse_args()

    with open(args.freq_dict_path, "rb") as f:
        freq_dict = pickle.load(f)
    with open(args.aoa_small_path, "rb") as f:
        aoa_small = pickle.load(f)
    with open(args.aoa_medium_path, "rb") as f:
        aoa_medium = pickle.load(f)

    log_freq = compute_log_freq(freq_dict)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plot_specs = [
        ("gpt2-small", aoa_small, axes[0]),
        ("gpt2-medium", aoa_medium, axes[1]),
    ]

    for model_name, aoa_dict, ax in plot_specs:
        x_vals = []
        y_vals = []
        for word in aoa_dict:
            if word in log_freq:
                x_vals.append(log_freq[word])
                y_vals.append(aoa_dict[word])
        if not x_vals:
            continue
        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)
        
        model, r, pval = fit_regression_xy(x_arr, y_arr)
        if model is None:
            continue
        
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.scatter(x_arr, y_arr, alpha=0.3)
        ax.plot(x_line, y_line)
        ax.set_xlabel("Word log-frequency")
        ax.set_ylabel("LLM AoA (steps, log10)")
        ax.set_title(f"Word log-freq vs LLM AoA ({model_name})")
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
    fig.savefig(args.fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()