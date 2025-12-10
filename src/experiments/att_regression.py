import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from plot import *
from frequency_regression import *
from simplicity_regression import *

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def last_finite(arr):
    arr = np.asarray(arr, dtype=float)
    for v in arr[::-1]:
        if np.isfinite(v):
            return float(v)
    return math.nan


def get_last_checkpoint_attention_layer12(word_to_act, layer_idx=11):
    """
    Get attention from layer 12 (index 11) at the last checkpoint.
    Assumes word_to_act structure has layer-specific data.
    """
    att_last = {}
    for w, series in word_to_act.items():
        # If series is a 2D array (layers x checkpoints), get layer 12
        if isinstance(series, (list, np.ndarray)):
            series_arr = np.asarray(series, dtype=float)
            if series_arr.ndim == 2 and series_arr.shape[0] > layer_idx:
                # Get layer 12 (index 11) data
                layer_series = series_arr[layer_idx, :]
                val = last_finite(layer_series)
            else:
                # Fallback if structure is different
                val = last_finite(series_arr.flatten())
        else:
            val = last_finite(series)
            
        if np.isfinite(val):
            att_last[w] = float(val)
    return att_last


def fit_regression_xy(x_arr, y_arr):
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None, None, None, None

    X = x_arr.reshape(-1, 1)
    model = LinearRegression().fit(X, y_arr)
    r2 = model.score(X, y_arr)
    n = len(x_arr)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    r, pval = pearsonr(x_arr, y_arr)
    return model, r, adj_r2, pval


def plot_scatter_with_regression(ax, x_arr, y_arr, xlabel, ylabel, title, r, pval, r2):
    ax.scatter(x_arr, y_arr, alpha=0.3)
    x_line = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), 200)
    X = x_arr.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_arr)
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=14)
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}, p = {pval:.3f}, R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=16,
    )


def main():
    parser = argparse.ArgumentParser(description="Layer 12 attention (last checkpoint) vs child AoA and word frequency.")
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--freq_dict_path", type=str, default="data/owt_model_frequency.pkl")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--max_simple", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    child_interp_aoa = compute_child_interp_aoa(args.wordbank_csv)

    _, _, small_act = load_results_dir(args.small_dir)
    _, _, medium_act = load_results_dir(args.medium_dir)

    words_small = set(small_act.keys())
    words_medium = set(medium_act.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    att_small_last = get_last_checkpoint_attention_layer12(small_act, layer_idx=11)
    att_medium_last = get_last_checkpoint_attention_layer12(medium_act, layer_idx=11)

    with open(args.freq_dict_path, "rb") as f:
        freq_dict = pickle.load(f)
    log_freq = compute_log_freq(freq_dict)

    fig_aoa, axes_aoa = plt.subplots(1, 2, figsize=(12, 5))
    if not isinstance(axes_aoa, np.ndarray):
        axes_aoa = np.array([axes_aoa])

    specs_aoa = [
        ("gpt2-small", att_small_last, axes_aoa[0]),
        ("gpt2-medium", att_medium_last, axes_aoa[1]),
    ]

    plotted_aoa = 0
    for model_name, att_dict, ax in specs_aoa:
        x_vals = []
        y_vals = []
        for w in simple_ranking:
            if w not in child_interp_aoa or w not in att_dict:
                continue
            x = child_interp_aoa[w]
            y = att_dict[w]
            if not (np.isfinite(x) and np.isfinite(y)) or x <= 0:
                continue
            x_vals.append(x)
            y_vals.append(y)
        if not x_vals:
            continue
        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)
        model, r, adj_r2, pval = fit_regression_xy(x_arr, y_arr)
        if model is None:
            continue

        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.scatter(x_arr, y_arr, alpha=0.3)
        ax.plot(x_line, y_line)
        ax.set_xlabel("Child AoA (months)")
        ax.set_ylabel("Layer 12 attention (last checkpoint)")
        ax.set_title(f"Child AoA vs attention ({model_name})")
        ax.tick_params(axis="both", labelsize=14)
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}, p = {pval:.3f}, R$^2$ = {adj_r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=16,
        )
        plotted_aoa += 1

    if plotted_aoa > 0:
        fig_aoa.tight_layout()
        out_path_aoa = os.path.join(args.out_dir, "child_aoa_vs_attention_layer12_small_medium.png")
        fig_aoa.savefig(out_path_aoa, bbox_inches="tight")
    plt.close(fig_aoa)

    fig_freq, axes_freq = plt.subplots(1, 2, figsize=(12, 5))
    if not isinstance(axes_freq, np.ndarray):
        axes_freq = np.array([axes_freq])

    specs_freq = [
        ("gpt2-small", att_small_last, axes_freq[0]),
        ("gpt2-medium", att_medium_last, axes_freq[1]),
    ]

    plotted_freq = 0
    for model_name, att_dict, ax in specs_freq:
        x_vals = []
        y_vals = []
        for w, att in att_dict.items():
            if w not in log_freq:
                continue
            x = log_freq[w]
            y = att
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            x_vals.append(x)
            y_vals.append(y)
        if not x_vals:
            continue
        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)
        model, r, adj_r2, pval = fit_regression_xy(x_arr, y_arr)
        if model is None:
            continue

        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.scatter(x_arr, y_arr, alpha=0.3)
        ax.plot(x_line, y_line)
        ax.set_xlabel("Word log-frequency")
        ax.set_ylabel("Layer 12 attention (last checkpoint)")
        ax.set_title(f"Word log-freq vs attention ({model_name})")
        ax.tick_params(axis="both", labelsize=14)
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}, p <= 0.001, R$^2$ = {adj_r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=16,
        )
        plotted_freq += 1

    if plotted_freq > 0:
        fig_freq.tight_layout()
        out_path_freq = os.path.join(args.out_dir, "log_freq_vs_attention_layer12_small_medium.png")
        fig_freq.savefig(out_path_freq, bbox_inches="tight")
    plt.close(fig_freq)


if __name__ == "__main__":
    main()