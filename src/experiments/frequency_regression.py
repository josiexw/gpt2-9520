import argparse 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import pickle
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)   

def compute_log_freq(freq_dict):
    return {w: np.log10(freq) for w, freq in freq_dict.items() if freq > 0}

def compile_df(aoa, log_freq):
    return pd.DataFrame([
        {'word': w, 'aoa': aoa[w], 'log_freq': log_freq[w]}
        for w in aoa.keys() if w in log_freq
    ])

def fit_regression(df):
    x = df["log_freq"].values
    y = df["aoa"].values
    X = x.reshape(-1, 1)
    model = LinearRegression().fit(X, y)

    r2 = model.score(X, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    r, pval = pearsonr(x, y)

    return model, r, adj_r2, pval

def plot_freq_vs_aoa(ax, df, model, r, pval, r2, model_name):
    ax.scatter(df["log_freq"], df["aoa"], alpha=0.3)

    x_line = np.linspace(df["log_freq"].min(), df["log_freq"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line)

    ax.set_xlabel("Word log-frequency")
    ax.set_ylabel("LLM AoA (steps, log10)")
    ax.set_title(f"Word log-freq vs LLM AoA ({model_name})")
    ax.tick_params(axis="both", labelsize=14)

    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}, p = {pval:.3f}, R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=16
    )

def main():
    parser = argparse.ArgumentParser(description="Plot regression between log-frequency and AoA for small/medium.")
    parser.add_argument("--fig_path", type=str, default="figs/log_freq_vs_llm_aoa.png")
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

    df_small = compile_df(aoa_small, log_freq)
    df_medium = compile_df(aoa_medium, log_freq)

    model_small, r_small, adj_r2_small, pval_small = fit_regression(df_small)
    model_medium, r_medium, adj_r2_medium, pval_medium = fit_regression(df_medium)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plot_freq_vs_aoa(axes[0], df_small, model_small, r_small, pval_small, adj_r2_small, "gpt2-small")
    plot_freq_vs_aoa(axes[1], df_medium, model_medium, r_medium, pval_medium, adj_r2_medium, "gpt2-medium")

    fig.tight_layout()
    fig.savefig(args.fig_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()