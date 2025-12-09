import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from plot import *
from frequency_regression import *
from simplicity_regression import *


def fit_regression_xy(df, x_col, y_col):
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    x_2d = x.reshape(-1, 1)
    model = LinearRegression().fit(x_2d, y)
    r2 = model.score(x_2d, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    r, _ = pearsonr(x, y)
    return model, r, adj_r2


def plot_corr_vs_feature(df, x_col, y_col, model, r, adj_r2, xlabel, ylabel, title_prefix, fig_path):
    plt.figure()
    plt.scatter(df[x_col], df[y_col], alpha=0.2)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}\n r={r:.3f}, adj R²={adj_r2:.3f}, n={len(df)}")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, default="pearson_results_small.csv")
    parser.add_argument("--freq_dict_path", type=str, default="data/owt_model_frequency.pkl")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_prefix", type=str, default="figs/small_pearson")
    parser.add_argument("--corr_col", type=str, default="r_child_surprisal", choices=["r_child_surprisal", "r_child_attention"])
    args = parser.parse_args()

    df_res = pd.read_csv(args.results_csv)

    with open(args.freq_dict_path, "rb") as f:
        freq_dict = pickle.load(f)
    log_freq = compute_log_freq(freq_dict)

    child_interp_aoa = compute_child_interp_aoa(args.wordbank_csv)

    rows = []
    for _, row in df_res.iterrows():
        w = row["word"]
        if w not in log_freq or w not in child_interp_aoa:
            continue
        corr_val = row[args.corr_col]
        if not np.isfinite(corr_val):
            continue
        aoa_val = child_interp_aoa[w]
        if not np.isfinite(aoa_val):
            continue
        rows.append(
            {
                "word": w,
                "corr": float(corr_val),
                "log_freq": float(log_freq[w]),
                "child_aoa": float(aoa_val),
            }
        )

    df = pd.DataFrame(rows)

    model_freq, r_freq, adj_r2_freq = fit_regression_xy(df, "log_freq", "corr")
    fig_path_freq = f"{args.out_prefix}_corr_vs_logfreq_{args.corr_col}.png"
    plot_corr_vs_feature(
        df,
        x_col="log_freq",
        y_col="corr",
        model=model_freq,
        r=r_freq,
        adj_r2=adj_r2_freq,
        xlabel="Word log-frequency",
        ylabel=f"{args.corr_col}",
        title_prefix=f"{args.corr_col} vs log-frequency",
        fig_path=fig_path_freq,
    )

    model_aoa, r_aoa, adj_r2_aoa = fit_regression_xy(df, "child_aoa", "corr")
    fig_path_aoa = f"{args.out_prefix}_corr_vs_childaoa_{args.corr_col}.png"
    plot_corr_vs_feature(
        df,
        x_col="child_aoa",
        y_col="corr",
        model=model_aoa,
        r=r_aoa,
        adj_r2=adj_r2_aoa,
        xlabel="Child AoA (months)",
        ylabel=f"{args.corr_col}",
        title_prefix=f"{args.corr_col} vs child AoA",
        fig_path=fig_path_aoa,
    )


if __name__ == "__main__":
    main()

# python src/experiments/correlation_regression.py --results_csv pearson_results_medium.csv --out_prefix figs/medium_pearson