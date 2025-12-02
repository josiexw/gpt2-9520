import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

def compute_log_freq(freq_dict):
    return {w: np.log10(freq) for w, freq in freq_dict.items() if freq > 0}

def compile_df(aoa, log_freq):
    return pd.DataFrame([
        {"word": w, "aoa": aoa[w], "log_freq": log_freq[w]}
        for w in aoa.keys() if w in log_freq
    ])

def fit_regression(df):
    x = df["log_freq"].values.reshape(-1, 1)
    y = df["aoa"].values

    model = LinearRegression().fit(x, y)

    r2 = model.score(x, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return model, r2, adj_r2

def plot_freq_vs_aoa(df, model, adj_r2, fig_path):
    plt.scatter(df['log_freq'], df['aoa'], alpha=0.2)

    x_line = np.linspace(df['log_freq'].min(), df['log_freq'].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.plot(x_line, y_line)

    plt.xlabel('Word log-frequency')
    plt.ylabel('GPT-2 AoA (steps, log10)')
    plt.title(f'GPT-2 Small: R2={adj_r2:.3f}')
    plt.savefig(fig_path) 

def compile_joint_aoa_df(gpt_aoa, child_aoa):
    rows = []
    for w, g_aoa in gpt_aoa.items():
        if w in child_aoa and child_aoa[w] > 0:
            c_aoa = child_aoa[w]
            rows.append({
                "word": w,
                "gpt_aoa": g_aoa,
                "child_aoa": c_aoa,
                "log_child_aoa": np.log10(c_aoa),
            })
    return pd.DataFrame(rows)


def fit_regression_child(df):
    x = df["log_child_aoa"].values.reshape(-1, 1)
    y = df["gpt_aoa"].values

    model = LinearRegression().fit(x, y)

    r2 = model.score(x, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return model, r2, adj_r2


def build_surprisal_dict_from_results(results_dir, value_key="avg_surprisal_per_token"):
    pattern = f"{results_dir}/results_ckpt_idx*.pkl"
    files = sorted(glob.glob(pattern))
    if not files:
        return {}

    last_file = files[-1]
    with open(last_file, "rb") as f:
        out = pickle.load(f)

    surprisal_dict = {}
    for w, d in out.items():
        v = d.get(value_key, float("nan"))
        if np.isfinite(v) and v > 0:
            surprisal_dict[w] = float(v)
    return surprisal_dict


def compile_prop_surprisal_df(freq_dict, surprisal_dict):
    total = sum(v for v in freq_dict.values() if v > 0)
    if total <= 0:
        return pd.DataFrame([])

    rows = []
    for w, freq in freq_dict.items():
        if freq <= 0:
            continue
        if w not in surprisal_dict:
            continue
        s = surprisal_dict[w]
        if not np.isfinite(s) or s <= 0:
            continue
        p = freq / total
        rows.append({
            "word": w,
            "log_prop": np.log10(p),
            "log_surprisal": np.log10(s),
        })
    return pd.DataFrame(rows)


def fit_regression_surprisal(df):
    x = df["log_prop"].values.reshape(-1, 1)
    y = df["log_surprisal"].values

    model = LinearRegression().fit(x, y)

    r2 = model.score(x, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return model, r2, adj_r2


def plot_logprop_vs_logsurprisal(df, model, adj_r2, fig_path):
    plt.figure()
    plt.scatter(df["log_prop"], df["log_surprisal"], alpha=0.2)

    x_line = np.linspace(df["log_prop"].min(), df["log_prop"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.plot(x_line, y_line)
    plt.xlabel("Word log-proportion")
    plt.ylabel("Log surprisal")
    plt.title(f"GPT-2 Small: log proportion vs log surprisal (adj R²={adj_r2:.3f})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot regression between log-frequency, AoA, and surprisal.")
    parser.add_argument("--fig_path", type=str, required=True)
    parser.add_argument("--freq_dict_path", type=str, required=True)
    parser.add_argument("--aoa_dict_path", type=str, required=True)
    parser.add_argument("--child_aoa_dict_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--surprisal_fig_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.freq_dict_path, "rb") as f:
        freq_dict = pickle.load(f)

    with open(args.aoa_dict_path, "rb") as g:
        gpt_aoa_dict = pickle.load(g)

    with open(args.child_aoa_dict_path, "rb") as h:
        child_aoa_dict = pickle.load(h)

    log_freq = compute_log_freq(freq_dict)
    compiled_df = compile_df(gpt_aoa_dict, log_freq)
    model_freq, _, adj_r2_freq = fit_regression(compiled_df)
    plot_freq_vs_aoa(compiled_df, model_freq, adj_r2_freq, args.fig_path)

    if args.results_dir is not None and args.surprisal_fig_path is not None:
        surprisal_dict = build_surprisal_dict_from_results(args.results_dir, value_key="avg_surprisal_per_token")
        if surprisal_dict:
            prop_df = compile_prop_surprisal_df(freq_dict, surprisal_dict)
            if len(prop_df) > 1:
                model_s, _, adj_r2_s = fit_regression_surprisal(prop_df)
                plot_logprop_vs_logsurprisal(prop_df, model_s, adj_r2_s, args.surprisal_fig_path)


if __name__ == "__main__":
    main()