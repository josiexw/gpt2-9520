import os
import re
import argparse
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression


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


def parse_step_from_fname(fname: str) -> int:
    m_label = re.search(r"idx\d+_([A-Za-z]+)(\d+)\.pkl$", fname)
    if not m_label:
        raise ValueError(f"Cannot parse label from {fname}")
    step = int(m_label.group(2))
    return step


def load_results_dir(results_dir: str):
    entries = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".pkl"):
            continue
        step = parse_step_from_fname(fname)
        path = os.path.join(results_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)
        word_surprisal = {}
        for w, info in data.items():
            word_surprisal[w] = float(info["avg_surprisal"])
        entries.append((step, word_surprisal))
    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    all_words = set()
    for _, ws in entries:
        all_words.update(ws.keys())
    word_to_surprisal: Dict[str, List[float]] = {}
    for w in all_words:
        s_series = []
        for _, ws in entries:
            s_series.append(float(ws.get(w, math.nan)))
        word_to_surprisal[w] = s_series
    return steps, word_to_surprisal


def compute_log_surprisal(word_to_surprisal: Dict[str, List[float]]) -> Dict[str, float]:
    log_surp: Dict[str, float] = {}
    for w, series in word_to_surprisal.items():
        s = np.array(series, dtype=float)
        mask = np.isfinite(s)
        if not mask.any():
            continue
        val = float(s[mask][-1])
        if val > 0:
            log_surp[w] = float(np.log10(val))
    return log_surp


def compute_log_child_prop(word_to_curve: Dict[str, np.ndarray]) -> Dict[str, float]:
    log_prop: Dict[str, float] = {}
    for w, curve in word_to_curve.items():
        arr = np.array(curve, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            continue
        val = float(arr[mask][-1])
        if val > 0:
            log_prop[w] = float(np.log10(val))
    return log_prop


def compile_df(log_surp: Dict[str, float], log_prop: Dict[str, float]) -> pd.DataFrame:
    words = set(log_surp.keys()) & set(log_prop.keys())
    rows = []
    for w in words:
        rows.append(
            {
                "word": w,
                "log_surprisal": log_surp[w],
                "log_prop": log_prop[w],
            }
        )
    return pd.DataFrame(rows)


def fit_regression(df: pd.DataFrame):
    x = df["log_prop"].values.reshape(-1, 1)
    y = df["log_surprisal"].values
    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)
    n = len(df)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return model, r2, adj_r2


def plot_log_surprisal_vs_log_prop(df: pd.DataFrame, model, adj_r2: float, fig_path: str):
    plt.figure()
    plt.scatter(df["log_prop"], df["log_surprisal"], alpha=0.2)

    x_line = np.linspace(df["log_prop"].min(), df["log_prop"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line)

    plt.xlabel("Word log proportion producing (Wordbank)")
    plt.ylabel("Word log surprisal (bits)")
    plt.title(f"GPT-2 small: adj R²={adj_r2:.3f}")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot regression between log child proportion and log surprisal.")
    parser.add_argument("--results_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--fig_path", type=str, default="figs/log_prop_vs_log_surprisal_small.png")
    args = parser.parse_args()

    _, word_to_curve = load_wordbank_curves(args.wordbank_csv)
    _, word_to_surprisal = load_results_dir(args.results_dir)

    log_surp = compute_log_surprisal(word_to_surprisal)
    log_prop = compute_log_child_prop(word_to_curve)

    df = compile_df(log_surp, log_prop)
    if df.empty:
        raise RuntimeError("No overlapping words with valid log surprisal and log proportions.")

    model, _, adj_r2 = fit_regression(df)
    plot_log_surprisal_vs_log_prop(df, model, adj_r2, args.fig_path)


if __name__ == "__main__":
    main()
