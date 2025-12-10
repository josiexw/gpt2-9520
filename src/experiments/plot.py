import os
import re
import argparse
import math
import statistics
import pickle
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
    }
)


def normalize_cdi_label(w: str) -> str:
    w = w.lower().strip()
    w = re.sub(r"\s*\(.*?\)\s*", "", w)
    w = re.sub(r"\s+", " ", w)
    return w


def load_wordbank_aoa(csv_path: str):
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    word_to_aoa = {}
    for _, row in df.iterrows():
        word = normalize_cdi_label(str(row["item_definition"]))
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
            if word in word_to_aoa:
                word_to_aoa[word] = statistics.mean([word_to_aoa[word], aoa])
            else:
                word_to_aoa[word] = aoa
    return word_to_aoa


def load_wordbank_curves(csv_path: str):
    df = pd.read_csv(csv_path)
    aoa_cols = [c for c in df.columns if c.isdigit()]
    aoa_cols_sorted = sorted(aoa_cols, key=lambda x: int(x))
    months = [int(c) for c in aoa_cols_sorted]
    word_to_curve: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        word = normalize_cdi_label(str(row["item_definition"]))
        if not word:
            continue
        vals = []
        for col in aoa_cols_sorted:
            v = row[col]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(math.nan)
        vals_arr = np.array(vals, dtype=float)
        if word in word_to_curve:
            prev = word_to_curve[word]
            stacked = np.vstack([prev, vals_arr])
            word_to_curve[word] = np.nanmean(stacked, axis=0)
        else:
            word_to_curve[word] = vals_arr
    return months, word_to_curve


def parse_step_from_fname(fname: str):
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
        word_act = {}
        for w, info in data.items():
            w_norm = normalize_cdi_label(w)
            word_surprisal[w_norm] = float(info["avg_surprisal"])
            layer_attn = info.get("avg_layer_attn", None)
            if layer_attn is not None:
                layer_arr = np.array(layer_attn, dtype=float)
                word_act[w_norm] = float(np.nanmean(layer_arr))
        entries.append((step, word_surprisal, word_act))
    entries.sort(key=lambda x: x[0])
    steps = [e[0] for e in entries]
    all_words_surprisal: Set[str] = set()
    all_words_act: Set[str] = set()
    for _, ws, wa in entries:
        all_words_surprisal.update(ws.keys())
        all_words_act.update(wa.keys())
    all_words = all_words_surprisal | all_words_act
    word_to_surprisal: Dict[str, List[float]] = {}
    word_to_act: Dict[str, List[float]] = {}
    for w in all_words:
        s_series = []
        a_series = []
        for _, ws, wa in entries:
            s_series.append(float(ws.get(w, math.nan)))
            a_series.append(float(wa.get(w, math.nan)))
        word_to_surprisal[w] = s_series
        word_to_act[w] = a_series
    return steps, word_to_surprisal, word_to_act


def get_simple_ranking(word_aoa: Dict[str, float], available_words: set[str], max_n: int):
    items = [(w, aoa) for w, aoa in word_aoa.items() if w in available_words and not math.isnan(aoa)]
    items.sort(key=lambda x: (x[1], x[0]))
    return [w for w, _ in items[:max_n]]


def compute_avg_series(word_to_series: Dict[str, List[float]], words: List[str]):
    if not words:
        return np.array([], dtype=float)
    arr = []
    for w in words:
        arr.append(np.array(word_to_series[w], dtype=float))
    arr = np.stack(arr, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def compute_thresholds_per_word(
    word_to_series: Dict[str, List[float]],
    baseline_bits: float,
    words: List[str],
    steps: List[int],
):
    aoa_log10 = compute_llm_aoa_steps(
        word_to_series=word_to_series,
        steps=steps,
        baseline_bits=baseline_bits,
        words=words,
    )
    aoa_steps = {}
    for w, x_star in aoa_log10.items():
        aoa_steps[w] = float(10.0 ** x_star)
    return aoa_steps


def logistic4(x, L, k, x0, b):
    x = np.asarray(x, dtype=float)
    z = k * (x - x0)
    z = np.clip(z, -60, 60)
    return L / (1.0 + np.exp(z)) + b


def compute_llm_aoa_steps(
    word_to_series: Dict[str, List[float]],
    steps: List[int],
    baseline_bits: float,
    words: List[str],
):
    aoa_log10: Dict[str, float] = {}
    if not words:
        return aoa_log10

    step_arr = np.array(steps, dtype=float)
    safe_steps = np.where(step_arr > 0, step_arr, 1.0)
    log_steps = np.log10(safe_steps)
    mask_log = np.isfinite(log_steps)

    for w in words:
        s = np.array(word_to_series[w], dtype=float)
        mask = mask_log & np.isfinite(s)
        x = log_steps[mask]
        y = s[mask]
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max == y_min:
            continue

        thr = 0.5 * (baseline_bits + y_min)

        L0 = max(y_max - y_min, 1e-3)
        b0 = y_min
        x0_0 = float(np.median(x))
        k0 = 1.0

        x_star = None

        try:
            popt, _ = curve_fit(
                logistic4,
                x,
                y,
                p0=[L0, k0, x0_0, b0],
                maxfev=10000,
            )
            L, k, x0, b = popt

            if L * k == 0:
                raise RuntimeError("flat logistic fit")

            f_lo = float(logistic4(x[0], *popt))
            f_hi = float(logistic4(x[-1], *popt))

            lo_val = min(f_lo, f_hi)
            hi_val = max(f_lo, f_hi)

            if not (lo_val <= thr <= hi_val):
                raise RuntimeError("threshold outside fit range")

            lo_x = x[0]
            hi_x = x[-1]

            if f_lo <= f_hi:
                for _ in range(60):
                    mid_x = 0.5 * (lo_x + hi_x)
                    val = float(logistic4(mid_x, *popt))
                    if val < thr:
                        lo_x = mid_x
                    else:
                        hi_x = mid_x
            else:
                for _ in range(60):
                    mid_x = 0.5 * (lo_x + hi_x)
                    val = float(logistic4(mid_x, *popt))
                    if val > thr:
                        lo_x = mid_x
                    else:
                        hi_x = mid_x

            x_star = 0.5 * (lo_x + hi_x)

        except Exception:
            idx_val = None

            for j in range(1, len(y)):
                y_prev, y_curr = y[j - 1], y[j]
                if not (np.isfinite(y_prev) and np.isfinite(y_curr)):
                    continue

                if (y_prev - thr) * (y_curr - thr) <= 0:
                    idx_val = j
                    break

            if idx_val is None:
                idx_val = int(np.argmin(np.abs(y - thr)))
            x_star = float(x[idx_val])

        if x_star is not None and np.isfinite(x_star):
            aoa_log10[w] = x_star

    return aoa_log10


def normalize_x(xs: np.ndarray):
    xs = np.array(xs, dtype=float)
    if xs.size == 0:
        return xs
    denom = xs[-1] - xs[0]
    if denom <= 0:
        return np.zeros_like(xs)
    return (xs - xs[0]) / denom


def crop_with_threshold(s, steps_arr, baseline_bits, margin_idx):
    s = np.array(s, dtype=float)
    if not np.isfinite(s).any():
        return None, None, None
    s_min = float(np.nanmin(s))
    thr = 0.5 * (baseline_bits + s_min)
    idx_cross = None
    for j, v in enumerate(s):
        if np.isfinite(v) and v <= thr:
            idx_cross = j
            break
    if idx_cross is None:
        idx_cross = len(s) - 1
    end_idx = min(idx_cross + margin_idx, len(s) - 1)
    s_crop = s[: end_idx + 1]
    steps_crop = steps_arr[: end_idx + 1]
    return s_crop, steps_crop, thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--ks", type=str, default="10,100,500")
    parser.add_argument("--baseline_bits", type=float, default=15.6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)

    steps_small, small_surpr, small_act = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, medium_act = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    # Surprisal for top-k simple words over timesteps
    for k in ks:
        words_k = simple_ranking[:k]
        avg_small = compute_avg_series(small_surpr, words_k)
        avg_medium = compute_avg_series(medium_surpr, words_k)

        def thr_mean(d: Dict[str, float]) -> float:
            if not d:
                return math.nan
            vals = np.array(list(d.values()), dtype=float)
            return float(np.mean(vals))

        thr_small_dict = compute_thresholds_per_word(small_surpr, args.baseline_bits, words_k, steps_small)
        thr_medium_dict = compute_thresholds_per_word(medium_surpr, args.baseline_bits, words_k, steps_medium)

        small_mean = thr_mean(thr_small_dict)
        med_mean = thr_mean(thr_medium_dict)

        fig = plt.figure()
        plt.plot(steps_small, avg_small, label="gpt2-small")
        plt.plot(steps_medium, avg_medium, label="gpt2-medium")
        if not math.isnan(small_mean):
            plt.axvline(small_mean, linestyle="--", linewidth=1, color="cornflowerblue", label="AoA small (mean step)")
        if not math.isnan(med_mean):
            plt.axvline(med_mean, linestyle="--", linewidth=1, color="orange", label="AoA medium (mean step)")
        plt.xlabel("Step")
        plt.ylabel("Average surprisal (bits)")
        plt.title(f"Top {k} simple words: surprisal vs step")
        plt.legend()
        out_path = os.path.join(args.out_dir, f"avg_surprisal_top{k}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # Mean layer attention for top-k simple words over timesteps
    for k in ks:
        words_k = simple_ranking[:k]
        avg_small_act = compute_avg_series(small_act, words_k)
        avg_medium_act = compute_avg_series(medium_act, words_k)

        fig = plt.figure()
        plt.plot(steps_small, avg_small_act, label="gpt2-small")
        plt.plot(steps_medium, avg_medium_act, label="gpt2-medium")
        plt.xlabel("Step")
        plt.ylabel("Mean layer attention")
        plt.title(f"Top {k} simple words: attention vs Step")
        plt.legend()
        out_path = os.path.join(args.out_dir, f"avg_attention_top{k}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # Child trajectories vs LLM surprisal (per word)
    if simple_ranking:
        max_words_side_by_side = 10
        n_plotted = 0
        margin_idx = 3
        months_arr = np.array(months, dtype=float)
        if months_arr.size > 1:
            months_norm = (months_arr - months_arr.min()) / (months_arr.max() - months_arr.min())
        else:
            months_norm = np.zeros_like(months_arr)

        for w in simple_ranking:
            if w not in word_to_curve:
                continue
            child_curve = word_to_curve[w]
            if not np.isfinite(child_curve).any():
                continue
            s_small = np.array(small_surpr[w], dtype=float)
            s_medium = np.array(medium_surpr[w], dtype=float)
            if not (np.isfinite(s_small).any() or np.isfinite(s_medium).any()):
                continue

            steps_small_arr = np.array(steps_small, dtype=float)
            steps_medium_arr = np.array(steps_medium, dtype=float)

            s_small_crop, steps_small_crop, thr_small = crop_with_threshold(
                s_small, steps_small_arr, args.baseline_bits, margin_idx
            )
            s_medium_crop, steps_medium_crop, thr_medium = crop_with_threshold(
                s_medium, steps_medium_arr, args.baseline_bits, margin_idx
            )

            if s_small_crop is None and s_medium_crop is None:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            if s_small_crop is not None and len(s_small_crop) > 0:
                axes[0].plot(steps_small_crop, s_small_crop, label="gpt2-small")
                if thr_small is not None:
                    axes[0].axhline(thr_small, linestyle="--", linewidth=1, color="cornflowerblue")
            if s_medium_crop is not None and len(s_medium_crop) > 0:
                axes[0].plot(steps_medium_crop, s_medium_crop, label="gpt2-medium")
                if thr_medium is not None:
                    axes[0].axhline(thr_medium, linestyle="--", linewidth=1, color="orange")

            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Surprisal (bits)")
            axes[0].set_title(f"{w} - LLM surprisal")
            axes[0].legend()
            axes[0].invert_yaxis()

            axes[1].plot(months, child_curve, marker="o")
            axes[1].axhline(0.5, linestyle="--", linewidth=1, color="cornflowerblue")
            axes[1].set_xlabel("Age (months)")
            axes[1].set_ylabel("Proportion producing")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_title(f"{w} - Children")

            ax3 = axes[2]
            ax3b = ax3.twinx()

            if s_small_crop is not None and len(s_small_crop) > 0:
                x_small_norm = normalize_x(steps_small_crop)
                ax3.plot(x_small_norm, s_small_crop, label="gpt2-small")
            if s_medium_crop is not None and len(s_medium_crop) > 0:
                x_medium_norm = normalize_x(steps_medium_crop)
                ax3.plot(x_medium_norm, s_medium_crop, label="gpt2-medium")

            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                ax3b.plot(months_norm[child_mask], child_curve[child_mask], marker="o", color="green", label="children")

            all_s_vals = []
            for arr in (s_small_crop, s_medium_crop):
                if arr is not None:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        all_s_vals.extend(arr.tolist())

            if all_s_vals:
                min_surprisal = float(min(all_s_vals))
                max_surprisal = float(max(all_s_vals))
                data_range = max_surprisal - min_surprisal
                if data_range <= 0:
                    data_range = 1.0
                buffer = 0.1 * data_range
                half_total = 0.5 * data_range + buffer
                y_max = thr_small + half_total
                y_min = thr_small - half_total
                ax3.set_ylim(y_max, y_min)

            ax3.set_xlim(0.0, 1.0)
            ax3.set_xlabel("Normalized timeline")
            ax3.set_ylabel("Surprisal (bits)")
            ax3b.set_ylabel("Proportion producing")
            ax3b.set_ylim(0.0, 1.0)
            ax3.set_title(f"{w} - Normalized aligned overlay")

            handles1, labels1 = ax3.get_legend_handles_labels()
            handles2, labels2 = ax3b.get_legend_handles_labels()
            if handles1 or handles2:
                ax3.legend(handles1 + handles2, labels1 + labels2, loc="best")

            fig.tight_layout()
            safe_w = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
            out_path_word = os.path.join(args.out_dir, f"word_{safe_w}_child_vs_llm_surprisal.png")
            fig.savefig(out_path_word, bbox_inches="tight")
            plt.close(fig)
            n_plotted += 1
            if n_plotted >= max_words_side_by_side:
                break

    # Child trajectories vs LLM attention (per word)
    if simple_ranking:
        max_words_side_by_side_att = 10
        n_plotted_att = 0

        months_arr = np.array(months, dtype=float)
        if months_arr.size > 1:
            months_norm = (months_arr - months_arr.min()) / (months_arr.max() - months_arr.min())
        else:
            months_norm = np.zeros_like(months_arr)

        for w in simple_ranking:
            if w not in word_to_curve:
                continue
            child_curve = word_to_curve[w]
            if not np.isfinite(child_curve).any():
                continue
            a_small = np.array(small_act[w], dtype=float)
            a_medium = np.array(medium_act[w], dtype=float)
            if not (np.isfinite(a_small).any() or np.isfinite(a_medium).any()):
                continue

            steps_small_arr = np.array(steps_small, dtype=float)
            steps_medium_arr = np.array(steps_medium, dtype=float)

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            if np.isfinite(a_small).any():
                axes[0].plot(steps_small_arr, a_small, label="gpt2-small")
            if np.isfinite(a_medium).any():
                axes[0].plot(steps_medium_arr, a_medium, label="gpt2-medium")

            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Mean layer attention")
            axes[0].set_title(f"{w} - LLM attention")
            axes[0].invert_yaxis()
            axes[0].legend()

            axes[1].plot(months, child_curve, marker="o")
            axes[1].axhline(0.5, linestyle="--", linewidth=1, color="cornflowerblue")
            axes[1].set_xlabel("Age (months)")
            axes[1].set_ylabel("Proportion producing")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_title(f"{w} - Children")

            ax3 = axes[2]
            ax3b = ax3.twinx()

            if np.isfinite(a_small).any():
                x_small_norm = normalize_x(steps_small_arr)
                ax3.plot(x_small_norm, a_small, label="gpt2-small")
            if np.isfinite(a_medium).any():
                x_medium_norm = normalize_x(steps_medium_arr)
                ax3.plot(x_medium_norm, a_medium, label="gpt2-medium")

            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                ax3b.plot(months_norm[child_mask], child_curve[child_mask], marker="o", color="green", label="children")

            ax3.set_xlim(0.0, 1.0)
            ax3.set_xlabel("Normalized timeline")
            ax3.set_ylabel("Mean layer attention")
            ax3b.set_ylabel("Proportion producing")
            ax3b.set_ylim(0.0, 1.0)
            ax3.set_title(f"{w} - Normalized aligned overlay")
            ax3.invert_yaxis()

            handles1, labels1 = ax3.get_legend_handles_labels()
            handles2, labels2 = ax3b.get_legend_handles_labels()
            if handles1 or handles2:
                ax3.legend(handles1 + handles2, labels1 + labels2, loc="best")

            fig.tight_layout()
            safe_w = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
            out_path_word_att = os.path.join(args.out_dir, f"word_{safe_w}_child_vs_llm_attention.png")
            fig.savefig(out_path_word_att, bbox_inches="tight")
            plt.close(fig)
            n_plotted_att += 1
            if n_plotted_att >= max_words_side_by_side_att:
                break


if __name__ == "__main__":
    main()
