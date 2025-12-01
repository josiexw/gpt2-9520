import os
import re
import argparse
import math
import pickle
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
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


def load_owt_counts(path: str) -> Dict[str, int]:
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            w = parts[0].lower()
            w = re.sub(r"\s*\([^)]*\)", "", w).strip()
            try:
                c = int(parts[1])
            except ValueError:
                continue
            if w:
                counts[w] = c
    return counts


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
    all_words_surprisal: Set[str] = set()
    all_words_act: Set[str] = set()
    for _, _, ws, wa in entries:
        all_words_surprisal.update(ws.keys())
        all_words_act.update(wa.keys())
    all_words = all_words_surprisal | all_words_act
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


def compute_avg_series(word_to_series: Dict[str, List[float]], words: List[str]) -> np.ndarray:
    if not words:
        return np.array([], dtype=float)
    arr = []
    for w in words:
        arr.append(np.array(word_to_series[w], dtype=float))
    arr = np.stack(arr, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def compute_thresholds_per_word(word_to_series: Dict[str, List[float]], baseline_bits: float, words: List[str]) -> Dict[str, float]:
    thr = {}
    for w in words:
        s = np.array(word_to_series[w], dtype=float)
        if not np.isfinite(s).any():
            continue
        s_min = float(np.nanmin(s))
        t = 0.5 * (baseline_bits + s_min)
        thr[w] = t
    return thr


def compute_llm_aoa_steps(word_to_series: Dict[str, List[float]], steps: List[int], baseline_bits: float, words: List[str]) -> Dict[str, float]:
    aoa_norm: Dict[str, float] = {}
    if not words:
        return aoa_norm
    step_arr = np.array(steps, dtype=float)
    min_step = float(np.min(step_arr))
    max_step = float(np.max(step_arr))
    span = max_step - min_step if max_step > min_step else 1.0
    for w in words:
        s = np.array(word_to_series[w], dtype=float)
        if not np.isfinite(s).any():
            continue
        s_min = float(np.nanmin(s))
        thr = 0.5 * (baseline_bits + s_min)
        idx_val = None
        for j, v in enumerate(s):
            if np.isfinite(v) and v <= thr:
                idx_val = j
                break
        if idx_val is None:
            continue
        step_val = float(step_arr[idx_val])
        aoa_norm[w] = (step_val - min_step) / span
    return aoa_norm


def find_nearest_step_index(steps: List[int], target_step: int) -> Tuple[int, int]:
    if not steps:
        return -1, -1
    arr = np.array(steps, dtype=int)
    j = int(np.argmin(np.abs(arr - target_step)))
    return j, int(arr[j])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_dir", type=str, default="stanford-gpt2-small-a_results")
    parser.add_argument("--medium_dir", type=str, default="stanford-gpt2-medium-a_results")
    parser.add_argument("--wordbank_csv", type=str, default="data/wordbank_item_data.csv")
    parser.add_argument("--owt_words_txt", type=str, default="data/owt_frequent.txt")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--out_txt", type=str, default="aoa_results.txt")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--ks", type=str, default="10,100,500")
    parser.add_argument("--baseline_bits", type=float, default=14.9)
    parser.add_argument("--attention_steps", type=str, default="200,20000,200000,392000")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)
    owt_counts = load_owt_counts(args.owt_words_txt)
    months, word_to_curve = load_wordbank_curves(args.wordbank_csv)

    steps_small, small_surpr, small_act, label_type_small = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, medium_act, label_type_medium = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    attention_steps = [int(x) for x in args.attention_steps.split(",") if x.strip()]

    with open(args.out_txt, "w") as fout:
        fout.write(f"Baseline bits (random-like): {args.baseline_bits:.4f}\n")
        fout.write(f"Label types: small={label_type_small}, medium={label_type_medium}\n")
        fout.write(f"Total words with AoA in Wordbank: {len(word_aoa)}\n")
        fout.write(f"Words present in both models: {len(available_words)}\n")
        fout.write(f"Words used in simple ranking (<= {args.max_simple}): {len(simple_ranking)}\n\n")

        fout.write("Average surprisal for top-k simple words and AoA thresholds (bits):\n")
        for k in ks:
            words_k = simple_ranking[:k]
            avg_small = compute_avg_series(small_surpr, words_k)
            avg_medium = compute_avg_series(medium_surpr, words_k)

            thr_small_dict = compute_thresholds_per_word(small_surpr, args.baseline_bits, words_k)
            thr_medium_dict = compute_thresholds_per_word(medium_surpr, args.baseline_bits, words_k)

            def stats_from_thr(d: Dict[str, float]) -> Tuple[float, float, float, float]:
                if not d:
                    return math.nan, math.nan, math.nan, math.nan
                vals = np.array(list(d.values()), dtype=float)
                return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

            small_mean, small_std, small_min, small_max = stats_from_thr(thr_small_dict)
            med_mean, med_std, med_min, med_max = stats_from_thr(thr_medium_dict)

            fig = plt.figure()
            plt.plot(steps_small, avg_small, label="gpt2-small")
            plt.plot(steps_medium, avg_medium, label="gpt2-medium")
            if not math.isnan(small_mean):
                plt.axhline(small_mean, linestyle="--", linewidth=1, color="cornflowerblue", label="AoA threshold small (mean)")
            if not math.isnan(med_mean):
                plt.axhline(med_mean, linestyle="--", linewidth=1, color="orange", label="AoA threshold medium (mean)")
            plt.xlabel(f"{label_type_small or 'step'}")
            plt.ylabel("Average surprisal (bits)")
            plt.title(f"Top {k} simple words: surprisal vs {label_type_small or 'step'}")
            plt.legend()
            out_path = os.path.join(args.out_dir, f"avg_surprisal_top{k}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            fout.write(
                f"k={k}, n_words={len(words_k)}\n"
                f"  small_thr_bits_mean={small_mean:.4f}, std={small_std:.4f}, min={small_min:.4f}, max={small_max:.4f}\n"
                f"  medium_thr_bits_mean={med_mean:.4f}, std={med_std:.4f}, min={med_min:.4f}, max={med_max:.4f}\n"
            )

        fout.write("\nMean layer attention for top-k simple words:\n")
        for k in ks:
            words_k = simple_ranking[:k]
            avg_small_act = compute_avg_series(small_act, words_k)
            avg_medium_act = compute_avg_series(medium_act, words_k)

            fig = plt.figure()
            plt.plot(steps_small, avg_small_act, label="gpt2-small")
            plt.plot(steps_medium, avg_medium_act, label="gpt2-medium")
            plt.xlabel(f"{label_type_small or 'step'}")
            plt.ylabel("Mean layer attention")
            plt.title(f"Top {k} simple words: attention vs {label_type_small or 'step'}")
            plt.legend()
            out_path = os.path.join(args.out_dir, f"avg_attention_top{k}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            fout.write(f"k={k}, n_words={len(words_k)} attention plots saved.\n")

        aoa_small = compute_llm_aoa_steps(small_surpr, steps_small, args.baseline_bits, simple_ranking)
        aoa_medium = compute_llm_aoa_steps(medium_surpr, steps_medium, args.baseline_bits, simple_ranking)
        common_for_aoa = [w for w in simple_ranking if w in aoa_small and w in aoa_medium]

        fout.write("\nLLM AoA vs simplicity rank:\n")
        if common_for_aoa:
            ranks = np.arange(1, len(common_for_aoa) + 1)
            y_small = np.array([aoa_small[w] for w in common_for_aoa])
            y_medium = np.array([aoa_medium[w] for w in common_for_aoa])

            fig2 = plt.figure()
            plt.plot(ranks, y_small, label="gpt2-small")
            plt.plot(ranks, y_medium, label="gpt2-medium")
            plt.xlabel("Word rank by simplicity (Wordbank AoA, lower = earlier)")
            plt.ylabel("LLM AoA (normalized training step)")
            plt.title("LLM AoA vs simplicity rank")
            plt.legend()
            out_path2 = os.path.join(args.out_dir, "llm_aoa_vs_rank.png")
            fig2.savefig(out_path2, bbox_inches="tight")
            plt.close(fig2)

            fout.write(f"LLM AoA points used: {len(common_for_aoa)}\n")
        else:
            fout.write("No overlapping words for LLM AoA curves.\n")

        fout.write("\nChild trajectories vs LLM surprisal (per word):\n")
        if common_for_aoa:
            max_words_side_by_side = 50
            n_plotted = 0
            margin_idx = 2
            for w in common_for_aoa:
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

                def crop_with_threshold(s, steps_arr, baseline_bits, margin_idx):
                    s = np.array(s, dtype=float)
                    if not np.isfinite(s).any():
                        return None, None, None, None
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
                    s_crop = s[:end_idx + 1]
                    steps_crop = steps_arr[:end_idx + 1]
                    return s_crop, steps_crop, thr, idx_cross

                s_small_crop, steps_small_crop, thr_small, idx_small = crop_with_threshold(
                    s_small, steps_small_arr, args.baseline_bits, margin_idx
                )
                s_medium_crop, steps_medium_crop, thr_medium, idx_medium = crop_with_threshold(
                    s_medium, steps_medium_arr, args.baseline_bits, margin_idx
                )

                if s_small_crop is None and s_medium_crop is None:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(8, 3))

                if s_small_crop is not None and len(s_small_crop) > 0:
                    axes[0].plot(steps_small_crop, s_small_crop, label="gpt2-small")
                    if thr_small is not None:
                        axes[0].axhline(thr_small, linestyle="--", linewidth=1, color="cornflowerblue")
                if s_medium_crop is not None and len(s_medium_crop) > 0:
                    axes[0].plot(steps_medium_crop, s_medium_crop, label="gpt2-medium")
                    if thr_medium is not None:
                        axes[0].axhline(thr_medium, linestyle="--", linewidth=1, color="orange")

                axes[0].set_xlabel(label_type_small or "step")
                axes[0].set_ylabel("Surprisal (bits)")
                axes[0].set_title(f"{w} - LLM surprisal")
                axes[0].legend()
                axes[0].invert_yaxis()

                axes[1].plot(months, child_curve, marker="o")
                axes[1].axhline(0.5, linestyle="--", linewidth=1, color="cornflowerblue")
                axes[1].set_xlabel("Age (months)")
                axes[1].set_ylabel("Proportion producing")
                axes[1].set_ylim(0.0, 1.0)
                axes[1].set_title(f"{w} - children")

                fig.tight_layout()
                safe_w = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
                out_path_word = os.path.join(args.out_dir, f"word_{safe_w}_child_vs_llm.png")
                fig.savefig(out_path_word, bbox_inches="tight")
                plt.close(fig)
                n_plotted += 1
                if n_plotted >= max_words_side_by_side:
                    break
            fout.write(f"  Side-by-side plots saved for {n_plotted} words.\n")
        else:
            fout.write("  No words for child vs LLM per-word plots.\n")

        fout.write("\nSurprisal trajectories stratified by child AoA bins:\n")
        bins = [("early", 16, 20), ("mid", 21, 24), ("late", 25, 30)]
        bin_to_words = {name: [] for name, _, _ in bins}
        for w in simple_ranking:
            aoa = word_aoa.get(w, math.nan)
            if math.isnan(aoa):
                continue
            for name, lo, hi in bins:
                if lo <= aoa <= hi:
                    bin_to_words[name].append(w)
                    break
        for name, lo, hi in bins:
            fout.write(f"  bin {name} ({lo}-{hi}): n_words={len(bin_to_words[name])}\n")
        fig_bins_small = plt.figure()
        for name, _, _ in bins:
            words_bin = bin_to_words[name]
            if not words_bin:
                continue
            avg_s = compute_avg_series(small_surpr, words_bin)
            plt.plot(steps_small, avg_s, label=name)
        plt.xlabel(f"{label_type_small or 'step'}")
        plt.ylabel("Average surprisal (bits)")
        plt.title("Small: surprisal trajectories by AoA bin")
        plt.legend()
        out_bins_small = os.path.join(args.out_dir, "surprisal_bins_small.png")
        fig_bins_small.savefig(out_bins_small, bbox_inches="tight")
        plt.close(fig_bins_small)
        fig_bins_medium = plt.figure()
        for name, _, _ in bins:
            words_bin = bin_to_words[name]
            if not words_bin:
                continue
            avg_m = compute_avg_series(medium_surpr, words_bin)
            plt.plot(steps_medium, avg_m, label=name)
        plt.xlabel(f"{label_type_medium or 'step'}")
        plt.ylabel("Average surprisal (bits)")
        plt.title("Medium: surprisal trajectories by AoA bin")
        plt.legend()
        out_bins_medium = os.path.join(args.out_dir, "surprisal_bins_medium.png")
        fig_bins_medium.savefig(out_bins_medium, bbox_inches="tight")
        plt.close(fig_bins_medium)

        fout.write("\nSurprisal vs OWT frequency (early vs late):\n")
        if simple_ranking:
            early_idx_s, early_step_s = 0, steps_small[0]
            late_idx_s, late_step_s = len(steps_small) - 1, steps_small[-1]
            early_idx_m, early_step_m = 0, steps_medium[0]
            late_idx_m, late_step_m = len(steps_medium) - 1, steps_medium[-1]
            freq_words = [w for w in simple_ranking if w in owt_counts]
            fout.write(f"  Words with OWT frequency among used: {len(freq_words)}\n")

            def build_xy(model_surpr, idx, words):
                xs = []
                ys = []
                for w in words:
                    c = owt_counts.get(w, None)
                    if c is None or c <= 0:
                        continue
                    s_val = model_surpr.get(w, [math.nan])[idx]
                    if not math.isfinite(s_val):
                        continue
                    xs.append(math.log10(c))
                    ys.append(s_val)
                return np.array(xs, dtype=float), np.array(ys, dtype=float)

            x_s_early, y_s_early = build_xy(small_surpr, early_idx_s, freq_words)
            x_s_late, y_s_late = build_xy(small_surpr, late_idx_s, freq_words)
            x_m_early, y_m_early = build_xy(medium_surpr, early_idx_m, freq_words)
            x_m_late, y_m_late = build_xy(medium_surpr, late_idx_m, freq_words)

            fig_freq_early = plt.figure()
            if len(x_s_early) > 0:
                plt.scatter(x_s_early, y_s_early, label=f"small (step {early_step_s})", alpha=0.5)
            if len(x_m_early) > 0:
                plt.scatter(x_m_early, y_m_early, label=f"medium (step {early_step_m})", alpha=0.5)
            plt.xlabel("log10 OWT frequency")
            plt.ylabel("Surprisal (bits)")
            plt.title("Surprisal vs OWT frequency (early)")
            plt.legend()
            out_freq_early = os.path.join(args.out_dir, "surprisal_vs_freq_early.png")
            fig_freq_early.savefig(out_freq_early, bbox_inches="tight")
            plt.close(fig_freq_early)

            fig_freq_late = plt.figure()
            if len(x_s_late) > 0:
                plt.scatter(x_s_late, y_s_late, label=f"small (step {late_step_s})", alpha=0.5)
            if len(x_m_late) > 0:
                plt.scatter(x_m_late, y_m_late, label=f"medium (step {late_step_m})", alpha=0.5)
            plt.xlabel("log10 OWT frequency")
            plt.ylabel("Surprisal (bits)")
            plt.title("Surprisal vs OWT frequency (late)")
            plt.legend()
            out_freq_late = os.path.join(args.out_dir, "surprisal_vs_freq_late.png")
            fig_freq_late.savefig(out_freq_late, bbox_inches="tight")
            plt.close(fig_freq_late)

            def corr_safe(x, y):
                if len(x) > 1:
                    return float(np.corrcoef(x, y)[0, 1])
                return math.nan

            fout.write(
                f"  corr_small_early={corr_safe(x_s_early, y_s_early):.4f}, corr_small_late={corr_safe(x_s_late, y_s_late):.4f}\n"
            )
            fout.write(
                f"  corr_medium_early={corr_safe(x_m_early, y_m_early):.4f}, corr_medium_late={corr_safe(x_m_late, y_m_late):.4f}\n"
            )
        else:
            fout.write("  No words for surprisal vs OWT frequency.\n")

        fout.write("\nSurprisal-attention coupling (early vs late):\n")

        def build_supr_act(model_surpr, model_act, idx, words):
            xs = []
            ys = []
            for w in words:
                s_val = model_surpr.get(w, [math.nan])[idx]
                a_val = model_act.get(w, [math.nan])[idx]
                if not (math.isfinite(s_val) and math.isfinite(a_val)):
                    continue
                xs.append(s_val)
                ys.append(a_val)
            return np.array(xs, dtype=float), np.array(ys, dtype=float)

        if simple_ranking:
            early_idx_s, early_step_s = 0, steps_small[0]
            late_idx_s, late_step_s = len(steps_small) - 1, steps_small[-1]
            early_idx_m, early_step_m = 0, steps_medium[0]
            late_idx_m, late_step_m = len(steps_medium) - 1, steps_medium[-1]

            x_se, y_se = build_supr_act(small_surpr, small_act, early_idx_s, simple_ranking)
            x_sl, y_sl = build_supr_act(small_surpr, small_act, late_idx_s, simple_ranking)
            x_me, y_me = build_supr_act(medium_surpr, medium_act, early_idx_m, simple_ranking)
            x_ml, y_ml = build_supr_act(medium_surpr, medium_act, late_idx_m, simple_ranking)

            fig_sa_se = plt.figure()
            if len(x_se) > 0:
                plt.scatter(x_se, y_se, alpha=0.5)
            plt.xlabel("Surprisal (bits)")
            plt.ylabel("Mean layer attention")
            plt.title(f"Small: Surprisal-attention (early step {early_step_s})")
            out_sa_se = os.path.join(args.out_dir, "surprisal_attention_small_early.png")
            fig_sa_se.savefig(out_sa_se, bbox_inches="tight")
            plt.close(fig_sa_se)

            fig_sa_sl = plt.figure()
            if len(x_sl) > 0:
                plt.scatter(x_sl, y_sl, alpha=0.5)
            plt.xlabel("Surprisal (bits)")
            plt.ylabel("Mean layer attention")
            plt.title(f"Small: Surprisal-attention (late step {late_step_s})")
            out_sa_sl = os.path.join(args.out_dir, "surprisal_attention_small_late.png")
            fig_sa_sl.savefig(out_sa_sl, bbox_inches="tight")
            plt.close(fig_sa_sl)

            fig_sa_me = plt.figure()
            if len(x_me) > 0:
                plt.scatter(x_me, y_me, alpha=0.5)
            plt.xlabel("Surprisal (bits)")
            plt.ylabel("Mean layer attention")
            plt.title(f"Medium: Surprisal-attention (early step {early_step_m})")
            out_sa_me = os.path.join(args.out_dir, "surprisal_attention_medium_early.png")
            fig_sa_me.savefig(out_sa_me, bbox_inches="tight")
            plt.close(fig_sa_me)

            fig_sa_ml = plt.figure()
            if len(x_ml) > 0:
                plt.scatter(x_ml, y_ml, alpha=0.5)
            plt.xlabel("Surprisal (bits)")
            plt.ylabel("Mean layer attention")
            plt.title(f"Medium: Surprisal-attention (late step {late_step_m})")
            out_sa_ml = os.path.join(args.out_dir, "surprisal_attention_medium_late.png")
            fig_sa_ml.savefig(out_sa_ml, bbox_inches="tight")
            plt.close(fig_sa_ml)

            def corr_safe2(x, y):
                if len(x) > 1:
                    return float(np.corrcoef(x, y)[0, 1])
                return math.nan

            fout.write(
                f"  small_early_corr={corr_safe2(x_se, y_se):.4f}, small_late_corr={corr_safe2(x_sl, y_sl):.4f}\n"
            )
            fout.write(
                f"  medium_early_corr={corr_safe2(x_me, y_me):.4f}, medium_late_corr={corr_safe2(x_ml, y_ml):.4f}\n"
            )
        else:
            fout.write("  No words for surprisal-attention coupling.\n")


if __name__ == "__main__":
    main()
