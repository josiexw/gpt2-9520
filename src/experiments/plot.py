import os
import re
import argparse
import math
import pickle
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset


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


def compute_neural_aoa_steps(word_to_series: Dict[str, List[float]], steps: List[int], baseline_bits: float, words: List[str]) -> Dict[str, float]:
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


def count_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def compute_cosmopedia_frequent_words(simple_words: Set[str], dataset_id: str, subsets: List[str], top_k: int, token_limit: int) -> Tuple[int, int, int, float]:
    counter = Counter()
    total_tokens = 0
    for subset in subsets:
        if total_tokens >= token_limit:
            break
        ds = load_dataset(dataset_id, subset, split="train")
        for ex in ds:
            if total_tokens >= token_limit:
                break
            text = ex.get("text") or ""
            tokens = count_words(text)
            if not tokens:
                continue
            counter.update(tokens)
            total_tokens += len(tokens)
            if total_tokens >= token_limit:
                break
    most_common = [w for w, _ in counter.most_common(top_k)]
    frequent_set = set(most_common)
    simple_used = set(simple_words)
    overlap = simple_used & frequent_set
    n_simple = len(simple_used)
    n_overlap = len(overlap)
    percent = 0.0
    if n_simple > 0:
        percent = 100.0 * n_overlap / n_simple
    return n_simple, n_overlap, top_k, percent


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
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--out_txt", type=str, default="aoa_results.txt")
    parser.add_argument("--max_simple", type=int, default=500)
    parser.add_argument("--ks", type=str, default="10,100,500")
    parser.add_argument("--baseline_bits", type=float, default=14.9)
    parser.add_argument("--cosmopedia_dataset", type=str, default="HuggingFaceTB/cosmopedia")
    parser.add_argument("--cosmopedia_subsets", type=str, default="auto_math_text,khanacademy,openstax,stanford,stories,web_samples_v1,web_samples_v2,wikihow")
    parser.add_argument("--cosmopedia_top_k", type=int, default=5000)
    parser.add_argument("--cosmopedia_token_limit", type=int, default=200000)
    parser.add_argument("--activation_steps", type=str, default="200,20000,200000,392000")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    word_aoa = load_wordbank_aoa(args.wordbank_csv)

    steps_small, small_surpr, small_act, label_type_small = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, medium_act, label_type_medium = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    activation_steps = [int(x) for x in args.activation_steps.split(",") if x.strip()]

    with open(args.out_txt, "w") as fout:
        fout.write(f"Baseline bits (random-like): {args.baseline_bits:.4f}\n")
        fout.write(f"Label types: small={label_type_small}, medium={label_type_medium}\n\n")

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

        fout.write("\nAverage activation (mean layer attention) for top-k simple words:\n")
        for k in ks:
            words_k = simple_ranking[:k]
            avg_small_act = compute_avg_series(small_act, words_k)
            avg_medium_act = compute_avg_series(medium_act, words_k)

            fig = plt.figure()
            plt.plot(steps_small, avg_small_act, label="gpt2-small")
            plt.plot(steps_medium, avg_medium_act, label="gpt2-medium")
            plt.xlabel(f"{label_type_small or 'step'}")
            plt.ylabel("Average activation (mean layer attention)")
            plt.title(f"Top {k} simple words: activation vs {label_type_small or 'step'}")
            plt.legend()
            out_path = os.path.join(args.out_dir, f"avg_activation_top{k}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            fout.write(f"k={k}, n_words={len(words_k)} activation plots saved.\n")

        aoa_small = compute_neural_aoa_steps(small_surpr, steps_small, args.baseline_bits, simple_ranking)
        aoa_medium = compute_neural_aoa_steps(medium_surpr, steps_medium, args.baseline_bits, simple_ranking)
        common_for_aoa = [w for w in simple_ranking if w in aoa_small and w in aoa_medium]

        fout.write("\nNeural AoA vs simplicity rank:\n")
        if common_for_aoa:
            ranks = np.arange(1, len(common_for_aoa) + 1)
            y_small = np.array([aoa_small[w] for w in common_for_aoa])
            y_medium = np.array([aoa_medium[w] for w in common_for_aoa])

            fig2 = plt.figure()
            plt.plot(ranks, y_small, label="small")
            plt.plot(ranks, y_medium, label="medium")
            plt.xlabel("Word rank by simplicity (Wordbank AoA, lower = earlier)")
            plt.ylabel("Neural AoA (normalized training step)")
            plt.title("Neural AoA vs simplicity rank")
            plt.legend()
            out_path2 = os.path.join(args.out_dir, "neural_aoa_vs_rank.png")
            fig2.savefig(out_path2, bbox_inches="tight")
            plt.close(fig2)

            fout.write(f"Neural AoA points used: {len(common_for_aoa)}\n")
        else:
            fout.write("No overlapping words for neural AoA curves.\n")

        fout.write("\nRank vs activation at selected steps:\n")
        for target in activation_steps:
            idx_s, actual_s = find_nearest_step_index(steps_small, target)
            idx_m, actual_m = find_nearest_step_index(steps_medium, target)
            if idx_s < 0 or idx_m < 0:
                fout.write(f"  target_step={target}: no valid checkpoints in one of the models.\n")
                continue

            acts_small = []
            acts_medium = []
            ranks = []
            for r, w in enumerate(simple_ranking, start=1):
                a_s = small_act.get(w, [math.nan])[idx_s]
                a_m = medium_act.get(w, [math.nan])[idx_m]
                if not (math.isfinite(a_s) and math.isfinite(a_m)):
                    continue
                ranks.append(r)
                acts_small.append(a_s)
                acts_medium.append(a_m)

            if not ranks:
                fout.write(f"  target_step={target}: no finite activations for common words.\n")
                continue

            ranks_arr = np.array(ranks, dtype=int)
            acts_small_arr = np.array(acts_small, dtype=float)
            acts_medium_arr = np.array(acts_medium, dtype=float)

            fig3 = plt.figure()
            plt.plot(ranks_arr, acts_small_arr, label=f"small (step {actual_s})")
            plt.plot(ranks_arr, acts_medium_arr, label=f"medium (step {actual_m})")
            plt.xlabel("Word rank by simplicity")
            plt.ylabel("Activation (mean layer attention)")
            plt.title(f"Rank vs activation at target step≈{target}")
            plt.legend()
            out_path3 = os.path.join(args.out_dir, f"rank_vs_activation_step{target}.png")
            fig3.savefig(out_path3, bbox_inches="tight")
            plt.close(fig3)

            fout.write(
                f"  target_step={target}: small_step={actual_s}, medium_step={actual_m}, n_words={len(ranks_arr)}\n"
            )

        subsets = [s.strip() for s in args.cosmopedia_subsets.split(",") if s.strip()]
        simple_used_set = set(simple_ranking)
        n_simple, n_overlap, top_k_freq, percent = compute_cosmopedia_frequent_words(
            simple_used_set,
            args.cosmopedia_dataset,
            subsets,
            args.cosmopedia_top_k,
            args.cosmopedia_token_limit,
        )
        fout.write(
            f"\nCosmopedia overlap:\n"
            f"  simple_words_used={n_simple}, frequent_top_k={top_k_freq}, overlap={n_overlap}, percent={percent:.2f}\n"
        )


if __name__ == "__main__":
    main()
