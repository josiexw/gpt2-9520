import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


def normalize_x_aligned(xs, align_idx, child_aoa_x):
    xs = np.array(xs, dtype=float)
    if xs.size == 0 or align_idx >= len(xs) or align_idx < 0:
        return normalize_x(xs)
    
    # Interpolate
    if align_idx == int(align_idx):
        x_align = xs[int(align_idx)]
    else:
        idx_low = int(np.floor(align_idx))
        idx_high = int(np.ceil(align_idx))
        if idx_high >= len(xs):
            x_align = xs[idx_low]
        else:
            frac = align_idx - idx_low
            x_align = xs[idx_low] + frac * (xs[idx_high] - xs[idx_low])
    
    x_start = xs[0]
    x_end = xs[-1]
    xs_norm = (xs - x_start) / (x_end - x_start)
    align_norm = (x_align - x_start) / (x_end - x_start)

    if align_norm == 0 or not np.isfinite(child_aoa_x) or child_aoa_x == 0:
        return xs_norm

    xs_aligned = xs_norm * (child_aoa_x / align_norm)
    
    return xs_aligned


def compute_threshold_crossing_idx(s, baseline_bits):
    s = np.array(s, dtype=float)
    if not np.isfinite(s).any():
        return None
    s_min = float(np.nanmin(s))
    thr = 0.5 * (baseline_bits + s_min)
    
    for j, v in enumerate(s):
        if np.isfinite(v) and v <= thr:
            return j
    return len(s) - 1


def crop_with_threshold(s, steps_arr, baseline_bits, margin_idx):
    s = np.array(s, dtype=float)
    if not np.isfinite(s).any():
        return None, None, None, None
    s_min = float(np.nanmin(s))
    thr = 0.5 * (baseline_bits + s_min)
    idx_cross = None
    
    for j in range(1, len(s)):
        v_prev = s[j-1]
        v_curr = s[j]
        if np.isfinite(v_prev) and np.isfinite(v_curr):
            if (v_prev - thr) * (v_curr - thr) <= 0:
                # Linear interpolation to find exact crossing point
                if v_curr != v_prev:
                    frac = (thr - v_prev) / (v_curr - v_prev)
                    idx_cross = (j - 1) + frac
                else:
                    idx_cross = float(j)
                break
    
    if idx_cross is None:
        idx_cross = float(len(s) - 1)
    
    # For cropping, use integer index
    idx_cross_int = int(np.ceil(idx_cross))
    end_idx = min(idx_cross_int + margin_idx, len(s) - 1)
    s_crop = s[: end_idx + 1]
    steps_crop = steps_arr[: end_idx + 1]
    
    return s_crop, steps_crop, thr, idx_cross


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
    child_interp_aoa = compute_child_interp_aoa(args.wordbank_csv)

    steps_small, small_surpr, _ = load_results_dir(args.small_dir)
    steps_medium, medium_surpr, _ = load_results_dir(args.medium_dir)

    words_small = set(small_surpr.keys())
    words_medium = set(medium_surpr.keys())
    available_words = words_small & words_medium

    simple_ranking = get_simple_ranking(word_aoa, available_words, args.max_simple)

    # Child trajectories vs LLM surprisal (per word)
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

        s_small_crop, steps_small_crop, thr_small, idx_cross_small = crop_with_threshold(
            s_small, steps_small_arr, args.baseline_bits, margin_idx
        )
        s_medium_crop, steps_medium_crop, thr_medium, idx_cross_medium = crop_with_threshold(
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

        # Align AoA
        child_aoa_x = None
        if w in child_interp_aoa:
            child_aoa_month = child_interp_aoa[w]
            if months_arr.size > 1:
                child_aoa_x = (child_aoa_month - months_arr.min()) / (months_arr.max() - months_arr.min())
                child_aoa_x = np.clip(child_aoa_x, 0.0, 1.0)
        else:
            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                first_valid_val = child_curve[child_mask][0]
                if first_valid_val >= 0.5:
                    child_aoa_x = 0.0
                else:
                    plt.close(fig)
                    continue

        if child_aoa_x is None:
            plt.close(fig)
            continue

        child_mask = np.isfinite(child_curve)
        if child_mask.any():
            ax3b.plot(months_norm[child_mask], child_curve[child_mask], marker="o", color="green", label="children")

        # Normalize surprisal curves so threshold is at 0.5
        if thr_small is not None and np.isfinite(s_small).any():
            s_small_max = np.nanmax(s_small)
            s_small_range = s_small_max - thr_small
            if s_small_range > 0:
                s_small_norm = 0.5 + 0.5 * (thr_small - s_small) / s_small_range
            else:
                s_small_norm = np.full_like(s_small, 0.5)
        else:
            s_small_norm = None

        if thr_medium is not None and np.isfinite(s_medium).any():
            s_medium_max = np.nanmax(s_medium)
            s_medium_range = s_medium_max - thr_medium
            if s_medium_range > 0:
                s_medium_norm = 0.5 + 0.5 * (thr_medium - s_medium) / s_medium_range
            else:
                s_medium_norm = np.full_like(s_medium, 0.5)
        else:
            s_medium_norm = None

        # Plot normalized surprisal
        x_small_norm = normalize_x_aligned(steps_small_arr, idx_cross_small, child_aoa_x)
        ax3.plot(x_small_norm, s_small_norm, label="gpt2-small")
        x_medium_norm = normalize_x_aligned(steps_medium_arr, idx_cross_medium, child_aoa_x)
        ax3.plot(x_medium_norm, s_medium_norm, label="gpt2-medium")
        ax3.set_ylim(0.0, 1.0)

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


        # MSE
        mse_results = []
            
        # Create interpolation functions and evaluate at 100 points
        x_eval = np.linspace(0, 1, 100)
        
        mask_small = np.isfinite(s_small_norm) & np.isfinite(x_small_norm)
        if mask_small.sum() >= 2:
            s_small_interp = np.interp(x_eval, x_small_norm[mask_small], s_small_norm[mask_small])
        else:
            s_small_interp = None
        
        mask_medium = np.isfinite(s_medium_norm) & np.isfinite(x_medium_norm)
        if mask_medium.sum() >= 2:
            s_medium_interp = np.interp(x_eval, x_medium_norm[mask_medium], s_medium_norm[mask_medium])
        else:
            s_medium_interp = None
        
        # Interpolate child curve
        if child_mask.sum() >= 2:
            child_interp = np.interp(x_eval, months_norm[child_mask], child_curve[child_mask])
        else:
            child_interp = None
        
        # Calculate MSEs
        mse_small_child = None
        mse_medium_child = None
        mse_small_medium = None

        # Invert surprisal so it increases
        if s_small_interp is not None and child_interp is not None:
            s_small_interp_inverted = 1.0 - s_small_interp
            mse_small_child = np.mean((s_small_interp_inverted - child_interp) ** 2)

        if s_medium_interp is not None and child_interp is not None:
            s_medium_interp_inverted = 1.0 - s_medium_interp
            mse_medium_child = np.mean((s_medium_interp_inverted - child_interp) ** 2)

        if s_small_interp is not None and s_medium_interp is not None:
            mse_small_medium = np.mean((s_small_interp - s_medium_interp) ** 2)

        mse_results.append({
            'word': w,
            'mse_small_child': mse_small_child,
            'mse_medium_child': mse_medium_child,
            'mse_small_medium': mse_small_medium
        })

        # Save MSE csv
        if mse_results:
            mse_df = pd.DataFrame(mse_results)
            mse_csv_path = os.path.join("figs_paper", "mse_results.csv")
            if os.path.exists(mse_csv_path):
                existing_df = pd.read_csv(mse_csv_path)

                if not existing_df.empty and not mse_df.empty:
                    mse_df = pd.concat([existing_df, mse_df], ignore_index=True)
                elif not existing_df.empty:
                    mse_df = existing_df
            mse_df.to_csv(mse_csv_path, index=False)


    # Merged fig for paper
    ranks = [14, 30, 48, 149, 396, 484]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.35, 'wspace': 0.1})
    axes = axes.flatten()
    
    for plot_idx, rank_idx in enumerate(ranks):
        if rank_idx >= len(simple_ranking):
            continue
            
        w = simple_ranking[rank_idx]
        ax = axes[plot_idx]
        ax2 = ax.twinx()
        
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
        
        _, _, thr_small, idx_cross_small = crop_with_threshold(
            s_small, steps_small_arr, args.baseline_bits, margin_idx
        )
        _, _, thr_medium, idx_cross_medium = crop_with_threshold(
            s_medium, steps_medium_arr, args.baseline_bits, margin_idx
        )
        
        # Get child AoA
        child_aoa_x = None
        if w in child_interp_aoa:
            child_aoa_month = child_interp_aoa[w]
            if months_arr.size > 1:
                child_aoa_x = (child_aoa_month - months_arr.min()) / (months_arr.max() - months_arr.min())
                child_aoa_x = np.clip(child_aoa_x, 0.0, 1.0)
        else:
            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                first_valid_val = child_curve[child_mask][0]
                if first_valid_val >= 0.5:
                    child_aoa_x = 0.0
        
        if child_aoa_x is None:
            continue
        
        # Normalize surprisal curves
        if thr_small is not None and np.isfinite(s_small).any():
            s_small_min = np.nanmin(s_small)
            s_small_range = np.nanmax(s_small) - s_small_min
            if s_small_range > 0:
                s_small_norm = (s_small - thr_small) / s_small_range + 0.5
            else:
                s_small_norm = np.full_like(s_small, 0.5)
        else:
            s_small_norm = None
        
        if thr_medium is not None and np.isfinite(s_medium).any():
            s_medium_min = np.nanmin(s_medium)
            s_medium_range = np.nanmax(s_medium) - s_medium_min
            if s_medium_range > 0:
                s_medium_norm = (s_medium - thr_medium) / s_medium_range + 0.5
            else:
                s_medium_norm = np.full_like(s_medium, 0.5)
        else:
            s_medium_norm = None
        
        # Plot normalized surprisal
        if s_small_norm is not None:
            x_small_norm = normalize_x_aligned(steps_small_arr, idx_cross_small, child_aoa_x)
            ax.plot(x_small_norm, s_small_norm, label="gpt2-small", color='C0', linewidth=2)
        
        if s_medium_norm is not None:
            x_medium_norm = normalize_x_aligned(steps_medium_arr, idx_cross_medium, child_aoa_x)
            ax.plot(x_medium_norm, s_medium_norm, label="gpt2-medium", color='C1', linewidth=2)
        
        # Plot child curve
        child_mask = np.isfinite(child_curve)
        if child_mask.any():
            ax2.plot(months_norm[child_mask], child_curve[child_mask], 
                    marker="o", color="green", label="children", markersize=5, linewidth=2)
        
        ax.set_ylim(1.0, 0.0)
        ax.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.0)
        
        ax.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax2.set_xticks([0.0, 0.5, 1.0])
        ax2.set_yticks([0.0, 0.5, 1.0])
        
        ax.set_title(f"{w} (rank {rank_idx})", fontsize=24)
        
        if plot_idx == 2:
            ax.set_ylabel("Normalized surprisal", fontsize=22)
        if plot_idx == 3:
            ax2.set_ylabel("Proportion producing", fontsize=22)
        
        if plot_idx % 2 == 1:
            ax.set_yticklabels([])
        if plot_idx % 2 == 0:
            ax2.set_yticklabels([])
    
    fig.text(0.5, 0.05, 'Normalized timeline', ha='center', fontsize=22)
    
    handles1 = [plt.Line2D([0], [0], color='C0', linewidth=2, label='gpt2-small'),
                plt.Line2D([0], [0], color='C1', linewidth=2, label='gpt2-medium'),
                plt.Line2D([0], [0], color='green', marker='o', linewidth=2, label='children')]
    fig.legend(handles=handles1, loc='upper center', ncol=3, 
              bbox_to_anchor=(0.5, 0.98), fontsize=20, frameon=False)
    
    merged_path = os.path.join("figs_paper", "merged_word_trajectories.png")
    fig.savefig(merged_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Saved merged plot to {merged_path}")


    # Misaligned plot examples
    target_words = ['applesauce', 'tickle', 'tummy']
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), gridspec_kw={'hspace': 0.4, 'wspace': 0.2})
    
    for col_idx, w in enumerate(target_words):
        if w not in simple_ranking or w not in word_to_curve:
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
        
        s_small_crop, steps_small_crop, thr_small, idx_cross_small = crop_with_threshold(
            s_small, steps_small_arr, args.baseline_bits, margin_idx
        )
        s_medium_crop, steps_medium_crop, thr_medium, idx_cross_medium = crop_with_threshold(
            s_medium, steps_medium_arr, args.baseline_bits, margin_idx
        )
        
        # LLM Surprisal
        ax0 = axes[0, col_idx]
        if s_small_crop is not None and len(s_small_crop) > 0:
            ax0.plot(steps_small_crop, s_small_crop, label="gpt2-small", color='C0', linewidth=2)
            if thr_small is not None:
                ax0.axhline(thr_small, linestyle="--", linewidth=2, color="C0", alpha=0.5)
        if s_medium_crop is not None and len(s_medium_crop) > 0:
            ax0.plot(steps_medium_crop, s_medium_crop, label="gpt2-medium", color='C1', linewidth=2)
            if thr_medium is not None:
                ax0.axhline(thr_medium, linestyle="--", linewidth=2, color="C1", alpha=0.5)
        ax0.invert_yaxis()
        ax0.set_title(f"{w}", fontsize=28)
        axes[0,0].set_xticks([0, 2e5, 4e5])
        axes[0,1].set_xticks([0, 2e5, 4e5])
        axes[0,2].set_xticks([0, 3e4, 6e4])
        ax0.tick_params(labelsize=18)
        if col_idx == 0:
            ax0.set_ylabel("Surprisal (bits)", fontsize=26)
        
        # Child proportions
        ax1 = axes[1, col_idx]
        ax1.plot(months, child_curve, marker="o", color="green", markersize=6, linewidth=2)
        ax1.axhline(0.5, linestyle="--", linewidth=2, color="green", alpha=0.5)
        ax1.set_ylim(0.0, 1.0)
        ax1.tick_params(labelsize=18)
        if col_idx == 0:
            ax1.set_ylabel("Proportion producing", fontsize=26)
        
        # Overlay
        ax2 = axes[2, col_idx]
        ax2_twin = ax2.twinx()
        
        # Get child AoA
        child_aoa_x = None
        if w in child_interp_aoa:
            child_aoa_month = child_interp_aoa[w]
            if months_arr.size > 1:
                child_aoa_x = (child_aoa_month - months_arr.min()) / (months_arr.max() - months_arr.min())
                child_aoa_x = np.clip(child_aoa_x, 0.0, 1.0)
        else:
            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                first_valid_val = child_curve[child_mask][0]
                if first_valid_val >= 0.5:
                    child_aoa_x = 0.0
        
        if child_aoa_x is not None:
            # Normalize surprisal curves
            if thr_small is not None and np.isfinite(s_small).any():
                s_small_min = np.nanmin(s_small)
                s_small_range = np.nanmax(s_small) - s_small_min
                if s_small_range > 0:
                    s_small_norm = (s_small - thr_small) / s_small_range + 0.5
                else:
                    s_small_norm = np.full_like(s_small, 0.5)
                x_small_norm = normalize_x_aligned(steps_small_arr, idx_cross_small, child_aoa_x)
                ax2.plot(x_small_norm, s_small_norm, color='C0', linewidth=2, label='gpt2-small')
            
            if thr_medium is not None and np.isfinite(s_medium).any():
                s_medium_min = np.nanmin(s_medium)
                s_medium_range = np.nanmax(s_medium) - s_medium_min
                if s_medium_range > 0:
                    s_medium_norm = (s_medium - thr_medium) / s_medium_range + 0.5
                else:
                    s_medium_norm = np.full_like(s_medium, 0.5)
                x_medium_norm = normalize_x_aligned(steps_medium_arr, idx_cross_medium, child_aoa_x)
                ax2.plot(x_medium_norm, s_medium_norm, color='C1', linewidth=2, label='gpt2-medium')
            
            child_mask = np.isfinite(child_curve)
            if child_mask.any():
                ax2_twin.plot(months_norm[child_mask], child_curve[child_mask], 
                            marker="o", color="green", markersize=6, linewidth=2, label='children')
        
        ax2.set_ylim(1.0, 0.0)
        ax2.set_xlim(0.0, 1.0)
        ax2_twin.set_ylim(0.0, 1.0)
        ax2.tick_params(labelsize=16)
        ax2_twin.tick_params(labelsize=18)
        
        if col_idx == 0:
            ax2.set_ylabel("Normalized surprisal", fontsize=26)
        if col_idx == 2:
            ax2_twin.set_ylabel("Proportion producing", fontsize=26)
        if col_idx != 0:
            ax2.set_yticklabels([])
        if col_idx != 2:
            ax2_twin.set_yticklabels([])
    
    axes[0, 1].set_xlabel("Training step", fontsize=26)
    axes[1, 1].set_xlabel("Age (months)", fontsize=26)
    axes[2, 1].set_xlabel("Normalized timeline", fontsize=26)
    
    handles = [plt.Line2D([0], [0], color='C0', linewidth=2, label='gpt2-small'),
               plt.Line2D([0], [0], color='C1', linewidth=2, label='gpt2-medium'),
               plt.Line2D([0], [0], color='green', marker='o', linewidth=2, label='children')]
    fig.legend(handles=handles, loc='upper center', ncol=3, 
              bbox_to_anchor=(0.5, 0.98), fontsize=26, frameon=False)
    
    three_words_path = os.path.join("figs_paper", "gpt_human_misalignment.png")
    fig.savefig(three_words_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Saved misalignment plot to {three_words_path}")


if __name__ == "__main__":
    main()
