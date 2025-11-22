import os
import argparse
import math
import pickle
from typing import Dict, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
from huggingface_hub.utils import RevisionNotFoundError


class Experiment:
    def __init__(self,
                 model: HookedTransformer,
                 batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.device = str(self.model.cfg.device)
        self.model.eval()
        self.model.tokenizer.padding_side = "right"

    def compute_batches(self,
                        prefixes: List[str],
                        word: str):
        full_text_l = [prefix.rstrip() + " " + word for prefix in prefixes]
        for i in range(0, len(full_text_l), self.batch_size):
            yield {
                "prefixes": prefixes[i:i + self.batch_size],
                "full_texts": full_text_l[i:i + self.batch_size],
            }

    def compute_output(self,
                       batch: Dict[str, List[str]]):
        prefixes, full_texts = batch["prefixes"], batch["full_texts"]
        batch_size = len(prefixes)

        prefix_lens = [
            len(self.model.to_tokens(prefix, prepend_bos=False)[0])
            for prefix in prefixes
        ]
        full_lens = [
            len(self.model.to_tokens(full_text, prepend_bos=False)[0])
            for full_text in full_texts
        ]
        num_word_tokens = [
            full_len - pref_len
            for pref_len, full_len in zip(prefix_lens, full_lens)
        ]

        full_tokens = self.model.to_tokens(full_texts, prepend_bos=False)

        with torch.no_grad():
            logits, cache = self.model.run_with_cache(full_tokens)

        total_logprob_e = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            if num_word_tokens[i] <= 0:
                continue
            word_token_positions = range(prefix_lens[i], full_lens[i])
            for position in word_token_positions:
                if position == 0:
                    continue
                logits_pos = logits[i, position - 1]
                log_probs = F.log_softmax(logits_pos, dim=-1)
                token_id = int(full_tokens[i, position].item())
                total_logprob_e[i] += log_probs[token_id]

        surprisal_bits = (-total_logprob_e / math.log(2.0)).tolist()

        n_layers = self.model.cfg.n_layers
        layer_avg_attn = torch.zeros(batch_size, n_layers, device=self.device)

        for i in range(batch_size):
            if num_word_tokens[i] <= 0:
                continue
            word_token_positions = range(prefix_lens[i], full_lens[i])
            for layer in range(n_layers):
                attn = cache["attn", layer][i]
                into_word = attn[:, :, word_token_positions]
                layer_avg_attn[i, layer] = into_word.mean()

        del cache
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return surprisal_bits, num_word_tokens, layer_avg_attn.cpu()

    def compute_output_dict(self,
                            contexts: Dict[str, List[str]]):
        output_dict = {}
        words = list(contexts.keys())

        for word in tqdm(words):
            prefixes = contexts[word]
            surprisals = []
            token_counts = []
            layer_attn_vals = []

            for batch in self.compute_batches(prefixes, word):
                surprisal_bits, num_word_tokens, layer_avg_attn = self.compute_output(batch)
                surprisals.extend(surprisal_bits)
                token_counts.extend(num_word_tokens)
                layer_attn_vals.append(layer_avg_attn)

            if len(surprisals) == 0:
                continue

            layer_attn_vals = torch.cat(layer_attn_vals, dim=0)

            avg = sum(surprisals) / len(surprisals)
            per_token_vals = [
                s / t if t > 0 else float("nan")
                for s, t in zip(surprisals, token_counts)
            ]
            per_token_vals = [x for x in per_token_vals if not math.isnan(x)]
            if len(per_token_vals) > 0:
                avg_per_token = sum(per_token_vals) / len(per_token_vals)
            else:
                avg_per_token = float("nan")

            avg_layer_attn = layer_attn_vals.mean(dim=0).tolist()

            output_dict[word] = {
                "avg_surprisal": avg,
                "avg_surprisal_per_token": avg_per_token,
                "surprisals_list": surprisals,
                "avg_layer_attn": avg_layer_attn,
            }

        return output_dict


def resolve_model_name(model_size: str) -> str:
    if model_size == "small":
        return "stanford-gpt2-small-a"
    elif model_size == "medium":
        return "stanford-gpt2-medium-a"
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute word surprisal and avg attention per layer across GPT-2 checkpoints."
    )
    parser.add_argument("--model_size", type=str, choices=["small", "medium"], required=True)
    parser.add_argument("--contexts_pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_stride", type=int, default=6)
    parser.add_argument("--checkpoint_start", type=int, default=0)
    parser.add_argument("--checkpoint_end", type=int, default=None)
    args = parser.parse_args()

    base_model_name = resolve_model_name(args.model_size)

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    labels, label_type = get_checkpoint_labels(base_model_name)
    num_ckpts = len(labels)
    start = max(args.checkpoint_start, 0)
    end = num_ckpts if args.checkpoint_end is None else min(args.checkpoint_end, num_ckpts)

    success_log_path = os.path.join(args.out_dir, "checkpoint_success.txt")
    failed_log_path = os.path.join(args.out_dir, "checkpoint_failed.txt")

    with open(args.contexts_pkl, "rb") as f:
        contexts = pickle.load(f)

    with open(success_log_path, "w") as f_success, open(failed_log_path, "w") as f_failed:
        f_success.write("idx,label_type,label,output_pkl\n")
        f_failed.write("idx,label_type,label,reason\n")

        for idx in range(start, end, args.checkpoint_stride):
            label = labels[idx]
            out_pkl = os.path.join(
                args.out_dir,
                f"results_ckpt_idx{idx:04d}_{label_type}{label}.pkl",
            )

            if os.path.exists(out_pkl):
                print(f"[{idx}] Skipping (results already exist): {out_pkl}")
                f_success.write(f"{idx},{label_type},{label},{out_pkl}\n")
                continue

            print(f"[{idx}] Loading model {base_model_name} checkpoint_index={idx} (label={label_type} {label})")

            try:
                model = HookedTransformer.from_pretrained_no_processing(
                    base_model_name,
                    checkpoint_index=idx,
                    device=device,
                    dtype=torch_dtype,
                )
            except RevisionNotFoundError as e:
                print(f"[{idx}] SKIP: Revision not found on Hugging Face: {e}")
                f_failed.write(f"{idx},{label_type},{label},RevisionNotFoundError\n")
                continue
            except OSError as e:
                msg = str(e)
                if "not a valid git identifier" in msg:
                    print(f"[{idx}] SKIP: Invalid git revision for this checkpoint: {msg}")
                    f_failed.write(f"{idx},{label_type},{label},InvalidRevision\n")
                    continue
                else:
                    print(f"[{idx}] ERROR: OSError when loading checkpoint: {msg}")
                    f_failed.write(f"{idx},{label_type},{label},OSError\n")
                    continue
            except Exception as e:
                print(f"[{idx}] ERROR: Unexpected exception: {e}")
                f_failed.write(f"{idx},{label_type},{label},{type(e).__name__}\n")
                continue

            experiment = Experiment(model=model, batch_size=args.batch_size)
            output_dict = experiment.compute_output_dict(contexts)

            with open(out_pkl, "wb") as f_out:
                pickle.dump(output_dict, f_out)

            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            f_success.write(f"{idx},{label_type},{label},{out_pkl}\n")
            print(f"[{idx}] Saved results to: {out_pkl}")

    print("Done.")


if __name__ == "__main__":
    main()

# python src/analyze_checkpts/mult_ckpt.py \
#   --model_size small \
#   --contexts_pkl contexts.pkl \
#   --out_dir stanford-gpt2-small-a_results \
#   --batch_size 8 \
#   --checkpoint_stride 12 \
#   --dtype float16
