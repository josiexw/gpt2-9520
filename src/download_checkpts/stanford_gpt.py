import os
import argparse
import torch
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
from huggingface_hub.utils import RevisionNotFoundError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="stanford-gpt2-medium-a")
    parser.add_argument("--out_dir", type=str, default="checkpoints_stanford-gpt2-medium-a")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    labels, label_type = get_checkpoint_labels(args.model_name)
    num_ckpts = len(labels)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint label type: {label_type}")
    print(f"Total checkpoints in table: {num_ckpts}")

    start = max(args.start, 0)
    end = num_ckpts if args.end is None else min(args.end, num_ckpts)
    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}")

    target_successes = (end - start + args.stride - 1) // args.stride
    print(f"Attempting checkpoints indices in [{start}, {end}) with stride {args.stride}")
    print(f"Target successful checkpoints: {target_successes}")

    success_log_path = os.path.join(args.out_dir, "checkpoint_success.txt")
    failed_log_path = os.path.join(args.out_dir, "checkpoint_failed.txt")

    successful = []
    failed = []

    with open(success_log_path, "w") as f_success, open(failed_log_path, "w") as f_failed:
        f_success.write("idx,label_type,label,filename\n")
        f_failed.write("idx,label_type,label,reason\n")

        idx = start
        num_success = 0

        while idx < end and num_success < target_successes:
            label = labels[idx]
            fname = f"ckpt_idx{idx:04d}_{label_type}{label}.pt"
            out_path = os.path.join(args.out_dir, fname)

            if os.path.exists(out_path):
                print(f"[{idx}] Skipping (already exists): {out_path}")
                successful.append((idx, label))
                f_success.write(f"{idx},{label_type},{label},{fname}\n")
                num_success += 1
                idx += args.stride
                continue

            print(f"[{idx}] Loading checkpoint (label={label_type} {label})...")

            try:
                model = HookedTransformer.from_pretrained_no_processing(
                    args.model_name,
                    checkpoint_index=idx,
                    device=args.device,
                    dtype=dtype,
                )
            except RevisionNotFoundError as e:
                print(f"[{idx}] SKIP: Revision not found on Hugging Face: {e}")
                failed.append((idx, label, "RevisionNotFoundError"))
                f_failed.write(f"{idx},{label_type},{label},RevisionNotFoundError\n")
                idx += 1
                continue
            except OSError as e:
                msg = str(e)
                if "not a valid git identifier" in msg:
                    print(f"[{idx}] SKIP: Invalid git revision for this checkpoint: {msg}")
                    failed.append((idx, label, "InvalidRevision"))
                    f_failed.write(f"{idx},{label_type},{label},InvalidRevision\n")
                    idx += 1
                    continue
                else:
                    print(f"[{idx}] ERROR: OSError when loading checkpoint: {msg}")
                    failed.append((idx, label, "OSError"))
                    f_failed.write(f"{idx},{label_type},{label},OSError\n")
                    idx += 1
                    continue
            except Exception as e:
                print(f"[{idx}] ERROR: Unexpected exception: {e}")
                failed.append((idx, label, type(e).__name__))
                f_failed.write(f"{idx},{label_type},{label},{type(e).__name__}\n")
                idx += 1
                continue

            print(f"[{idx}] Saving state dict to: {out_path}")
            torch.save(model.state_dict(), out_path)

            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            successful.append((idx, label))
            f_success.write(f"{idx},{label_type},{label},{fname}\n")
            num_success += 1
            idx += args.stride

    print("\nSummary")
    print(f"  Target successful checkpoints: {target_successes}")
    print(f"  Actual successful checkpoints: {len(successful)}")
    print(f"  Failed / skipped checkpoints: {len(failed)}")
    print(f"  Success log: {success_log_path}")
    print(f"  Failed log: {failed_log_path}")


if __name__ == "__main__":
    main()
