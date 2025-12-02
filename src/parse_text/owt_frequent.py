import argparse
import re
import warnings
from collections import Counter
import urllib3
from urllib3.exceptions import NotOpenSSLWarning
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
urllib3.disable_warnings(NotOpenSSLWarning)


def get_word_frequencies(max_docs: int, dataset: str) -> Counter:
    ds = load_dataset(dataset, split="train", streaming=True)
    word_counter = Counter()
    docs_scanned = 0
    for item in tqdm(ds):
        text = item.get("text") or ""
        words = re.findall(r"[A-Za-z]+", text.lower())
        if words:
            word_counter.update(words)
        docs_scanned += 1
        if docs_scanned >= max_docs:
            break
    return word_counter


def main():
    parser = argparse.ArgumentParser(description="Output top-K frequent words from a text dataset.")
    parser.add_argument("--dataset_name", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--max_docs", type=int, default=10_000_000)
    parser.add_argument("--output_txt", type=str, required=True)
    args = parser.parse_args()

    word_counter = get_word_frequencies(
        max_docs=args.max_docs,
        dataset=args.dataset_name,
    )
    most_common = word_counter.most_common(args.top_k)

    with open(args.output_txt, "w") as f:
        for word, count in most_common:
            f.write(f"{word}\t{count}\n")


if __name__ == "__main__":
    main()
