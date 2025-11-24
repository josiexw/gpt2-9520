import argparse
import pickle
import re
from collections import Counter, defaultdict
from typing import List, Dict
import spacy
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

K = 5000
X = 30
TOKEN_LIMIT = 200000

COSMOPEDIA_SUBSETS = [
    "auto_math_text",
    "khanacademy",
    "openstax",
    "stanford",
    "stories",
    "web_samples_v1",
    "web_samples_v2",
    "wikihow",
]

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def sample_cosmopedia_texts(
    token_limit: int = TOKEN_LIMIT,
    subsets: List[str] = None,
) -> List[str]:
    if subsets is None:
        subsets = COSMOPEDIA_SUBSETS

    per_subset_limit = max(token_limit // len(subsets), 1)
    texts: List[str] = []
    total_tokens = 0

    for subset in subsets:
        subset_tokens = 0
        ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True,)
        for item in ds:
            text = item["text"]
            words = count_words(text)
            n = len(words)
            if n == 0:
                continue

            texts.append(text)
            subset_tokens += n
            total_tokens += n

            if subset_tokens >= per_subset_limit or total_tokens >= token_limit:
                break

        if total_tokens >= token_limit:
            break

    print(f"Collected {len(texts)} documents, approx {total_tokens} word tokens.")
    return texts


def get_word_frequencies_from_texts(texts: List[str]) -> Counter:
    word_counter = Counter()
    for text in tqdm(texts, desc="Counting word frequencies"):
        words = count_words(text)
        word_counter.update(words)
    return word_counter


def is_single_bpe_token(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return len(ids) == 1


def find_simple_words(word_counter: Counter, k: int = K, char_limit: int = 5):
    frequent_words = word_counter.most_common(k)
    simple_words = [word for word, _ in frequent_words if len(word) <= char_limit and is_single_bpe_token(word)]
    return set(simple_words)


def collect_contexts_from_texts(
    texts: List[str],
    simple_words: set,
    max_context: int = X,
    window_size: int = 10,
) -> Dict[str, List[str]]:
    contexts = defaultdict(set)
    completed_words = set()

    def done() -> bool:
        return len(completed_words) == len(simple_words)

    for doc in tqdm(
        nlp.pipe(texts, batch_size=32),
        total=len(texts),
        desc="Collecting contexts",
    ):
        for sent in doc.sents:
            words = [tok.text.lower() for tok in sent if tok.is_alpha]
            for i, w in enumerate(words):
                if w in simple_words and len(contexts[w]) < max_context:
                    start = max(0, i - window_size)
                    prefix = " ".join(words[start:i])

                    if prefix:
                        contexts[w].add(prefix)

                    if len(contexts[w]) >= max_context:
                        completed_words.add(w)
        if done():
            break

    return {w: list(contexts[w]) for w in simple_words if len(contexts[w]) >= max_context}


def main():
    parser = argparse.ArgumentParser(
        description="Collect contexts for simple words from a ~200k-token subset of Cosmopedia."
    )
    parser.add_argument("--K", type=int, default=K, help="Number of most frequent words to consider.")
    parser.add_argument("--X", type=int, default=X, help="Number of contexts to collect per simple word.")
    parser.add_argument("--token_limit", type=int, default=TOKEN_LIMIT, help="Approximate total number of word tokens to sample from Cosmopedia.")
    parser.add_argument("--char_limit", type=int, default=5, help="Maximum character length for a word to be considered simple.",)
    parser.add_argument("--subsets", type=str, default=",".join(COSMOPEDIA_SUBSETS), help="Comma-separated list of Cosmopedia subsets to sample from.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the collected contexts as a pickle file.")
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    texts = sample_cosmopedia_texts(
        token_limit=args.token_limit,
        subsets=subsets,
    )

    word_counter = get_word_frequencies_from_texts(texts)
    simple_words = find_simple_words(word_counter, k=args.K, char_limit=args.char_limit)
    print(f"Found {len(simple_words)} simple words.")

    contexts = collect_contexts_from_texts(texts, simple_words, max_context=args.X,)
    print(f"{len(contexts)} simple words have at least {args.X} contexts.")

    with open(args.output_path, "wb") as f:
        pickle.dump(contexts, f)

if __name__ == "__main__":
    main()

# python src/parse_text/cosmopedia.py \
#   --output_path data/contexts_cosmopedia.pkl
