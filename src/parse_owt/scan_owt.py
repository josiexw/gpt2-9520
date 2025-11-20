import argparse
import json
import random
import re
import spacy

from collections import Counter, defaultdict
from datasets import load_dataset 
from tqdm import tqdm
from typing import List

K = 5000
X = 30
MAX_TOKENS_CHARS = 5
MAX_DOCS = 1000

nlp = spacy.load('en_core_web_sm')

def get_word_frequencies(max_docs=MAX_DOCS):
    dataset = load_dataset('openwebtext', split='train', streaming=True, trust_remote_code=True)
    docs_scanned = 0

    word_counter = Counter()
    for item in tqdm(dataset):
        text = item['text']
        words = re.findall(r'[A-Za-z]+', text.lower())
        word_counter.update(words)

        docs_scanned += 1
        if docs_scanned >= max_docs:
            break

    return word_counter 

# We define frequent words as those that appear in the top K most common words 
# We define simple words as those with length <= MAX_TOKENS_CHARS and appear in the top K most common words

def find_simple_words(word_counter: Counter, 
                      k: int = K, 
                      max_length: int = MAX_TOKENS_CHARS):
    frequent_words = word_counter.most_common(k)
    simple_words = [word for word, _ in frequent_words if len(word) <= max_length]
    return set(simple_words) 

def collect_contexts(simple_words: set,
                     max_context: int = X,
                     max_docs: int = MAX_DOCS,
                     window_size: int = 10):
    def done():
        return len(completed_words) == len(simple_words)
    
    dataset = load_dataset('openwebtext', split='train', streaming=True, trust_remote_code=True)
    docs_scanned = 0
    contexts = defaultdict(set)
    completed_words = set()

    for doc in nlp.pipe((item['text'] for item in dataset), batch_size=32):
        for sentence in doc.sents:
            words = [token.text.lower() for token in sentence if token.is_alpha]
            for i, word in enumerate(words):
                if word in simple_words and len(contexts[word]) < max_context:
                    start = max(0, i - window_size)
                    prefix = ' '.join(words[start:i])

                    if prefix:
                        contexts[word].add(prefix)
                    
                    if len(contexts[word]) >= max_context:
                        completed_words.add(word)
        
        docs_scanned += 1
        if docs_scanned >= max_docs or done():
            break
    
    return {word: list(contexts[word]) for word in simple_words if len(contexts[word]) >= max_context}

def main():
    parser = argparse.ArgumentParser(description='Collect sentence pairs for simple words from OpenWebText.')
    parser.add_argument('--K', type=int, default=K, help='Number of most frequent words to consider.')
    parser.add_argument('--X', type=int, default=X, help='Number of contexts to consider per simple word.')
    parser.add_argument('--max_tokens_chars', type=int, default=MAX_TOKENS_CHARS, help='Maximum character length of a simple word.')
    parser.add_argument('--max_docs', type=int, default=MAX_DOCS, help='Maximum number of documents to scan.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the collected contexts as a JSON file.')
    args = parser.parse_args()

    word_counter = get_word_frequencies(max_docs=args.max_docs)
    simple_words = find_simple_words(word_counter, k=args.K, max_length=args.max_tokens_chars)
    contexts = collect_contexts(simple_words, max_context=args.X, max_docs=args.max_docs)

    with open(args.output_path, 'w') as f:
        json.dump(contexts, f)

if __name__ == '__main__':
    main()