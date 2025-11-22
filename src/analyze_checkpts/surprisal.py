import argparse
import math
import pickle
import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer
from tqdm import tqdm
from typing import Dict, List


class Experiment():
    def __init__(self,
                 model_name: str,
                 output_path: str,
                 batch_size: int):
        self.model_name = model_name 
        self.output_path = output_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        self.model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
        self.model.eval()
        print(f'loaded model {self.model_name} on {self.device}')
        self.model.tokenizer.padding_side = 'right'

    def compute_batches(self,
                        prefixes: List[str],
                        word: str):
        full_text_l = [prefix.rstrip() + ' ' + word for prefix in prefixes]
        for i in range(0, len(full_text_l), self.batch_size):
            yield {
                'prefixes': prefixes[i:i + self.batch_size],
                'full_texts': full_text_l[i:i + self.batch_size]
            }
    
    def compute_output(self,
                       batch: Dict[str, List[str]]):
        prefixes, full_texts = batch['prefixes'], batch['full_texts']
        batch_size = len(prefixes)

        prefix_lens = [len(self.model.to_tokens(prefix, prepend_bos=False)[0]) for prefix in prefixes]
        full_lens = [len(self.model.to_tokens(full_text, prepend_bos=False)[0]) for full_text in full_texts]
        num_word_tokens = [full - prefix for prefix, full in zip(prefix_lens, full_lens)]

        full_tokens = self.model.to_tokens(full_texts, prepend_bos=False)

        with torch.no_grad():
            logits, cache = self.model.run_with_cache(full_tokens)

        total_logprob_e = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            word_token_positions = range(prefix_lens[i], full_lens[i])
            for position in word_token_positions:
                logits_pos = logits[i, position - 1]
                log_probs = F.log_softmax(logits_pos, dim=-1)
                token_id = int(full_tokens[i, position].item())
                total_logprob_e[i] += log_probs[token_id]
        
        surprisal_bits = (- total_logprob_e / math.log(2.0)).tolist()
        
        n_layers = self.model.cfg.n_layers
        attn_maps = []

        for i in range(batch_size):
            word_token_positions = range(prefix_lens[i], full_lens[i])
            layer_maps = []

            for layer in range(n_layers):
                attn = cache['attn', layer]
                attn_avg_heads = attn[i].mean(dim=0)
                into_word = attn_avg_heads[:, word_token_positions]
                layer_maps.append(into_word)
            
            attn_maps.append(torch.stack(layer_maps, dim=0).cpu())

        del cache
        torch.cuda.empty_cache()

        return surprisal_bits, num_word_tokens, attn_maps 

    def compute_output_dict(self, 
                               contexts: Dict[str, List[str]]):
        output_dict = {}
        words = list(contexts.keys())

        for word in tqdm(words):
            prefixes = contexts[word]
            surprisals, token_counts, word_attn_maps = [], [], []
            for batch in self.compute_batches(prefixes, word):
                surprisal_bits, num_word_tokens, attn_maps = self.compute_output(batch)
                surprisals.extend(surprisal_bits)
                token_counts.extend(num_word_tokens)
                word_attn_maps.extend(attn_maps)
            
            avg = sum(surprisals) / len(surprisals) 
            per_token_vals = [s / t for s, t in zip(surprisals, token_counts)]
            avg_per_token = sum(per_token_vals) / len(per_token_vals) 

            output_dict[word] = {
                'avg_surprisal': avg, 
                'avg_surprisal_per_token': avg_per_token, 
                'surprisals_list': surprisals,
                'word_attn_maps': word_attn_maps
            }
        return output_dict 

def main():
    parser = argparse.ArgumentParser(description='Compute average surprisal and attn maps per simple word using GPT-2.')
    parser.add_argument('--model_name', type=str, default='gpt2-small', help='Name of pretrained GPT-2 model to use.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output dict as pickle file.')
    parser.add_argument('--contexts_pkl', type=str, required=True, help='Path to contexts pickle file.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for GPT-2 processing.')
    args = parser.parse_args()

    with open(args.contexts_pkl, 'rb') as f:
        contexts = pickle.load(f)
    experiment = Experiment(model_name=args.model_name, output_path=args.output_path, batch_size=args.batch_size)
    output_dict = experiment.compute_output_dict(contexts)

    with open(args.output_path, 'wb') as f:
        pickle.dump(output_dict, f)

if __name__ == '__main__':
    main()