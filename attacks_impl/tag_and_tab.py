import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '..')
from attacks import *
import spacy
import numpy as np
from pathlib import Path
import re, string
from transformers import AutoTokenizer

from attacks_impl.tnt_helpers import (
    bottom_k_entropy_words,
    create_line_to_top_words_map,
    create_entropy_map_func,
)

MAX_LEN_LINE_GENERATE = 40
MIN_LEN_LINE_GENERATE = 7

class AttackTagAndTab(ThresholdAttack):

    def __init__(self, input_args, k_keywords=4):
        super().__init__(input_args)
        self.k_keywords = k_keywords

        tnt_parser = argparse.ArgumentParser(description="Tag&Tab options parser")
        tnt_parser.add_argument(
            "-split", "--split",
            type=str,
            default="train",
            help="Dataset split to use (e.g., train, WikiMIA_length64)",
        )
        tnt_args, _ = tnt_parser.parse_known_args(input_args)
        self.split = tnt_args.split

    def select_keywords(self, text):
        words = text.split()
        if len(words) <= self.k_keywords:
            return words
        sorted_words = sorted(words, key=len, reverse=True)
        return sorted_words[:self.k_keywords]

    def getPerplexityScore(self, sentence, model, tokenizer, gpu):        
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(
            gpu).long()  # Ensure the tensor is of type Long
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        all_prob = []
        input_ids_processed = input_ids[0][1:]
        for i, token_id in enumerate(input_ids_processed):
            probability = probabilities[0, i, token_id].item()
            all_prob.append(probability)
        return torch.exp(loss).item(), all_prob, logits, input_ids_processed
    
    #From Tag&tab's github
    def compute_tag_scores(self, eval_dataset, trainer, entropy_map, nlp_spacy, tokenizer):
        scores = []
        model = trainer.model

        model.eval()

        for item in eval_dataset:
            if "input_ids" in item:
                text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            else:
                text = item.get("text", "")
            if not text:
                scores.append(0.0)
                continue

            prep, all_prob, logits, input_ids_processed = self.getPerplexityScore(text, model, tokenizer, gpu=model.device)

            try:                
                # Tokenize + normalize to match helper normalization
                raw_tokens = tokenizer.tokenize(text)
                # remove tokenizer artifacts (e.g., '▁') and punctuation, lowercase
                decoded_tokens = [t.replace("▁", "") for t in raw_tokens]
                norm_tokens = [re.sub(r"\s+", "", token.lower().translate(str.maketrans("", "", string.punctuation)))
                            for token in decoded_tokens]
                concatenated_tokens = "".join(norm_tokens)

                # Create the map for the Top-K entropy Keywords
                line_to_top_words_map, sentences = create_line_to_top_words_map(
                    text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, self.k_keywords, nlp_spacy
                )

                # Define the values of k you want to iterate over
                k_values = list(range(1, self.k_keywords + 1))
                all_bottom_k_words = {}

                for line_num, top_words in line_to_top_words_map.items():
                    bottom_k_words = bottom_k_entropy_words(" ".join(top_words), entropy_map, max(k_values))
                    all_bottom_k_words[line_num] = bottom_k_words

                # Intermediate storage for results
                intermediate_results = {
                    "relevant_log_probs": [],
                    "relevant_log_probs_one_token": [],
                }

                # This loop finds the indexes of the keywords tokens and calculate them
                for line_num, bottom_k_words in all_bottom_k_words.items():
                    for i, word in enumerate(bottom_k_words):
                        if word in concatenated_tokens:
                            start_index = concatenated_tokens.find(word)
                            end_index = start_index + len(word)
                            start_token_index = end_token_index = None
                            current_length = 0
                            for j, token in enumerate(norm_tokens):
                                current_length += len(token)
                                if current_length > start_index and start_token_index is None:
                                    start_token_index = j
                                if current_length >= end_index:
                                    end_token_index = j
                                    break
                            if start_token_index is not None and end_token_index is not None:
                                if start_token_index < len(all_prob):
                                    intermediate_results["relevant_log_probs_one_token"].append((i, all_prob[start_token_index]))
                                for idx in range(start_token_index, end_token_index + 1):
                                    if idx < len(all_prob):
                                        intermediate_results["relevant_log_probs"].append((i, all_prob[idx]))

                relevant_log_probs = [val for i, val in intermediate_results["relevant_log_probs"] if i < self.k_keywords]
                if relevant_log_probs:
                    sentence_log_likelihood = np.mean(relevant_log_probs)
                    scores.append(sentence_log_likelihood)
                else:
                    scores.append(np.mean(all_prob) if all_prob else 0.0)

            except Exception as e:
                # Safe fallback if an edge-case sample fails parsing
                scores.append(np.mean(all_prob) if all_prob else 0.0)
            

        return scores

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        wdataset = attack_options['wdataset']
        split = attack_options.get("split") or self.split

        dataset_identifier = wdataset.dataset_name if wdataset.dataset_name is not None else wdataset.dataset_path
        print(f"Generating or fetching entropy map for: {dataset_identifier}...")

        #Get or load the entropy map
        entropy_map = create_entropy_map_func(dataset_identifier, "Dataset", "validation", split=split)

        try:
            nlp_spacy = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            nlp_spacy = spacy.load("en_core_web_sm")

        tokenizer = (
                    getattr(trainer, "processing_class", None)
                    or getattr(trainer, "tokenizer", None)
                    or getattr(wdataset, "tokenizer", None)
                )
        if tokenizer is None:
            model_name = (
                getattr(wdataset, "tokenizer_name", None)
                or getattr(trainer.model.config, "_name_or_path", None)
                or "roberta-base"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        scores = self.compute_tag_scores(attack_dataset, trainer, entropy_map, nlp_spacy, tokenizer=tokenizer)
        
        if self.num_score_thr and self.num_score_thr > 0 and len(scores) > 0:
            thrs = torch.linspace(min(scores), max(scores), steps=self.num_score_thr).tolist()
            self.score_thr = thrs
        else:
            thrs = self.score_thr

        self.membership_pred = self.thresholds_comp(scores, thrs, direction="ge")
        self.plot_roc_curves(
            self.membership_pred,
            membership_true,
            f"{self.output_dir}/{self.output_name}"
        )
