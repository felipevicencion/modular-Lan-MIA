import sys
import argparse
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackPerplexity(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(len(attack_dataset))
        perplexity_scores = []
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            # input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                inputs = ex['input_ids'].unsqueeze(0).to(self.device) 
                outputs = self.model(inputs, labels=inputs)
            _, logits = outputs[:2]
            inputs = inputs[0][1:].unsqueeze(-1)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=inputs).squeeze(-1)
            perplexity_score = torch.exp(-torch.mean(token_log_probs)).item()
            perplexity_scores.append(perplexity_score)
        print(f"perplexity_scores = {perplexity_scores}")
        if self.num_score_thr > 0:
            self.score_thr = torch.linspace(min(perplexity_scores), max(perplexity_scores), steps=self.num_score_thr).tolist()
        self.membership_pred = self.thresholds_comp(perplexity_scores, self.score_thr, "le") # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
        # self.membership_pred[ind_ex] = int(perplexity_score <= self.score_thr)