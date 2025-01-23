import sys
import argparse
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackPerplexity(ThresholdAttack):

    def __init__(self, attack_name, num_trials, unknown_args):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.score_thr = attack_options['score_thr']
        self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            _, logits = outputs[:2]
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            perplexity_score = np.exp(-np.mean(token_log_probs))
            self.membership_pred[ind_ex] = int(perplexity_score <= self.score_thr)