import sys
import argparse
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackMinkPlusPlus(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)
        attack_options_parser = argparse.ArgumentParser(description="mink options parser")
        attack_options_parser.add_argument("-mink_plus_plus_ratio", "--mink-plus-plus-ratio", type=float, help="percentage determining the count of lowest probable tokens considered for mink plus plus score", default=0.2)
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.mink_ratio = attack_args.mink_ratio

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        # self.score_thr = attack_options['score_thr']
        # self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            _, logits = outputs[:2]
            probs = torch.nn.functional.softmax(logits[0, :-1], dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            k_length = int(len(mink_plus) * self.mink_ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            mink_plus_score = np.mean(topk).item()
            self.membership_pred[ind_ex] = int(mink_plus_score >= self.score_thr)