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
        self.mink_plus_plus_ratio = attack_args.mink_plus_plus_ratio

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        mink_plus_scores = []
        for _, ex in enumerate(tqdm(attack_dataset)):
            with torch.no_grad():
                inputs = ex['input_ids'].unsqueeze(0).to(self.device) 
                outputs = self.model(inputs, labels=inputs)
            _, logits = outputs[:2]
            inputs = inputs[0][1:].unsqueeze(-1)
            probs = torch.nn.functional.softmax(logits[0, :-1], dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=inputs).squeeze(-1)
            # print(f"token_log_probs = {token_log_probs}")
            print(token_log_probs.shape)
            mu = (probs * log_probs).sum(-1)
            # print(f"mu = {mu}")
            # print(mu.shape)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
            # print(f"sigma = {sigma}")
            print(sigma.shape)
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            k_length = int(len(mink_plus) * self.mink_plus_plus_ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            mink_plus_score = np.mean(topk).item()
            mink_plus_scores.append(mink_plus_score)
        if self.num_score_thr > 0:
            self.score_thr = torch.linspace(min(mink_plus_scores), max(mink_plus_scores), steps=self.num_score_thr).tolist()
        self.membership_pred = self.thresholds_comp(mink_plus_scores, self.score_thr, "ge") # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")