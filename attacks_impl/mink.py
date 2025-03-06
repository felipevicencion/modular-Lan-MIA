import sys
import argparse
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackMink(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)
        attack_options_parser = argparse.ArgumentParser(description="mink options parser")
        attack_options_parser.add_argument("-mink_ratio", "--mink-ratio", type=float, help="percentage determining the count of lowest probable tokens considered for mink score", default=0.2)
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.mink_ratio = attack_args.mink_ratio

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        mink_scores = []
        for _, ex in enumerate(tqdm(attack_dataset)):
            with torch.no_grad():
                inputs = ex['input_ids'].unsqueeze(0).to(self.device) 
                outputs = self.model(inputs, labels=inputs)
            _, logits = outputs[:2]
            inputs = inputs[0][1:].unsqueeze(-1)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=inputs).squeeze(-1)
            k_length = int(len(token_log_probs) * self.mink_ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            mink_score = np.mean(topk).item()
            mink_scores.append(mink_score)
        print(f"mink_scores = {mink_scores}")
        thrs = self.score_thr
        if self.num_score_thr > 0:
            thrs = torch.linspace(min(mink_scores), max(mink_scores), steps=self.num_score_thr).tolist()
        self.membership_pred = self.thresholds_comp(mink_scores, thrs, "ge") # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
        