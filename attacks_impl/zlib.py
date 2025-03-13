import sys
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackZlib(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(len(attack_dataset))
        zlib_scores = []
        for _, ex in enumerate(tqdm(attack_dataset)):
            with torch.no_grad():
                inputs = ex['input_ids'].unsqueeze(0).to(self.device) 
            # ll = get_ll(ex["text"], self.model, self.tokenizer, self.device)[1]
            ll = get_ll(inputs, self.model)[1]
            zlib_score = ll / len(zlib.compress(bytes(ex["text"], "utf-8")))
            zlib_scores.append(zlib_score)
        print(f"zlib_scores = {zlib_scores}")
        if self.num_score_thr > 0:
            self.score_thr = torch.linspace(min(zlib_scores), max(zlib_scores), steps=self.num_score_thr).tolist()
        self.membership_pred = self.thresholds_comp(zlib_scores, self.score_thr, "le") # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")