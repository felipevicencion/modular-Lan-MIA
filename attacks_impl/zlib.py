import torch 
import sys
sys.path.insert(0, '..')
from attacks import *

class attack_zlib(metric_attack):

    def run_attack(attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.score_thr = attack_options['score_thr']
        self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            ll = get_ll(ex["text"], self.model, self.tokenizer, self.device)[1]
            zlib_score = ll / len(zlib.compress(bytes(ex["text"], "utf-8")))
            # print(f"zlib_score = {zlib_score}")
            self.membership_pred[ind_ex] = int(zlib_score >= self.score_thr)