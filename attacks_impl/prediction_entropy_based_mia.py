import torch 
import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionEntropyBasedMia(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        sample_entropies = np.apply_along_axis(scipy.stats.entropy, axis=1, arr=probabilities, base=10)
        self.membership_pred = self.thresholds_comp(sample_entropies, self.score_thr) # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
        # self.membership_pred = torch.tensor( (sample_entropies <= attack_options['confidence_thr']).astype(int) )