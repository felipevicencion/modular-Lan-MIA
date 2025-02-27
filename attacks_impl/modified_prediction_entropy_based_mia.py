import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackModifiedPredictionEntropyBasedMia(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        modified_sample_entropies = -np.sum( probabilities * np.log10(1 - probabilities), axis=1) 
        label_ids = predictions_output.label_ids
        py = probabilities[np.arange(probabilities.shape[0]), label_ids]
        modified_entropy = modified_sample_entropies + py * np.log10(1-py) - (1-py) * np.log10(py)
        self.membership_pred = self.thresholds_comp(modified_entropy, self.score_thr) # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
