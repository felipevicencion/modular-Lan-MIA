import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionConfidenceBasedMia(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        max_probabilities = np.max(probabilities, axis=1)
        # self.membership_pred = torch.tensor( (max_probabilities >= attack_options['confidence_thr']).astype(int) )
        self.membership_pred = self.thresholds_comp(1-max_probabilities, self.score_thr) # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")