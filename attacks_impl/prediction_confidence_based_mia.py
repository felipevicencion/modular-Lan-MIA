import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionConfidenceBasedMia(ThresholdAttack):

    def __init__(self, attack_name, num_trial, unknown_argss):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        max_probabilities = np.max(probabilities, axis=1)
        self.membership_pred = torch.tensor( (max_probabilities >= attack_options['confidence_thr']).astype(int) )