import torch 
import sys
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionEntropyBasedMia(ThresholdAttack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        sample_entropies = np.apply_along_axis(scipy.stats.entropy, axis=1, arr=probabilities, base=10)
        self.membership_pred = torch.tensor( (sample_entropies <= attack_options['confidence_thr']).astype(int) )