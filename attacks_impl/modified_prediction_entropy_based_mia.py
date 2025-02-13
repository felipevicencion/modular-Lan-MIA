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
        modified_sample_entropies = np.apply_along_axis(scipy.stats.entropy, axis=1, arr=probabilities, base=10)
        label_ids = predictions_output.label_ids
        py = probabilities[np.arange(probabilities.shape[0]), label_ids]
        modified_entropy = modified_sample_entropies + (1+2*py) * np.log(py)
        self.membership_pred = torch.tensor( (modified_entropy <= attack_options['confidence_thr']).astype(int) )