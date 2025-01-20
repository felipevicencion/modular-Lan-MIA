import torch 
import sys
sys.path.insert(0, '..')
from attacks import *

class attack_prediction_entropy_based_mia(metric_attack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        entropy = scipy.stats.entropy(pred_vector, base=10)
        self.membership_pred = torch.tensor( (entropy <= attack_options['confidence_thr']).astype(int) )