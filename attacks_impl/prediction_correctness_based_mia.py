import torch
import sys
sys.path.insert(0, '..')
from attacks import *

class attack_prediction_correctness_based_mia(metric_attack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        label_ids = predictions_output.label_ids
        predicted_labels = np.argmax(predictions, axis=1)
        self.membership_pred = torch.tensor( (predicted_labels == label_ids).astype(int) )
    