import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionCorrectnessBasedMia(MetricAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        with torch.no_grad():
            trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        label_ids = predictions_output.label_ids
        predicted_labels = np.argmax(predictions, axis=1)
        self.membership_pred = torch.tensor( (predicted_labels == label_ids).astype(int) )
    