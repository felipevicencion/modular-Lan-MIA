import torch 
import sys
sys.path.insert(0, '..')
from attacks import *

class attack_modified_prediction_entropy_based_mia(metric_attack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

    def run_attack(self, attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        modified_sample_entropies = np.apply_along_axis(scipy.stats.entropy, axis=1, arr=probabilities, base=10)
        label_ids = predictions_output.label_ids
        py = probabilities[np.arange(probabilities.shape[0]), label_ids]
        modified_entropy = modified_entropy + (1+2*py) * np.log(py)
        self.membership_pred = torch.tensor( (modified_entropy <= attack_options['confidence_thr']).astype(int) )