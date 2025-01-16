import torch 

class attack_prediction_confidence_based_mia(metric_attack):

    def run_attack(attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        probabilities = scipy.special.softmax(predictions, axis=1)
        max_probabilities = np.max(probabilities, axis=1)
        self.membership_pred = torch.tensor( (max_probabilities >= attack_options['confidence_thr']).astype(int) )