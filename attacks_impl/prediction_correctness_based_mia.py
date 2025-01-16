import torch

class attack_prediction_correctness_based_mia(metric_attack):

    def run_attack(attack_dataset, membership_true, attack_options):
        trainer = attack_options['trainer']
        predictions_output = trainer.predict(attack_dataset)
        predictions = predictions_output.predictions
        label_ids = predictions_output.label_ids
        predicted_labels = np.argmax(predictions, axis=1)
        self.membership_pred = torch.tensor( (predicted_labels == label_ids).astype(int) )
    