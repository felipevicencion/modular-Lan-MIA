import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class AttackPredictionLossBasedMia(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)
        attack_options_parser = argparse.ArgumentParser(description="prediction loss based mia options parser")
        attack_options_parser.add_argument("-use_average_training_loss", "--use-average-training-loss", type=bool, help="Whether the algorithm is provided with score threshold or if average training loss should be used", default=False)
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.use_average_training_loss = attack_args.use_average_training_loss

    def run_attack(self, attack_dataset, membership_true, attack_options):
    
        # evaluating the average training loss at the best iteration 
        trainer = attack_options['trainer']
        if self.use_average_training_loss:
            train_loss_result = trainer.evaluate(eval_dataset=trainer.train_dataset)
            self.score_thr = train_loss_result.get("eval_loss", None)
    
        predictions = trainer.predict(attack_dataset)
    
        # makes sense for classification tasks only
        logits = predictions.predictions
        labels = predictions.label_ids
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # using `reduction='none'` to get per-sample losses
        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)
    
        # calculate per-sample loss
        per_sample_loss = criterion(logits_tensor, labels_tensor)
        self.membership_pred = self.thresholds_comp(per_sample_loss, self.score_thr) # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
        
        