import torch
import sys
sys.path.insert(0, '..')
from attacks import *

class attack_prediction_loss_based_mia(metric_attack):

    def run_attack(attack_dataset, membership_true, attack_options):
        # find average loss over iterations and over samples
        # log_history = trainer.state.log_history
        # training_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
        # training_loss_mean = np.mean(training_loss)
    
        # find average loss over samples in last iteration
        # last_record = trainer.state.log_history[-1]
        # print(last_record)
        # training_loss = last_record['train_loss']
    
        # evaluating the average training loss at last iteration
        trainer = attack_options['trainer']
        train_loss_result = trainer.evaluate(eval_dataset=trainer.train_dataset)
        training_loss = train_loss_result.get("eval_loss", None)
    
        predictions = trainer.predict(attack_dataset)
    
        # makes sense for classification tasks only
        logits = predictions.predictions
        labels = predictions.label_ids
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  # using `reduction='none'` to get per-sample losses
        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)
    
        # calculate per-sample loss
        per_sample_loss = criterion(logits_tensor, labels_tensor)
        self.membership_pred = (per_sample_loss < training_loss).int()
        
        