import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score, precision_score
import numpy as np
import torch
import datasets
from tqdm import tqdm
from utils import *    
from dataset import *
import scipy
import zlib
    
class Attack:

    def __init__(self, attack_name, num_trials):
            self.attack_name = attack_name
            self.num_trials = num_trials
            self.num_trials_per_cat = int( 0.5 * num_trials )
            self.mia_metrics = {}

    def calculate_mia_metrics(self, true_labels, predicted_labels):
        # tensors are of type long (for classification)
        true_labels = true_labels.long()
        predicted_labels = predicted_labels.long()

        tp = torch.sum((true_labels == 1) & (predicted_labels == 1)).item()
        fp = torch.sum((true_labels == 0) & (predicted_labels == 1)).item()
        tn = torch.sum((true_labels == 0) & (predicted_labels == 0)).item()
        fn = torch.sum((true_labels == 1) & (predicted_labels == 0)).item()
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
        tpr = recall  
        adv = tpr - fpr
        metrics = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall (TPR)": recall,
            "F1 Score": f1,
            "FPR": fpr,
            "TPR": tpr,
            "Adv": adv
        }
        return metrics

    def print_mia_metrics(self):
        for metric, value in self.mia_metrics.items():
            print(f"{metric}: {value:.4f}")

    def run_attack(self, attack_dataset, membership_true, attack_options):
        raise NotImplementedError("Attack must implement the run_attack method.")
            
    def perform_attack(self, wdataset, attack_options):

        train_data, test_data = wdataset.get_train_test()
        # attack_options is a dictionary containing additional inputs and parameters required to performa MIAs such as thresold values and other
        if self.attack_name not in ['recall', 'zlib', 'mink']:
            train_data, test_data = tokenize_data(attack_options['tokenizer'], train_data, test_data, attack_options['max_sequence_length'])
        shf_train_data = train_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        shf_test_data = test_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        attack_dataset = datasets.concatenate_datasets([shf_train_data, shf_test_data])
        membership_true = torch.cat((torch.ones(self.num_trials_per_cat), torch.zeros(self.num_trials_per_cat)))
        self.run_attack(attack_dataset, membership_true, attack_options)
        self.mia_metrics = self.calculate_mia_metrics(membership_true, self.membership_pred)
        self.print_mia_metrics()
        return self.mia_metrics

# subclass representing shadow-type attacks
class ShadowAttack(Attack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

# subclass representing metric-type attacks
class MetricAttack(Attack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)

# subclass of metric attacks based on thresholding
class ThresholdAttack(MetricAttack):

    def __init__(self, attack_name, num_trials):
        super().__init__(attack_name, num_trials)