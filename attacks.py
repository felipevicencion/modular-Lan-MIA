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
        """
        Parameters:
            true_labels (torch.Tensor): Ground truth labels as a tensor of type long. 
                                        Expected to contain binary values (0 or 1).
            predicted_labels (torch.Tensor): Predicted labels as a tensor of type long. 
                                             Expected to contain binary values (0 or 1).
        
        Returns:
            dict: A dictionary containing the following metrics:
                - TP: True Positives
                - FP: False Positives
                - TN: True Negatives
                - FN: False Negatives
                - Accuracy: Accuracy of the model
                - Precision: Positive predictive value
                - Recall (TPR)
                - F1 Score: Harmonic mean of precision and recall
                - FPR: False Positive Rate
                - TPR: True Positive Rate 
                - Adv: Advantage (TPR - FPR)
        """
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
        """
        Perform a membership inference attack (MIA) on the given dataset.
    
        Parameters:
            wdataset: A dataset object 
            attack_options (dict): A dictionary containing general options and parameters 
                                   required for the attack, such as:
                                   - 'tokenizer': Tokenizer function for text processing.
                                   - 'max_sequence_length': Maximum sequence length for tokenized inputs.
                                   - Other attack-specific parameters (e.g., thresholds).
    
        Returns:
            dict: A dictionary containing MIA metrics computed after the attack.
        """
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

    def thresholds_comp(scores, thrs):
        """
        Creates a membership matrix where each entry is 1 if the score is below the threshold, else 0.
        
        Parameters:
        scores (list or array-like): A list of scores for each example in the dataset.
        thrs (list or array-like): A vector of thresholds.
        
        Returns:
        np.ndarray: A binary membership matrix of shape (len(scores), len(thrs)).
        """
        scores = np.array(scores).reshape(-1, 1)
        thrs = np.array(thrs).reshape(1, -1)
        
        membership_matrix = (scores < thrs).astype(int) 
        
        return membership_matrix

    def plot_roc_curves(membership_matrix, membership_true, out_loc):
        mean_membership = membership_matrix.mean(axis=1)  # Average score per sample
        fpr, tpr, _ = roc_curve(membership_true, mean_membership)
        roc_auc = auc(fpr, tpr)
        
        # Full ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{out_loc}_full_roc_curve.png")
        plt.close()
    
        print(f"full ROC curve saved to {out_loc}_full_roc_curve.png")
        
        # ROC Curve up to 1% FDR (False Discovery Rate ~ False Positive Rate)
        fdr_threshold = 0.01
        mask = fpr <= fdr_threshold
        plt.figure(figsize=(8, 6))
        plt.plot(fpr[mask], tpr[mask], color='red', lw=2, label=f'ROC (up to 1% FDR)')
        plt.xlabel('False Positive Rate (FDR)')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (1% FDR)')
        plt.legend(loc='lower right')
        plt.savefig(f"{out_loc}_roc_curve_1percent_fdr.png")
        plt.close()
        
        print(f"1percent ROC curve saved to {out_loc}_roc_curve_1percent_fdr.png")
