import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score, precision_score, roc_curve, auc
import numpy as np
import torch
import datasets
from tqdm import tqdm
from utils import *    
from dataset import *
import scipy
import zlib
import matplotlib.pyplot as plt
import argparse
import time
import json
    
class Attack:

    def __init__(self, input_args):
            attack_options_parser = argparse.ArgumentParser(description="attack options parser")
            attack_options_parser.add_argument("-attack_name", "--attack-name", type=str, help = "Name of an attack to be performed", default = "prediction_loss_based_mia")
            attack_options_parser.add_argument("-num_trials", "--num-trials", type=int, help = "Number of trials of an attack to be performed", default = 500)
            attack_options_parser.add_argument("-score_thr", "--score-thr", type=str, help = "Thresholds to be used for attacks based on scores", default = 1.002)
            attack_options_parser.add_argument("-max_sequence_length", "--max-sequence-length", type=int, help = "Maximum sequence length", default = 512)
            attack_options_parser.add_argument("-output_dir", "--output-dir", type=str, help = "Location of the output directory", default = None)
            attack_options_parser.add_argument("-output_name", "--output-name", type=str, help = "Name of the output file", default = None)  
            attack_options_parser.add_argument("-confidence_thr", "--confidence-thr", type=float, help = "Confidence threshold for MIA attacks", default = 0.9)
    
            self.input_args = input_args
            attack_args, _ = attack_options_parser.parse_known_args(input_args)
            self.attack_name = attack_args.attack_name
            self.num_trials = attack_args.num_trials
            self.score_thr = parse_thresholds(str(attack_args.score_thr))
            self.max_sequence_length = attack_args.max_sequence_length
            self.output_dir = attack_args.output_dir
            self.output_name = attack_args.output_name
            if self.output_name is None:
                self.output_name = self.attack_name
            self.confidence_thr = attack_args.confidence_thr
            self.num_trials_per_cat = int( 0.5 * self.num_trials )
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
        # Check if self.mia_metrics is a dictionary of dictionaries
        if all(isinstance(value, dict) for value in self.mia_metrics.values()):
            # Case 1: Dictionary of dictionaries
            for threshold, metrics_dict in self.mia_metrics.items():
                print(f"{threshold}:")
                for metric, value in metrics_dict.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        else:
            # Case 2: Single dictionary
            print("Metrics:")
            for metric, value in self.mia_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")


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
                                   - Other attack-specific parameters (e.g., thresholds).
    
        Returns:
            dict: A dictionary containing MIA metrics computed after the attack.
        """
        train_data, test_data = wdataset.get_train_test()
        # attack_options is a dictionary containing additional inputs and parameters required to performa MIAs such as thresold values and other
        if self.attack_name not in ['recall', 'zlib', 'mink']:
            # train_data, test_data = tokenize_data(attack_options['tokenizer'], train_data, test_data, attack_options['max_sequence_length'])
            wdataset.train_data = tokenize_data(attack_options['tokenizer'], train_data, self.max_sequence_length)
            wdataset.test_data = tokenize_data(attack_options['tokenizer'], test_data, self.max_sequence_length)
            if wdataset.val_data is not None:
                wdataset.val_data = tokenize_data(attack_options['tokenizer'], wdataset.val_data, self.max_sequence_length)


        # Add an index column if not already present
        if "index" not in wdataset.train_data.column_names:
            wdataset.train_data = wdataset.train_data.add_column("index", list(range(len(wdataset.train_data))))
        if "index" not in wdataset.test_data.column_names:
            wdataset.test_data = wdataset.test_data.add_column("index", list(range(len(wdataset.test_data))))

        # Shuffle dataset
        shuffled_train = wdataset.train_data.shuffle(seed=42)
        shuffled_test = wdataset.test_data.shuffle(seed=42)

        # Select subset
        shf_train_data = shuffled_train.select(range(self.num_trials_per_cat))
        shf_test_data = shuffled_test.select(range(self.num_trials_per_cat))

        # Extract the original indices from the shuffled data
        train_shuffle_order = list(shf_train_data["index"])
        test_shuffle_order = list(shf_test_data["index"])

        # Store in dictionary
        self.attack_indices = {
            "train": train_shuffle_order,
            "test": test_shuffle_order
        }

        # Remove the index column from the datasets
        wdataset.train_data = wdataset.train_data.remove_columns(["index"])
        wdataset.test_data = wdataset.test_data.remove_columns(["index"])

        attack_dataset = datasets.concatenate_datasets([shf_train_data, shf_test_data])
        self.membership_true = torch.cat((torch.ones(self.num_trials_per_cat), torch.zeros(self.num_trials_per_cat)))
        start_time_perform_attack = time.time()
        self.run_attack(attack_dataset, self.membership_true, attack_options)
        end_time_perform_attack = time.time()
        elapsed_time = end_time_perform_attack - start_time_perform_attack
        print(f"Attack completed in {elapsed_time:.4f} seconds.")
        self.mia_metrics = self.calculate_mia_metrics(self.membership_true, self.membership_pred)
        mia_metrics_output_path = f"{self.output_dir}/{self.output_name}_mia_metrics.json"
        with open(mia_metrics_output_path, "w") as json_file:
            json.dump(self.mia_metrics, json_file, indent=4)
        print(f"MIA metrics saved to {mia_metrics_output_path}")
        self.print_mia_metrics()
        return self.mia_metrics

# subclass representing shadow-type attacks
class ShadowAttack(Attack):

    def __init__(self, input_args):
        super().__init__(input_args)

        attack_options_parser = argparse.ArgumentParser(description="shadow attack options parser")
        attack_options_parser.add_argument("--num_shadow_models", "--num-shadow-models", type=int, help = "Number of shadow models to be created", default = 2)
        # attack_options_parser.add_argument("-model_name", "--model-name", type=str, help = "Name of the pretrained NLP model", default = "roberta-base")
        # attack_options_parser.add_argument("-max_sequence_length", "--max-sequence-length", type=int, help = "Maximum sequence length", default = 512)
        # attack_options_parser.add_argument("-use_float16", "--use-float16", type=bool, help = "Whether or not to store weight in 16 bit floats", default = False)
        # attack_options_parser.add_argument("-num_epochs", "--num-epochs", type=int, help = "Number of epochs used in the fine-tuning of the model", default = 5)
        # attack_options_parser.add_argument("-per_device_train_batch_size", "--per-device-train-batch-size", type=int, help = "Per device train batch size", default = 4)
        # attack_options_parser.add_argument("-per_device_eval_batch_size", "--per-device-eval-batch-size", type=int, help = "Per device eval batch size", default = 16)
        # attack_options_parser.add_argument("-learning_rate", "--learning-rate", type=float, help = "Learning rate used in fine-tunning", default = 2e-5)
        # attack_options_parser.add_argument("-weight_decay", "--weight-decay", type=float, help = "Weight decay parameter used in fine-tunning", default = 0.01)
        # attack_options_parser.add_argument("-warmup_ratio", "--warmup-ratio", type=float, help = "Warmup ratio parameter used in fine-tunning", default = 0.1)
        # attack_options_parser.add_argument("-val_size", "--val-size", type=float, help = "Size of the validation set as ratio of the train set", default = 0.1)
        # attack_options_parser.add_argument("-num_classes", "--num-classes", type=int, help = "Number of classes for the classification task", default = 2)
     
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.num_shadow_models = attack_args.num_shadow_models
        # self.model_name = attack_args.model_name
        # self.max_sequence_length = attack_args.max_sequence_length
        # self.use_float16 = attack_args.use_float16
        # self.num_epochs = attack_args.num_epochs
        # self.per_device_train_batch_size = attack_args.per_device_train_batch_size
        # self.per_device_eval_batch_size = attack_args.per_device_eval_batch_size
        # self.learning_rate = attack_args.learning_rate
        # self.weight_decay = attack_args.weight_decay
        # self.warmup_ratio = attack_args.warmup_ratio
        # self.val_size = attack_args.val_size
        # self.num_classes = attack_args.num_classes

# subclass representing metric-type attacks
class MetricAttack(Attack):

    def __init__(self, input_args):
        super().__init__(input_args)

# subclass of metric attacks based on thresholding
class ThresholdAttack(MetricAttack):

    def __init__(self, input_args):
        super().__init__(input_args)

    def thresholds_comp(self, scores, thrs):
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

    def plot_roc_curves(self, membership_matrix, membership_true, out_loc):
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
        
        # ROC Curve up to 5% FDR (False Discovery Rate ~ False Positive Rate)
        fpr_threshold = 0.05
        mask = (fpr <= fpr_threshold)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr[mask], tpr[mask], color='red', lw=2, label=f'ROC (up to 5% FDR)')
        plt.xlim([0,fpr_threshold])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate (FDR)')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (1% FDR)')
        plt.legend(loc='lower right')
        plt.savefig(f"{out_loc}_roc_curve_5percent_fpr.png")
        plt.close()
        
        print(f"1percent ROC curve saved to {out_loc}_roc_curve_5percent_fpr.png")

    def calculate_mia_metrics(self, true_labels, predicted_labels):
        """
        Compute MIA metrics for binary classification.
        
        Handles cases where predicted_labels is either:
        - A 1D tensor of binary values.
        - A 2D tensor of shape (len(scores), len(thrs)), where each column represents predictions at a different threshold.
        
        Parameters:
            true_labels (torch.Tensor): Ground truth labels as a tensor of type long (binary values: 0 or 1).
            predicted_labels (torch.Tensor or np.ndarray): Predicted labels, either a 1D tensor or a matrix where each column 
                                                        stores predicted labels at a different threshold.
        
        Returns:
            dict: A dictionary where keys are threshold indices (or "single" if only one set of predictions is provided),
                and values contain the computed metrics.
        """
        true_labels = true_labels.long()
        
        # If predicted_labels is 1D, convert it to 2D for uniform handling
        if len(predicted_labels.shape) == 1:
            predicted_labels = predicted_labels.unsqueeze(1)  # Shape (N, 1)

        num_thresholds = predicted_labels.shape[1]
        metrics = {}

        for i in range(num_thresholds):
            preds = predicted_labels[:, i]  # Get predictions for threshold i
            preds = torch.tensor(preds, dtype=torch.long)
            # preds = preds.clone().detach().to(torch.long)
            
            # Compute TP, FP, TN, FN
            tp = torch.sum((true_labels == 1) & (preds == 1)).item()
            fp = torch.sum((true_labels == 0) & (preds == 1)).item()
            tn = torch.sum((true_labels == 0) & (preds == 0)).item()
            fn = torch.sum((true_labels == 1) & (preds == 0)).item()

            # Compute metrics
            accuracy = accuracy_score(true_labels.cpu(), preds.cpu())
            precision = precision_score(true_labels.cpu(), preds.cpu(), zero_division=0)
            recall = recall_score(true_labels.cpu(), preds.cpu(), zero_division=0)
            f1 = f1_score(true_labels.cpu(), preds.cpu(), zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
            tpr = recall  
            adv = tpr - fpr

            if isinstance(self.score_thr, (list, tuple)):
                name = f"thr_{self.score_thr[i]}" if len(self.score_thr) > 1 else f"thr_{self.score_thr[0]}"
            else:
                name = f"thr_{self.score_thr}"

            # Store metrics for this threshold
            metrics[name] = {
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
        
        all_f1_scores = [metrics[key]["F1 Score"] for key in metrics]
        all_accuracies = [metrics[key]["Accuracy"] for key in metrics]

        max_f1 = max(all_f1_scores)
        max_accuracy = max(all_accuracies)

        print(f"Maximum F1 Score: {max_f1:.4f}")
        print(f"Maximum Accuracy: {max_accuracy:.4f}")

        return metrics
