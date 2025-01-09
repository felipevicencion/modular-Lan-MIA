import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score, precision_score
import numpy as np
import torch
import datasets
from tqdm import tqdm
from utils import *    
from dataset import *
import zlib
    
class attack:

    def __init__(self, attack_name, num_trials):
        # attack_name = 'prediction_loss_based_mia'
        # num_trials = 500
            self.attack_name = attack_name
            self.num_trials = num_trials
            self.num_trials_per_cat = int( 0.5 * num_trials )
            self.mia_metrics = {}

    def calculate_mia_metrics(self, true_labels, predicted_labels):
        # Ensure the tensors are of type long (for classification)
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
        tpr = recall  # recall is the same as TPR
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


    def perform_attack(self, wdataset, attack_options):

        train_data, test_data = wdataset.get_train_test()

        # attack_options is a dictionary containing additional inputs and parameters required to performa MIAs such as thresold values and other
        if self.attack_name not in ['recall', 'zlib']:
            train_data, test_data = tokenize_data(attack_options['tokenizer'], train_data, test_data, attack_options['max_sequence_length'])
        shf_train_data = train_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        shf_test_data = test_data.shuffle(seed=42).select(range(self.num_trials_per_cat))
        attack_dataset = datasets.concatenate_datasets([shf_train_data, shf_test_data])
        membership_true = torch.cat((torch.ones(self.num_trials_per_cat), torch.zeros(self.num_trials_per_cat)))

        if self.attack_name == 'prediction_loss_based_mia':

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
            membership_pred = (per_sample_loss < training_loss).int()
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()

        elif self.attack_name == 'prediction_correctness_based_mia':

            trainer = attack_options['trainer']
            predictions_output = trainer.predict(attack_dataset)
            predictions = predictions_output.predictions
            label_ids = predictions_output.label_ids
            predicted_labels = np.argmax(predictions, axis=1)
            membership_pred = torch.tensor( (predicted_labels == label_ids).astype(int) )
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()

        elif self.attack_name == 'prediction_confidence_based_mia':

            trainer = attack_options['trainer']
            predictions_output = trainer.predict(attack_dataset)
            predictions = predictions_output.predictions
            probabilities = scipy.special.softmax(predictions, axis=1)
            max_probabilities = np.max(probabilities, axis=1)
            membership_pred = torch.tensor( (max_probabilities >= attack_options['confidence_thr']).astype(int) )
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()

        elif self.attack_name == 'bb-shadow':
            print(self.attack_name)

        elif self.attack_name == 'recall':
            self.prefix = attack_options['prefix']
            self.model = attack_options['model']
            self.score_thr = attack_options['score_thr']
            self.max_length = attack_options['max_length']
            self.num_shots = attack_options['num_shots']
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pass_window = attack_options['pass_window']
            self.tokenizer = attack_options['tokenizer']
            if int(self.num_shots) != 0:  
                for ex in attack_dataset:
                    print(ex)
                    break
                # avg_length = int(np.mean([len(self.tokenizer.encode(ex["input_ids"])) for ex in attack_dataset])) 
                avg_length = int(np.mean([len(ex["text"]) for ex in attack_dataset])) 
                print(f"avg_length={avg_length}")
                prefix = recall_process_prefix(self.max_length, self.prefix, avg_length, self.pass_window, self.num_shots, self.tokenizer) 
                print(f"prefix = {prefix}")
            membership_pred = torch.zeros(2*self.num_trials_per_cat)
            for ind_ex, ex in enumerate(tqdm(attack_dataset)):
                # unconditional log-likelihood calculation
                ll = get_ll(ex["text"], self.model, self.tokenizer, self.device)[1]
                print(f"ll = {ll}")
                # ReCaLL
                recall_score = 0
                if int(self.num_shots) != 0:   
                    # conditional log-likelihood with prefix     
                    ll_nonmember = recall_get_conditional_ll("".join(self.prefix), ex["text"], self.model, self.tokenizer, self.device)[1]
                    print(f"ll_nonmember={ll_nonmember}")
                    recall_score = ll_nonmember / ll
                    if np.isnan(ll_nonmember / ll).all():
                        recall_score = 0              
                # membership_pred = torch.tensor(int(recall_score >= self.score_thr))
                membership_pred[ind_ex] = int(recall_score >= self.score_thr)
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()

        elif self.attack_name == 'zlib':
            self.model = attack_options['model']
            self.score_thr = attack_options['score_thr']
            self.max_length = attack_options['max_length']
            self.tokenizer = attack_options['tokenizer']
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            membership_pred = torch.zeros(2*self.num_trials_per_cat)
            for ind_ex, ex in enumerate(tqdm(attack_dataset)):
                ll = get_ll(ex["text"], self.model, self.tokenizer, self.device)[1]
                zlib_score = ll / len(zlib.compress(bytes(ex["text"], "utf-8")))
                # print(f"zlib_score = {zlib_score}")
                membership_pred[ind_ex] = int(zlib_score >= self.score_thr)
            self.mia_metrics = self.calculate_mia_metrics(membership_true, membership_pred)
            self.print_mia_metrics()
            
        else:
    
            raise ValueError("Invalid attack was chosen!")

        return self.mia_metrics


