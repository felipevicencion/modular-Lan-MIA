import sys
import argparse
import torch
import scipy
sys.path.insert(0, '..')
from attacks import *
from model_train import *

class AttackLira(ShadowAttack):

    def __init__(self, input_args):
        super().__init__(input_args)
        attack_options_parser = argparse.ArgumentParser(description="lira options parser")
        attack_options_parser.add_argument("-lira_variant", "--lira-variant", type=str, help="either 'online' or 'offline'", default='online')
        attack_options_parser.add_argument("-global_vars", "--global-vars", type=bool, help="determines if the loss distribution variance should be estimated globally or per-sample", default='online')
        attack_options_parser.add_argument("-use_train_percentage", "--use-train-percentage", type=bool, help="determines the percentage of samples that are used in each shadow model training", default=50)
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.lira_variant = attack_args.lira_variant
        self.global_vars = attack_args.global_vars
        self.use_train_percentage = attack_args.use_train_percentage

    def fit_gaussian(data):
        """
        Fit a Gaussian distribution to the data and return the mean and variance.
    
        Parameters:
            data (array-like): Input data to fit.
    
        Returns:
            mean (float): Estimated mean of the Gaussian distribution.
            variance (float): Estimated variance of the Gaussian distribution.
        """
        mean = np.mean(data)
        variance = np.var(data)
    
        return mean, variance
    
    import torch

    def evaluate_loss_at_sample(trainer, inputs):
        """
        Evaluates the loss of a given sample using a trained model in the trainer.

        Args:
            trainer (transformers.Trainer): The fine-tuned Trainer object.
            inputs (str): The input tokenized text for evaluation.

        Returns:
            float: The loss value at the given sample point.
        """
        model = trainer.model
        model.eval()
        with torch.no_grad(): 
            outputs = model(**inputs, labels=inputs["input_ids"])  # Forward pass with labels
            loss = outputs.loss 
        return loss.item()

    def stable_logit_transform(self, probs):
        logit_transformed_loss = torch.zeros_like(probs)
        for i in range(len(probs)):
            sum_others = 0
            for j in range(len(probs)):
                sum_others = sum_others + probs[i]
            logit_transformed_loss[i] = torch.log(probs[i] / sum_others)
        return logit_transformed_loss

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        # self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        wdataset = attack_options['wdataset']

        train_sample_size = len(wdataset.train_data)
        wdataset.sample_n_in_k(self.num_shadow_models, int(float(self.use_train_percentage) / 100 * train_sample_size))
        in_indices, out_indices = wdataset.distribute_to_shadow_datasets(attack_dataset)
        wmodel = PretrainedModels(self.input_args)
        trainers = {}
        for idx_shadow in range(self.num_shadow_models):
            wmodel.fine_tune(wdataset, idx_shadow)
            trainers[idx_shadow] = wmodel.trainer
        
        # performing an attack on attack dataset
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        in_mean = torch.zeros(2*self.num_trials_per_cat)
        out_mean = torch.zeros(2*self.num_trials_per_cat)
        in_var = torch.zeros(2*self.num_trials_per_cat)
        out_var = torch.zeros(2*self.num_trials_per_cat)
        in_losses_all = []
        out_losses_all = []
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            print(ex)
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            in_losses = []
            out_losses = []
            for i, trainer in enumerate(trainers):
                with torch.no_grad():
                    outputs = trainer.model(input_ids, labels=input_ids)
                    logits = outputs.logits 
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    print(probs)
                    logit_transformed_loss = self.stable_logit_transform(probs)
                    # loss = outputs.loss.item()  
                    # logit_transformed_loss = np.exp(-loss)
                    # logit_transformed_loss = np.log(logit_transformed_loss / (1 - logit_transformed_loss)) 
                    if i in in_indices[ind_ex]:
                        in_losses.append(logit_transformed_loss)
                    elif i in out_indices[ind_ex]:
                        out_losses.append(logit_transformed_loss)
            in_losses_all = in_losses_all + in_losses
            out_losses_all = out_losses_all + out_losses
            in_mean[ind_ex], in_var[ind_ex] = self.fit_gaussian(in_losses)
            out_mean[ind_ex], out_var[ind_ex] = self.fit_gaussian(out_losses)

        if self.global_vars == "offline":
                _ , in_var_global = self.fit_gaussian(in_losses_all)
                _ , out_var_global = self.fit_gaussian(out_losses_all)
                in_var = in_var_global
                out_var = out_var_global

        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            # loss_value = self.evaluate_loss_at_sample(attack_options['trainer'], self.tokenizer, input_ids, self.device)
            outputs_ev = attack_options['trainer'].model(input_ids, labels=input_ids)
            logits_ev = outputs_ev.logits 
            probs_ev = torch.nn.functional.softmax(logits_ev, dim=-1)
            loss_value = self.stable_logit_transform(probs_ev)
            loss_value_likelihood_in = scipy.stats.norm(loss_value, in_mean[ind_ex], np.sqrt(in_var[ind_ex]))
            loss_value_likelihood_out = scipy.stats.norm(loss_value, out_mean[ind_ex], np.sqrt(out_var[ind_ex]))
            ratio = loss_value_likelihood_in / loss_value_likelihood_out
            self.membership_pred[ind_ex] = int(ratio >= 1)

            
                
                
                    

            
            

        



    
    