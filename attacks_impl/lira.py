import sys
import argparse
import scipy.stats
import torch
import scipy
sys.path.insert(0, '..')
from attacks import *
from model_train import *

class AttackLira(ShadowAttack, ThresholdAttack):

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

    def fit_gaussian(self, data):
        """
        Fit a Gaussian distribution to the data and return the mean and variance.
    
        Parameters:
            data (array-like): Input data to fit.
    
        Returns:
            mean (float): Estimated mean of the Gaussian distribution.
            variance (float): Estimated variance of the Gaussian distribution.
        """
        mean = data.mean() 
        var = data.var()    
        return mean, variance
    
    import torch

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
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        wdataset = attack_options['wdataset']

        train_sample_size = len(wdataset.train_data)
        wdataset.sample_n_in_k(self.num_shadow_models, int(float(self.use_train_percentage) / 100 * train_sample_size))
        if self.lira_variant == "offline":
            in_indices = []
            out_indices = list(range(train_sample_size))
        else:
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
            # input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            in_losses = []
            out_losses = []
            for idx_shadow, trainer in trainers.items():
                print(trainer)
                with torch.no_grad():
                    # input_ids = ex['input_ids']
                    # outputs = trainer.model(input_ids)
                    input_ids = torch.tensor(ex['input_ids']).unsqueeze(0).to(self.device)  # adding batch dimension at position 0
                    attention_mask = torch.tensor(ex['attention_mask']).unsqueeze(0).to(self.device) 
                    outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    logits = outputs.logits 
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    logit_transformed_loss = self.stable_logit_transform(probs)
                    # loss = outputs.loss.item()  
                    # logit_transformed_loss = np.exp(-loss)
                    # logit_transformed_loss = np.log(logit_transformed_loss / (1 - logit_transformed_loss)) 
                    if idx_shadow in in_indices[ind_ex]:
                        in_losses.append(logit_transformed_loss[ex['label']])
                    elif idx_shadow in out_indices[ind_ex]:
                        out_losses.append(logit_transformed_loss[ex['label']])
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
            if self.lira_variant == "offline":
                prob_member = scipy.stats.norm.cdf(loss_value, loc=out_mean[ind_ex], scale=np.sqrt(out_var[ind_ex]))
                # self.membership_pred[ind_ex] = int(ratio >= 1)
                self.membership_pred = self.thresholds_comp(prob_member, self.score_thr) # matrix of shape (len(scores), len(thrs)) 
                self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")
            else:
                ratio = loss_value_likelihood_in / loss_value_likelihood_out
                self.membership_pred[ind_ex] = int(ratio >= 1)

            
                
                
                    

            
            

        



    
    