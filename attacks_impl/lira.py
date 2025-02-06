import sys
import torch
import scipy
sys.path.insert(0, '..')
from attacks import *
from model_train import *

class LiRA(ShadowAttack):

    def __init__(self, attack_name, num_trials, unknown_args):
        super().__init__(attack_name, num_trials)
        attack_options_parser = argparse.ArgumentParser(description="lira options parser")
        attack_options_parser.add_argument("-lira_variant", "--lira-variant", type=str, help="either 'online' or 'offline'", default='online')
        attack_options_parser.add_argument("-global_vars", "--global-vars", type=bool, help="determines if the loss distribution variance should be estimated globally or per-sample", default='online')
        attack_options_parser.add_argument("-use_train_percentage", "--use-train-percentage", type=bool, help="determines the percentage of samples that are used in each shadow model training", default=50)
        attack_args, _ = attack_options_parser.parse_known_args(unknown_args)
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

    def evaluate_loss_at_sample(trainer, inputs, device):
        """
        Evaluates the loss of a given sample using a trained model in the trainer.

        Args:
            trainer (transformers.Trainer): The fine-tuned Trainer object.
            inputs (str): The input tokenized text for evaluation.
            device (torch.device): Device on which the model runs.

        Returns:
            float: The loss value at the given sample point.
        """
        model = trainer.model
        model.eval()
        with torch.no_grad(): 
            outputs = model(**inputs, labels=inputs["input_ids"])  # Forward pass with labels
            loss = outputs.loss 
        return loss.item()


    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        self.num_shadow_models = attack_options['num_shadow_models']
        wdataset = attack_options['wdataset']
        num_epochs = 10

        # if the attack is of ShadowAttack type then we pass model training dataset under 'attack_options'
        train_sample_size = len(wdataset.train_data)

        wdataset.sample_n_in_k(self.num_shadow_models, int(float(self.use_train_percentage) / 100 * len(train_sample_size)))
        in_indices, out_indices = wdataset.distribute_to_shadow_datasets(attack_dataset)
        wmodel = PretrainedModels(attack_options['model_options'])
        trainers = {}
        for idx_shadow in range(wmodel.num_shadow_models):
            trainers[idx_shadow], _, _ = wmodel.fine_tune(wdataset, num_epochs, idx_shadow)
            # trainers[idx_shadow] = trainer
        
        # performing an attack on attack dataset
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            in_losses = []
            out_losses = []
            for i, trainer in enumerate(trainers):
            with torch.no_grad():
                outputs = trainer.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()  
                logit_transformed_loss = np.exp(-loss)
                logit_transformed_loss = np.clip(logit_transformed_loss, 1e-8, 1 - 1e-8) 
                logit_transformed_loss = np.log(logit_transformed_loss / (1 - logit_transformed_loss)) 
                if in_indices[ind_ex][i] == 1:
                    in_losses.append(logit_transformed_loss)
                elif out_indices[ind_ex][i] == 1:
                    out_losses.append(logit_transformed_loss)
            in_mean, in_var = fit_gaussian(in_losses)
            out_mean, out_var = fit_gaussian(out_losses)
            loss_value = evaluate_loss_at_sample(attack_options['trainer'], self.tokenizer, input_ids, self.device)
            loss_value_likelihood_in = scipy.stats.norm(loss_value, in_mean, np.sqrt(in_var))
            loss_value_likelihood_out = scipy.stats.norm(loss_value, out_mean, np.sqrt(out_var))
            ratio = loss_value_likelihood_in / loss_value_likelihood_out
            self.membership_pred[ind_ex] = int(ratio >= 1)

            
                
                
                    

            
            

        



    
    