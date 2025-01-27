import sys
import torch
sys.path.insert(0, '..')
from attacks import *

class LiRA(ShadowAttack):

    def __init__(self, attack_name, num_trials, unknown_args):
        super().__init__(attack_name, num_trials)
        attack_options_parser = argparse.ArgumentParser(description="lira options parser")
        attack_options_parser.add_argument("-lira_variant", "--lira-variant", type=str, help="either 'online' or 'offline'", default='online')
        attack_options_parser.add_argument("-global_vars", "--global-vars", type=bool, help="determines if the loss distribution variance should be estimated globally or per-sample", default='online')
        attack_args, _ = attack_options_parser.parse_known_args(unknown_args)
        self.lira_variant = attack_args.lira_variant
        self.global_vars = attack_args.global_vars

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

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)

        
    
    