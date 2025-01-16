import torch 

class attack_recall():

    def run_attack(attack_dataset, membership_true, attack_options):
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
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
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
            self.membership_pred[ind_ex] = int(recall_score >= self.score_thr)