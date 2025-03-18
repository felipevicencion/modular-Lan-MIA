import sys
import re
import argparse
import torch 
import random
sys.path.insert(0, '..')
from attacks import *
from utils import *
import transformers

class AttackRecall(ThresholdAttack):

    def __init__(self, input_args):
        super().__init__(input_args)
        attack_options_parser = argparse.ArgumentParser(description="recall options parser")
        attack_options_parser.add_argument("-num_shots", "--num-shots", type=int, help = "attack ReCall: number of shots for prefix", default = 1)
        attack_options_parser.add_argument("-pass_window", "--pass-window", type=bool, help = "attack ReCall: (Optional) exceed the context window", default = True)
        attack_options_parser.add_argument("-max_length", "--max-length", type=int, help = "attack ReCall: Maximum length of the window", default = 512)
        attack_options_parser.add_argument("-prefix", "--prefix", type=str, help = "attack ReCall: a prefix to be used in an attack", default = "This is a synthetic prefix!")
        attack_args, _ = attack_options_parser.parse_known_args(input_args)
        self.num_shots = attack_args.num_shots
        self.pass_window = attack_args.pass_window
        self.max_length = attack_args.max_length
        self.prefix = attack_args.prefix

    def generate_prompt(self, example):
        return f"Generate a gramatically correct passage that is similar to the given text in length, domain, and style: {example}"

    # def get_completion(self, prompt):
    #     model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    #     tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    #     inputs = tokenizer.encode(prompt, return_tensors="pt")
    #     attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    #     outputs = model.generate(inputs, max_length=200, attention_mask=attention_mask, temperature=0.3, num_return_sequences=1, do_sample=True, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2)
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     response = generated_text[len(prompt):]
    #     response = re.sub(r'[^A-Za-z0-9\s!?.,]', '', response)
    #     response = response.replace("\n", " ").replace("\r", " ")
    #     return response

    def get_completion(self, prompt):
        model_name = "EleutherAI/gpt-neo-2.7B"  # Or "EleutherAI/gpt-j-6B"
        model = transformers.GPTNeoForCausalLM.from_pretrained(model_name)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        outputs = model.generate(
            inputs, 
            max_length=self.max_length // 2, 
            attention_mask=attention_mask,
            temperature=0.3, 
            num_return_sequences=1, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id, 
            no_repeat_ngram_size=2
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt)+2:]
        response = re.sub(r'[^A-Za-z0-9\s!?.,]', '', response)
        response = response.replace("\n", " ").replace("\r", " ")
        return response

    def gpt_synthetic_prefix(self, original_prefixes):
        # default generate synthetic prefix from non-member data
        synthetic_prefixes = []
        for original_prefix in original_prefixes:
            prompt = self.generate_prompt(original_prefix)
            response = self.get_completion(prompt)      
            print(f"response = {response}")
            synthetic_prefixes.append(response)
        return synthetic_prefixes

    def get_conditional_ll(self, input_text, target_text, model, tokenizer, device):
        """
        Calculates conditional log-likelihood, probabilities, and perplexity for a target text
        conditioned on the prefix.

        Parameters:
            input_text (str): The input context text.
            target_text (str): The target text for which to compute metrics.
            model: The pre-trained language model.
            tokenizer: Tokenizer associated with the model.
            device: The device (CPU/GPU) for computation.

        Returns:
            tuple: See `get_all_prob` for details.
        """
        input_encodings = tokenizer(input_text, return_tensors="pt")
        target_encodings = tokenizer(target_text, return_tensors="pt")
        concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
        labels = concat_ids.clone()
        labels[:, : input_encodings.input_ids.size(1)] = -100
        with torch.no_grad():
            outputs = model(concat_ids, labels=labels)
        loss, logits = outputs[:2]
        return get_all_prob(labels, loss, logits)


    def process_prefix(self, max_length, prefix, avg_length, pass_window, total_shots, tokenizer):
        """
        Processes the prefix for a recall attack 

        Parameters:
            max_length (int): Maximum allowed token length for the sequence.
            prefix (list): A list of prefix shots
            avg_length (int): Average target length for the sequence.
            pass_window (bool): Whether to bypass truncation and use the full prefix.
            total_shots (int): Total number of shots to include in the prefix.
            tokenizer: Tokenizer used to encode the text.

        Returns:
            list: The processed prefix, truncated if necessary to fit within the max token length.
        """
        if pass_window:
            return prefix
        token_counts = [len(tokenizer.encode(shot)) for shot in prefix]
        target_token_count = avg_length
        total_tokens = sum(token_counts) + target_token_count
        if total_tokens <= max_length:
            return prefix
        # Determine the maximum number of shots that can fit within the max_length
        max_shots = 0
        cumulative_tokens = target_token_count
        for count in token_counts:
            if cumulative_tokens + count <= max_length:
                max_shots += 1
                cumulative_tokens += count
            else:
                break
        # Truncate the prefix to include only the maximum number of shots
        truncated_prefix = prefix[-max_shots:]
        # total_shots = max_shots
        return truncated_prefix


    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = attack_options['tokenizer']
        if int(self.num_shots) != 0:  
            avg_length = int(np.mean([len(ex["text"]) for ex in attack_dataset])) 
            print(f"avg_length={avg_length}")
            if self.prefix == "None":
                print(attack_dataset[random.choice([i for i, v in enumerate(membership_true) if v == 1])]['text'])
                self.prefix = self.gpt_synthetic_prefix([attack_dataset[random.choice([i for i, v in enumerate(membership_true) if v == 1])]['text']])
            else:
                self.prefix = self.process_prefix(self.max_length, self.prefix, avg_length, self.pass_window, self.num_shots, self.tokenizer) 
            print(f"prefix = {self.prefix}")
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        recall_scores = []
        for _, ex in enumerate(tqdm(attack_dataset)):
            # unconditional log-likelihood calculation
            with torch.no_grad():
                inputs = ex['input_ids'].unsqueeze(0).to(self.device) 
            ll = get_ll(inputs, self.model)[1]
            recall_score = 0
            if int(self.num_shots) != 0:   
                # conditional log-likelihood with prefix     
                ll_nonmember = self.get_conditional_ll("".join(self.prefix), ex["text"], self.model, self.tokenizer, self.device)[1]
                recall_score = ll_nonmember / ll
                if np.isnan(ll_nonmember / ll).all():
                    recall_score = 0              
            recall_scores.append(recall_score)
        if self.num_score_thr > 0:
            self.score_thr = torch.linspace(min(recall_scores), max(recall_scores), steps=self.num_score_thr).tolist()
        self.membership_pred = self.thresholds_comp(recall_scores, self.score_thr, "ge") # matrix of shape (len(scores), len(thrs)) 
        self.plot_roc_curves(self.membership_pred, membership_true, f"{self.output_dir}/{self.output_name}")