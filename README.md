# modular-Lan-MIA
modular-Lan-MIA is a modular python library for benchmarking membership inference attacks with several NLP models and datasets.

# Example (run_main.sh bash script)

```
#!/bin/bash

#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPU cores per task for data loading, etc.
#SBATCH --mem=64G                  # Memory per node
#SBATCH --time=14:00:00            # Time limit
#SBATCH --partition=gpu            # GPU partition (adjust to your cluster's GPU partition)
#SBATCH --output=<output_path>.log

# Load necessary modules
module load python3                # Load Python
module load cuda/11.1              # Load CUDA toolkit

source <environment_location>/bin/activate

export OMP_NUM_THREADS=16

python3 main.py --dataset IMDb --num-epochs 10 --attack-name prediction_loss_based_mia 

```

# (General) Input options

| Option | Description | Default |
| --- | --- | --- |
| `--run-mode` | specifies the modality in which the library is used - either 'fine-tunning' or 'attack'| attack |
| `--dataset` | specifies a dataset on which an attack or fine-tunning is to be performed | IMDb |
| `--model-name`| name of the pretrained NLP model | roberta-base |
| `--task`| descibes the task which a model was trained for: 'classification' or 'generative' | classification |
| `--max-sequence-length` | maximum sequence length | 512 |
| `--torch-num-threads` | number of CPU threads PyTorch uses for parallel operations | 1 |
| `--seed` | specifies a seed used for random calculations inside the package | 0 |

# Fine-tuning options

| Option | Description | Default |
| --- | --- | --- |
| `--num-epochs` | number of epochs used in the fine-tuning of the model | 5 |
| `--use-data-percentage` | percentage of available train samples used to fine-tune the model | 100 |

# Attack options

| Option | Description | Default |
| --- | --- | --- |
| `--attack-name` | name of an attack to be performed | prediction_loss_based_mia |
| `--num-trials` | number of trials of an attack to be performed | 500 |
| `--load-model-dir` | directory from which a trained model is loaded | None |

## ReCall Input options

| Option | Description | Default |
| --- | --- | --- |
| `--num-shots` | Number of shots for prefix | 1 |
| `--pass-window`| (Optional) exceed the context window | True |
| `--max-length`| Maximum length of the window | 512 |


## Shadow Input options

| Option | Description | Default |
| --- | --- | --- |
| `--num-shadow-models` | Number of shadow models to be trained | 1 |


# Evaluation metrics

Once {num-trials} attacks are performed we summarize theit effectivness through the following evaluation metrics:
- Accuracy
- Precision
- Recall
- F1
- FPR
- Advantage

# Covered datasets
| Dataset | Description |
| --- | --- |
| `IMDb` | IMDb dataset contains 50,000 movie reviews labeled as either positive or negative. The dataset is balanced, with an equal number of positive and negative reviews. It is divided into 25,000 reviews for training and 25,000 for testing. |
| `Yelp` | The Yelp Review Full Dataset is a text classification dataset that consists of 700,000 reviews from the Yelp website. Reviews are labeled with one of five possible star ratings (1 to 5), representing sentiments from Very negative (1) to Very positive (5). It is divided into 650,000 reviews for training and 50,000 for testing. 

# Covered NLP models
| Model name | Description |
| --- | --- |
| `roberta-base` | RoBERTa-base model is a pre-trained transformer model designed for natural language understanding tasks. It builds on BERT by using larger datasets, training for longer time, removing next-sentence prediction objective and by training with dynamic masking.

It supports a number of generative models that can be loaded using `transformers.AutoModelForCausalLM.from_pretrained`. These include GPT-style autoregressive models models like GPT-2, GPT-Neo, GPT-J, and GPT-NeoX-20B, but also supports Meta’s LLaMA models and the multilingual Bloom model as well as Falcon-7B, Falcon-40B, Meta’s OPT series, Pythia etc.

# Covered attacks
| Attack name | Keyword | Description |
| --- | --- | --- | 
| Prediction Correctness Based MIA[^1] | `prediction_correctness_based_mia` | An attacker infers an input record as a member if it is correctly predicted by the target model, otherwise the attacker infers it as a non-member.  |
| Prediction Loss Based MIA[^1] | `prediction_loss_based_mia` | An attacker infers an input record as a member if its prediction loss is smaller than the average loss of all training members, otherwise the attacker infers it as a nonmember. |
| Prediction Confidence Based MIA[^2] | `prediction_confidence_based_mia` | An attacker infers an input record as a member if its maximum prediction confidence is larger than a preset threshold, otherwise the attacker infers it as a non-member |
| Prediction Entropy Based MIA[^2] | `prediction_entropy_based_mia` | An attacker infers an input record as a member if its prediction entropy is smaller than a preset threshold, otherwise the attacker infers it as a nonmember. |
| Modified Prediction Entropy Based MIA[^3] | `prediction_entropy_based_mia` | An attacker infers an input record as a member if its modified prediction entropy (which involves information on true labels) is smaller than a preset threshold, otherwise the attacker infers it as a nonmember. |
| ReCall | `recall` | Based on the assumption that the relative change in conditional log-likelihoods when prefixing target data points with non-target data points will be larger than the realtive change when prefixing non-target data point with non-target data points. More details can be found on [ReCall project page](https://royxie.com/recall-project-page/)  |
| Black-box Shadow MIA[^4] | `bb-shadow` |  Based on training a binary classifier, which can distinguish a target model’s behavior of itstraining members from the non-members by creating multiple shadow models to mimic the behavior of the target model. Still under construction. |
| Zlib Entropy MIA[^5] | `zlib` | The Zlib Entropy normalizes the target model’s loss using the zlib compression size of the input sample and decides on membership by comparing to a threshod. |
| Mink-K% MIA[^6] | `mink` | Thresholds against membership scores computed using the average log-likelihood of the k% of tokens with the lowest probabilities, predicting membership if above threshold as the least likely tokens are expected to have higher probabilities for member samples compared to non-members. |
| perplexity[^7] | `perplexity` | Attacks calculates perplexity score (introduced in [^7]) for a sequence and compares it to a threshold. Perplexity measures how surprised the model is to see a given sequence, so a lower perplexity means that sequence is more likely to have been generated by the model. |
| LiRA[^8] | `lira` | for each datapoint tains a number of shadow models half of which contains it, half of which don't. The assignment of membership is based upon a likelihood ratio test with Gaussian approximations for membership and non-membership loss distributions. |

# Adding a custom attack to the framework

Every embedded attack should be an instance of either `ShadowAttack` or `MetricAttack` class (or its subclass named `ThresholdAttack`) and must implement all the required operations:
1. constructor `__init__(self, attack_name, num_trials, unknown_args)`
2. run_attack procedure `run_attack(self, attack_dataset, membership_true, attack_options)`

Newly defined attack must be placed in `attacks_impl` subdirectory within the many repository alongside other implemented attacks. Currently, constructor must accept the following parameters:

| Parameter        | Description                                                                                                           |
|------------------|-----------------------------------------------------------------------------------------------------------------------|
| `attack_name`    | Name of the implemented attack.                                                                                       |
| `num_trials`     | Number of membership inference attacks to be performed (50% members, 50% non-members).                                |
| `unknown_args`   | List of unused input options to be parsed and used withing the attack implementation.                                 |


and a function `run_attack(self, attack_dataset, membership_true, attack_options)` must accept the following parameters:

| Parameter        | Description                                                                                                           |
|------------------|-----------------------------------------------------------------------------------------------------------------------|
| `attack_dataset` | A dataset of format supported by library `dataset` that combines members and non-members data for attack evaluation.  |
| `membership_true`| A torch vector indicating membership status: 1 for member, 0 for non-member.                                          |
| `attack_options` | A dictionary of parameters used to configure the attack depending information needed for a particular attack.         |

and should load the `attacks` library in the parental folder. 

In the continuation of this chapter, we provide an example of implmentation for one of the available attacks called _recall_:

```
import sys
import argparse
import torch 
sys.path.insert(0, '..')
from attacks import *

class AttackMink(ThresholdAttack):

    def __init__(self, attack_name, num_trials, unknown_args):
        super().__init__(attack_name, num_trials)
        attack_options_parser = argparse.ArgumentParser(description="mink options parser")
        attack_options_parser.add_argument("-mink_ratio", "--mink-ratio", type=float, help="percentage determining the count of lowest probable tokens considered for mink score", default=0.2)
        attack_args, _ = attack_options_parser.parse_known_args(unknown_args)
        self.mink_ratio = attack_args.mink_ratio

    def run_attack(self, attack_dataset, membership_true, attack_options):
        self.model = attack_options['model']
        self.score_thr = attack_options['score_thr']
        self.max_length = attack_options['max_length']
        self.tokenizer = attack_options['tokenizer']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.membership_pred = torch.zeros(2*self.num_trials_per_cat)
        self.mink_ratio = 0.2
        for ind_ex, ex in enumerate(tqdm(attack_dataset)):
            input_ids = torch.tensor(self.tokenizer.encode(ex["text"])).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            _, logits = outputs[:2]
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            mink_score = np.mean(topk).item()
            self.membership_pred[ind_ex] = int(mink_score >= self.score_thr)
```


[^1]: Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018. Privacy risk in machine learning: Analyzing
the connection to overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF). IEEE, 268–282.
[^2]: Ahmed Salem, Yang Zhang, Mathias Humbert, Mario Fritz, and Michael Backes. 2019. ML-Leaks: Model and Data
Independent Membership Inference Attacks and Defenses on Machine Learning Models. In Network and Distributed
Systems Security Symposium 2019. Internet Society
[^3]: Liwei Song and Prateek Mittal. 2021. Systematic evaluation of privacy risks of machine learning models. In 30th
USENIX Security Symposium (USENIX Security 21). USENIX Association, 2615–2632.
[^4]: Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Membership inference attacks against
machine learning models. In 2017 IEEE Symposium on Security and Privacy (S&P). IEEE, 3–18.
[^5]: Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650.
[^6]: Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024. Detecting pretraining data from large language models. In The Twelfth International Conference on Learning Representations.
[^7]: Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Xiaodong Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting training data from large language models. In USENIX Security Symposium, 2020. URL https://api.semanticscholar.org/ CorpusID:229156229.
[^8]: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles, 2022.

            