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
| `--dataset` | specifies a dataset on which an attack is to be performed | IMDb |
| `--model-name`| name of the pretrained NLP model | roberta-base |
| `--attack-name` | name of an attack to be performed | prediction_loss_based_mia |
| `--num-trials` | number of trials of an attack to be performed | 500 |
| `--load-model-dir` | directory from which a trained model is loaded | None |
| `--num-epochs` | number of epochs used in the fine-tuning of the model | 5 |
| `--max-sequence-length` | maximum sequence length | 512 |
| `--torch-num-threads` | number of CPU threads PyTorch uses for parallel operations | 1 |


## ReCall Input options

| Option | Description | Default |
| --- | --- | --- |
| `--num-shots` | Number of shots for prefix | 1 |
| `--pass-window`| (Optional) exceed the context window | True |
| `--max-length`| Maximum length of the window | 512 |

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

# Covered attacks
| Attack name | Keyword | Description |
| --- | --- | --- | 
| Prediction Correctness Based MIA | `prediction_correctness_based_mia` | An attacker infers an input record as a member if it is correctly predicted by the target model, otherwise the attacker infers it as a non-member.  |
| Prediction Loss Based MIA | `prediction_loss_based_mia` | An attacker infers an input record as a member if its prediction loss is smaller than the average loss of all training members, otherwise the attacker infers it as a nonmember. |
| Prediction Confidence Based MIA | `prediction_confidence_based_mia` | An attacker infers an input record as a member if its maximum prediction confidence is larger than a preset threshold, otherwise the attacker infers it as a non-member |
| ReCall | `recall` | Based on the assumption that the relative change in conditional log-likelihoods when prefixing target data points with non-target data points will be larger than the realtive change when prefixing non-target data point with non-target data points. More details can be found on [ReCall project page](https://royxie.com/recall-project-page/)  |





            