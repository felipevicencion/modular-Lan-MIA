from dataset import dataset
from model_train import *
from attacks import *
import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets
import os
from options import *
from utils import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parsing input arguments and initilizing variables


args = Options()
args = args.parser.parse_args()
dataset_name = args.dataset 
torch_num_threads = int(args.torch_num_threads)
torch.set_num_threads(torch_num_threads) 
max_sequence_length = int(args.max_sequence_length)
seed = int(args.seed)
fix_seed(seed)
model_name = args.model_name
num_epochs = float(args.num_epochs)
use_data_percentage = int(args.use_data_percentage)
load_model_dir = args.load_model_dir
attack_name = args.attack_name
num_trials = int(args.num_trials)
score_thr = float(args.score_thr)
num_shots = args.num_shots
pass_window = args.pass_window
max_length = args.max_length
prefix = args.prefix
num_shadow_models = int(args.num_shadow_models)
run_mode = args.run_mode
output_dir = args.output_dir
log_dir = args.log_dir

# dataset_name = "IMDb"
# model_name = "roberta-base"
# max_sequence_length = 512
# attack_name = "prediction_loss_based_mia"
# num_trials = 500
# load_model_dir = None
# num_epochs = 1
# log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
# output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'

print("..loading dataset")
#### loading a dataset
wdataset = dataset(dataset_name)
wdataset.train_test_split(use_data_percentage)

if run_mode == 'fine-tunning':
    print("..fine-mapping of pretrained NLP model")
    #### fine-tuning of pretrained NLP model
    wmodel = pretrained_models(model_name, max_sequence_length, load_model_dir, output_dir, log_dir)
    trainer, train_data, test_data = wmodel.fine_tune(wdataset, num_epochs)

elif run_mode == 'attack':
    wmodel = pretrained_models(model_name, max_sequence_length, load_model_dir)
    trainer, model, tokenizer = wmodel.load_model(wdataset)
    attack_options = {
        'pass_window': pass_window,
        'max_length': max_length,
        'num_shots': num_shots,
        'confidence_thr': 0.9,
        'tokenizer' : tokenizer,
        'trainer' : trainer,
        'score_thr' : score_thr,
        'prefix' : prefix,
        'model' : model,
        'num_shadow_models' : num_shadow_models
    }
    # print(attack_options)
    print(f"..performing attack {attack_name}")
    #### performing an attack
    wattack = attack(attack_name, num_trials)
    mia_metrics = wattack.perform_attack(wdataset, attack_options)





