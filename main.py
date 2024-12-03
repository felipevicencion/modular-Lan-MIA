from dataset import dataset
from model_train import *
from attacks import *
import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets
import os
from options import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parsing input arguments and initilizing variables


args = Options()
args = args.parser.parse_args()
dataset_name = args.dataset 
torch_num_threads = int(args.torch_num_threads)
torch.set_num_threads(torch_num_threads) 
max_sequence_length = int(args.max_sequence_length)
model_name = args.model_name
num_epochs = float(args.num_epochs)
load_model_dir = args.load_model_dir
attack_name = args.attack_name
num_trials = int(args.num_trials)
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
train_data, test_data = wdataset.train_test_split()

if run_mode == 'fine-tunning':
    print("..fine-mapping of pretrained NLP model")
    #### fine-tuning of pretrained NLP model
    wmodel = pretrained_models(model_name, max_sequence_length, load_model_dir, output_dir, log_dir)
    trainer, train_data, test_data = wmodel.fine_tune(train_data, test_data, num_epochs)

elif run_mode == 'attack':
    wmodel = pretrained_models(model_name, max_sequence_length, load_model_dir)
    trainer, train_data, test_data = wmodel.load_model(train_data, test_data)
    print(f"..performing attack {attack_name}")
    #### performing an attack
    wattack = attack(attack_name, num_trials)
    mia_metrics = wattack.perform_attack(trainer, train_data, test_data)





