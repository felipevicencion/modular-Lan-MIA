from dataset import *
from model_train import *
from attacks import *
import torch
# import datasets
import os
import sys
from options import *
from utils import *
from attacks_impl import *
import importlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parsing input arguments and initilizing variables
args = Options()
args, _ = args.parser.parse_known_args()
all_args = sys.argv[1:]
# args = args.parser.parse_args()
# dataset_name = args.dataset
# task = args.task
torch_num_threads = int(args.torch_num_threads)
torch.set_num_threads(torch_num_threads) 
max_sequence_length = int(args.max_sequence_length)
seed = int(args.seed)
fix_seed(seed)
# model_name = args.model_name
# use_float16 = bool(args.use_float16)
# num_epochs = float(args.num_epochs)
use_data_percentage = int(args.use_data_percentage)
load_model_dir = args.load_model_dir
attack_name = args.attack_name
num_trials = int(args.num_trials)
score_thr = parse_thresholds(str(args.score_thr))
# score_thr = float(args.score_thr)
num_shadow_models = int(args.num_shadow_models)
run_mode = args.run_mode
output_dir = args.output_dir
log_dir = args.log_dir

#### loading a dataset
wdataset = Dataset(all_args)
wdataset.train_test_split(use_data_percentage)

# model_options = {
#     'model_name': model_name,
#     'task': task,
#     'max_sequence_length': max_sequence_length,
#     'use_float16': use_float16,
#     'load_model_dir': load_model_dir,
#     'output_dir': output_dir,
#     'log_dir' : log_dir
# }

if run_mode == 'fine-tunning':
    print("..fine-tunning of pretrained NLP model")
    #### fine-tuning of pretrained NLP model
    wmodel = PretrainedModels(all_args)
    trainer, train_data, test_data = wmodel.fine_tune(wdataset)

elif run_mode == 'attack':
    wmodel = PretrainedModels(all_args)
    trainer, model, tokenizer = wmodel.prepare_model(wdataset)
    attack_options = {
        'load_model_dir': load_model_dir,
        'confidence_thr': 0.9,
        'tokenizer' : tokenizer,
        'trainer' : trainer,
        'score_thr' : score_thr,
        'model' : model,
        'num_shadow_models' : num_shadow_models,
        'max_sequence_length' : max_sequence_length,
        'output_dir' : output_dir,
        'wdataset' : wdataset
        # 'model_options' : model_options
    }
    print(f"..performing attack {attack_name}")
    #### performing an attack
    attack_class_name = capitalize_after_underscore(attack_name)
    attack_class = importlib.import_module("attacks_impl." + f"{attack_name}")
    attack_class = getattr(attack_class, f"Attack{attack_class_name}")
    if not attack_class:
        raise ValueError(f"Attack class '" + f"Attack{attack_class_name}" + "' not found.")
    wattack = attack_class(attack_name, num_trials, all_args)
    mia_metrics = wattack.perform_attack(wdataset, attack_options)





