from dataset import *
from model_train import *
from attacks import *
import torch
import os
import sys
from options import *
from utils import *
from attacks_impl import *
import importlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parsing general input arguments and initilizing variables
args = Options()
args, _ = args.parser.parse_known_args()
all_args = sys.argv[1:]
torch_num_threads = int(args.torch_num_threads)
torch.set_num_threads(torch_num_threads) 
seed = int(args.seed)
fix_seed(seed)
load_model_dir = args.load_model_dir
attack_name = args.attack_name
run_mode = args.run_mode
output_dir = args.output_dir
log_dir = args.log_dir
task = args.task

#### loading a dataset
wdataset = Dataset(all_args)
if task == 'classification':    
    wdataset.train_test_val_split()
elif task == 'generative':
    wdataset.load_dataset()

if run_mode == 'fine-tunning':
    print("..fine-tunning of pretrained NLP model")
    #### fine-tuning of pretrained NLP model
    wmodel = PretrainedModels(all_args)
    wmodel.fine_tune(wdataset)

elif run_mode == 'attack':
    wmodel = PretrainedModels(all_args)
    # trainer, model, tokenizer = wmodel.prepare_model(wdataset)
    trainer = transformers.Trainer(
        model=wmodel.model,
        compute_metrics=wmodel.compute_metrics,
        train_dataset=wdataset.train_data,
        eval_dataset=wdataset.val_data
    )
    attack_options = {
        'load_model_dir': load_model_dir,
        'tokenizer' : wmodel.tokenizer,
        'trainer' : trainer,
        'model' : wmodel.model,
        'wdataset' : wdataset
    }
    print(f"..performing attack {attack_name}")
    #### performing an attack
    attack_class_name = capitalize_after_underscore(attack_name)
    attack_class = importlib.import_module("attacks_impl." + f"{attack_name}")
    attack_class = getattr(attack_class, f"Attack{attack_class_name}")
    if not attack_class:
        raise ValueError(f"Attack class '" + f"Attack{attack_class_name}" + "' not found.")
    wattack = attack_class(all_args)
    mia_metrics = wattack.perform_attack(wdataset, attack_options)





