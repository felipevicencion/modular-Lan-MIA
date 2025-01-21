import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
# import torch
import datasets
import pandas as pd
from utils import *
from dataset import *
# from accelerate import Accelerator

class pretrained_models:

    def __init__(self, model_options):
        # model_name = 'roberta_base'
        # max_sequence_length = 512
        # log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
        # output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'
        self.model_name = model_options['model_name']
        self.max_sequence_length = model_options['max_sequence_length']
        self.output_dir = model_options['output_dir']
        self.log_dir = model_options['log_dir']
        self.load_model_dir = model_options['load_model_dir']
        self.use_float16 = model_options['use_float16']
        self.task = model_options['task']
        self.load_specific_model()
        if self.task == 'generative':
            self.model.eval()
        
    def load_specific_model(self):
        if self.task == 'classification':
            try:
                self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_model_dir)
                self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
            except Exception as e:
                print(f"Error: {e}")
                print("Ensure you are using the correct directory for loading the model.")
                print(f"..model {model} is loaded")
        elif self.task == 'generative':
            print(f"..loading {self.task} model: {self.model_name}")
            if "roberta" in self.model_name:
                self.model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name)
                self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
            elif "mamba" in self.model_name:
                self.model = transformers.MambaForCausalLM.from_pretrained(
                    self.model_name, return_dict=True, trust_remote_code=True,
                    torch_dtype=torch.float16 if self.use_float16 else torch.float32,
                    device_map="cuda:0"
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # model_name = "EleutherAI/pythia-70m-deduped"
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                  self.model_name, return_dict=True, trust_remote_code=True, 
                    torch_dtype=torch.float16 if self.use_float16 else torch.float32
                )    
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # piece of code for testing the code above
                # inputs = self.tokenizer("Hello, square root of -4 is", return_tensors="pt")
                # tokens = self.model.generate(**inputs, max_length=30, num_return_sequences=1)
                # print(self.tokenizer.decode(tokens[0]))

    def prepare_model(self, wdataset):

        # fetch train and test data
        train_data, test_data = wdataset.get_train_test()
        if self.task == 'classification':
            self.num_classes = len(set(train_data.to_pandas()["label"]))
            self.model.config.num_labels = self.num_classes
            print(f"..num_classes = {self.num_classes}")
            
        train_data, test_data = tokenize_data(self.tokenizer, train_data, test_data, self.max_sequence_length)

        trainer = transformers.Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics,
            train_dataset=train_data,
            eval_dataset=test_data
        )
        return trainer, self.model, self.tokenizer

    def compute_metrics(self, pred):
        if self.task == 'classification':
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            print(pred)
            # if len(set(labels)) != len(set(pred.flatten())):
            #     raise ValueError("The number of unique classes in 'labels' and 'pred' must match.")
            if len(set(labels)) == 2: 
                average_opt = 'binary'
            else:
                average_opt = 'micro'
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average_opt)
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        else:
            return {}
    
    def fine_tune(self, wdataset, num_epochs):

        # fetch train and test data
        train_data, test_data = wdataset.get_train_test()
        
        # load model and tokenizer and define length of the text sequence, number of classes
        self.num_classes = len(set(train_data.to_pandas()["label"]))
        self.model.config.num_labels = self.num_classes
        print(f"..num_classes = {self.num_classes}")
        # model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels = self.num_classes)
        # tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
        # tokenize_data = lambda batch: tokenizer(batch['text'], padding=True, truncation=True)
        # train_data = train_data.map(tokenize_data, batched=True, batch_size=len(train_data))
        # test_data = test_data.map(tokenize_data, batched=True, batch_size=len(test_data))

        # train_data, test_data = self.tokenize_data(train_data, test_data)
        train_data, test_data = tokenize_data(self.tokenizer, train_data, test_data)
        
        # train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # define the training arguments
        training_args = transformers.TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size = 20,
            gradient_accumulation_steps = 16,
            per_device_eval_batch_size= 8,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end=True,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps = 8,
            fp16 = True,
            logging_dir=self.log_dir,
            dataloader_num_workers = 2,
            disable_tqdm=True # not diplaying a progress bar
        )

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_data,
            eval_dataset=test_data
        )

        print(f"device = {training_args.device}")

        train_res = trainer.train()
        print("..train ended")

        return trainer, train_data, test_data
