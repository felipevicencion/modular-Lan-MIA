import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets
import pandas as pd
from utils import *

class pretrained_models:

    def __init__(self, model_name, max_sequence_length, load_model_dir, output_dir=None, log_dir=None):
        # model_name = 'roberta_base'
        # max_sequence_length = 512
        # log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
        # output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.load_model_dir = load_model_dir
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        if self.num_classes == 2:
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

    # def tokenize_data(self, train_data, test_data):

    #     tokenize_data = lambda batch: self.tokenizer(batch['text'], padding=True, truncation=True)
    #     train_data = train_data.map(tokenize_data, batched=True, batch_size=len(train_data))
    #     test_data = test_data.map(tokenize_data, batched=True, batch_size=len(test_data))
    #     train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    #     test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    #     return train_data, test_data

    def load_model(self, train_data, test_data):

        _, uniques = pd.factorize(train_data.to_pandas()["label"])
        self.num_classes = len(uniques)
        self.model.config.num_labels = self.num_classes
        print(f"..num_classes = {self.num_classes}")
        # train_data, test_data = self.tokenize_data(train_data, test_data)
        train_data, test_data = tokenize_data(self.tokenizer, train_data, test_data)
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_model_dir)
        except Exception as e:
            print(f"Error: {e}")
            print("Ensure you are using the correct directory for loading the model.")
            print(f"..model {model} is loaded")
 
        trainer = transformers.Trainer(
            model=model,
            compute_metrics=self.compute_metrics,
            train_dataset=train_data,
            eval_dataset=test_data
        )
        return trainer, model, self.tokenizer
    
    def fine_tune(self, train_data, test_data, num_epochs):
        
        # load model and tokenizer and define length of the text sequence, number of classes
        _, uniques = pd.factorize(train_data.to_pandas()["label"])
        self.num_classes = len(uniques)
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device = {device}")
        self.model.to(device)

        train_res = trainer.train()
        print("..train ended")
        # eval_res = trainer.evaluate()
        # print("..eval ended")

        return trainer, train_data, test_data
