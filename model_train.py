import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
# import torch
import datasets
import pandas as pd
from utils import *
from dataset import *
# from accelerate import Accelerator
import argparse
import os 
import json

class PretrainedModels:

    def __init__(self, input_args):
        model_options_parser = argparse.ArgumentParser(description="model options parser")
        model_options_parser.add_argument("-model_name", "--model-name", type=str, help = "Name of the pretrained NLP model", default = "roberta-base")
        model_options_parser.add_argument("-max_sequence_length", "--max-sequence-length", type=int, help = "Maximum sequence length", default = 512)
        model_options_parser.add_argument("-log_dir", "--log-dir", type=str, help = "Location of the logfile", default = None)
        model_options_parser.add_argument("-output_dir", "--output-dir", type=str, help = "Location of the output directory", default = None)  
        model_options_parser.add_argument("-load_model_dir", "--load-model-dir", type=str, help = "Directory from which a trained model is loaded", default = None)
        model_options_parser.add_argument("-use_float16", "--use-float16", type=bool, help = "Whether or not to store weight in 16 bit floats", default = False)
        model_options_parser.add_argument("-task", "--task", type=str, help = "Describes the task which a model was trained for: 'classification' or 'generative'", default = "classification")
        model_options_parser.add_argument("-num_epochs", "--num-epochs", type=int, help = "Number of epochs used in the fine-tuning of the model", default = 5)
        model_options_parser.add_argument("-per_device_train_batch_size", "--per-device-train-batch-size", type=int, help = "Per device train batch size", default = 16)
        model_options_parser.add_argument("-per_device_eval_batch_size", "--per-device-eval-batch-size", type=int, help = "Per device eval batch size", default = 16)
        model_options_parser.add_argument("-learning_rate", "--learning-rate", type=float, help = "Learning rate used in fine-tunning", default = 2e-5)
        model_options_parser.add_argument("-weight_decay", "--weight-decay", type=float, help = "Weight decay parameter used in fine-tunning", default = 0.01)
        model_options_parser.add_argument("-warmup_ratio", "--warmup-ratio", type=float, help = "Warmup ratio parameter used in fine-tunning", default = 0.1)

        model_args, _ = model_options_parser.parse_known_args(input_args)
        self.model_name = model_args.model_name
        self.max_sequence_length = model_args.max_sequence_length
        self.output_dir = model_args.output_dir
        self.log_dir = model_args.log_dir
        self.load_model_dir = model_args.load_model_dir
        self.use_float16 = model_args.use_float16
        self.task = model_args.task
        self.num_epochs = model_args.num_epochs
        self.per_device_train_batch_size = model_args.per_device_train_batch_size
        self.per_device_eval_batch_size = model_args.per_device_eval_batch_size
        self.learning_rate = model_args.learning_rate
        self.weight_decay = model_args.weight_decay
        self.warmup_ratio = model_args.warmup_ratio
        # self.model_name = model_options['model_name']
        # self.max_sequence_length = model_options['max_sequence_length']
        # self.output_dir = model_options['output_dir']
        # self.log_dir = model_options['log_dir']
        # self.load_model_dir = model_options['load_model_dir']
        # self.use_float16 = model_options['use_float16']
        # self.task = model_options['task']
        self.load_specific_model()
        if self.task == 'generative':
            self.model.eval()
        
    def load_specific_model(self):
        """
        Load a specific model and tokenizer based on the task type and model name.
    
        Attributes Modified:
            - self.model: The loaded model object, either for classification or generative tasks.
            - self.tokenizer: The corresponding tokenizer for the loaded model.
            
        Raises:
            Exception: If an error occurs during model or tokenizer loading.
        """
        if self.task == 'classification':
            try:
                self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_model_dir)
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                # self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(self.model_name, max_length = self.max_sequence_length)
            except Exception as e:
                print(f"Error: {e}")
                print("Ensure you are using the correct directory for loading the model.")
            print(f"..model {self.model_name} is loaded")
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
                # piece of code for testing the part above
                # inputs = self.tokenizer("Hello, square root of -4 is", return_tensors="pt")
                # tokens = self.model.generate(**inputs, max_length=30, num_return_sequences=1)
                # print(self.tokenizer.decode(tokens[0]))

    def prepare_model(self, wdataset):
        """
        Parameters:
            wdataset: A dataset object 
        Returns:
            tuple: A tuple containing:
                   - trainer (transformers.Trainer): Configured Trainer object for model training.
                   - model: The prepared model.
                   - tokenizer: The corresponding tokenizer.
        """
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
        """
        Compute evaluation metrics for the model based on predictions.
    
        Parameters:
            pred: Predictions object containing `label_ids` and `predictions`.
    
        Returns:
            dict: A dictionary containing the calculated metrics:
                  - 'accuracy': Accuracy of predictions.
                  - 'f1': F1 score of predictions.
                  - 'precision': Precision of predictions.
                  - 'recall': Recall of predictions.
                  If the task is not classification, returns an empty dictionary.
        """
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
    
    def fine_tune(self, wdataset, idx_shadow = None):
        """
        Fine-tune a pre-trained model on a given dataset for a specified number of epochs.
    
        Parameters:
            wdataset: A dataset object 
    
        Returns:
            tuple: A tuple containing:
                   - trainer (transformers.Trainer): The Trainer object used for fine-tuning.
                   - train_data: Tokenized training dataset.
                   - test_data: Tokenized testing dataset.
        """

        # fetch train and test data
        train_data, test_data = wdataset.get_train_test()
        if idx_shadow is not None:
            train_data = wdataset.shadow_datasets[idx_shadow]
        
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
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size = self.per_device_train_batch_size,
            per_device_eval_batch_size= self.per_device_eval_batch_size,
            gradient_accumulation_steps = 1,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end=True,
            warmup_ratio = self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps = 8,
            fp16 = True,
            logging_dir=self.log_dir,
            dataloader_num_workers = 2,
            lr_scheduler_type="linear",
            learning_rate = self.learning_rate,
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

        # Create fine-tuning details object based on instance attributes
        fine_tuning_details = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": 1,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": 8,
            "fp16": True,
            "logging_dir": self.log_dir,
            "dataloader_num_workers": 2,
            "lr_scheduler_type": "linear",
            "learning_rate": self.learning_rate,
            "disable_tqdm": True 
        }

        # Define output file path for saving
        output_file_path = os.path.join(self.output_dir, "fine_tuning_details.json")
        os.makedirs(self.output_dir, exist_ok=True)
        # Save fine-tuning details to JSON file
        with open(output_file_path, "w") as json_file:
            json.dump(fine_tuning_details, json_file, indent=4)
        print(f"..fine-tuning details saved to {output_file_path}")

        return trainer, train_data, test_data
