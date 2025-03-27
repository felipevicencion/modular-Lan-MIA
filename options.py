import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # general arguments
        # self.parser.add_argument("-dataset", "--dataset", help = "specifies a dataset on which an attack is to be performed", default = "IMDb")
        self.parser.add_argument("-torch_num_threads", "--torch-num-threads", help = "number of CPU threads PyTorch uses for parallel operations", default = 1)
        # self.parser.add_argument("-model_name", "--model-name", help = "name of the pretrained NLP model", default = "roberta-base")
        self.parser.add_argument("-task", "--task", help = "descibes the task which a model was trained for: 'classification' or 'causal'", default = "classification")
        # self.parser.add_argument("-max_sequence_length", "--max-sequence-length", help = "maximum sequence length", default = 512)
        self.parser.add_argument("-seed", "--seed", help = "setting seed value for all random calculations", default = 1)

        ## fine-tuning parameters
        # self.parser.add_argument("-num_epochs", "--num-epochs", help = "number of epochs used in the fine-tuning of the model", default = 5)
        self.parser.add_argument("-load_model_dir", "--load-model-dir", help = "directory from which a trained model is loaded", default = None)
        self.parser.add_argument("-log_dir", "--log-dir", help = "location of the logfile", default = None)
        self.parser.add_argument("-output_dir", "--output-dir", help = "location of the output directory", default = None)  
        self.parser.add_argument("--use_data_percentage", "--use-data-percentage", help = "use only a certain percentage of the samples for fine-tunning", default = 100)  
       
        ## attack parameters
        self.parser.add_argument("-attack_name", "--attack-name", help = "name of an attack to be performed", default = "prediction_loss_based_mia")
    
        # run mode parameter
        self.parser.add_argument("-run_mode", "--run-mode", help = "specifies the mode in which the package is run - 'fine-tunning' or 'attack'", default = 'attack')