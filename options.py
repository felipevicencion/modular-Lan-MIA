import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # general arguments
        self.parser.add_argument("-dataset", "--dataset", help = "specifies a dataset on which an attack is to be performed", default = "IMDb")
        self.parser.add_argument("-torch_num_threads", "--torch-num-threads", help = "number of CPU threads PyTorch uses for parallel operations", default = 1)
        self.parser.add_argument("-model_name", "--model-name", help = "name of the pretrained NLP model", default = "roberta-base")
        self.parser.add_argument("-max_sequence_length", "--max-sequence-length", help = "maximum sequence length", default = 512)
        self.parser.add_argument("-seed", "--seed", help = "setting seed value for all random calculations", default = 0)
        
        ## fine-tuning parameters
        self.parser.add_argument("-num_epochs", "--num-epochs", help = "number of epochs used in the fine-tuning of the model", default = 5)
        self.parser.add_argument("-load_model_dir", "--load-model-dir", help = "directory from which a trained model is loaded", default = None)
        self.parser.add_argument("-log_dir", "--log-dir", help = "location of the logfile", default = None)
        self.parser.add_argument("-output_dir", "--output-dir", help = "location of the output directory", default = None)  
        self.parser.add_argument("--use_data_percentage", "--use-data-percentage", help = "use only a certain percentage of the samples for fine-tunning", default = 100)  
        self.parser.add_argument("--num_shadow_models", "--num-shadow-models", help = "use only a certain percentage of the training samples for fine-tunning", default = 100)
        ## attack parameters
        self.parser.add_argument("-attack_name", "--attack-name", help = "name of an attack to be performed", default = "prediction_loss_based_mia")
        self.parser.add_argument("-num_trials", "--num-trials", help = "number of trials of an attack to be performed", default = 500)
        self.parser.add_argument("-score_thr", "--score-thr", help = "threshold to be used for attacks based on scores", default = 1.1)
        # attack: recall parameters 
        self.parser.add_argument("-num_shots", "--num-shots", help = "attack ReCall: number of shots for prefix", default = 1)
        self.parser.add_argument("-pass_window", "--pass-window", help = "attack ReCall: (Optional) exceed the context window", default = True)
        self.parser.add_argument("-max_length", "--max-length", help = "attack ReCall: Maximum length of the window", default = 512)
        self.parser.add_argument("-prefix", "--prefix", help = "attack ReCall: a prefix to be used in an attack", default = "This is a synthetic prefix!")
        # run mode parameter
        self.parser.add_argument("-run_mode", "--run-mode", help = "specifies the mode in which the package is run - 'fine-tunning' or 'attack'", default = 'attack')