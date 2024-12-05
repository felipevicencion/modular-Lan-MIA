# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# from torchtext.datasets import IMDB
import datasets
import torch

class dataset:

        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

        def train_test_split(self, use_data_percentage):

            if self.dataset_name == 'IMDb':

                self.dataset_name_internal = 'imdb'

            elif self.dataset_name == 'Yelp':

                self.dataset_name_internal = 'yelp_review_full'
                
            else:
                
                raise ValueError(f"{self.dataset_name} dataset is not available!")

            train_data, test_data = datasets.load_dataset(self.dataset_name_internal, split =['train', 'test'])

            # subsetting train and test data to 'use_data_percentage'
            num_included_samples_train = int(len(train_data) * float(use_data_percentage) / 100)
            indices_train = torch.randperm(len(train_data))[:num_included_samples_train].tolist()
            train_data = train_data.select(indices_train)

            num_included_samples_test = int(len(test_data) * float(use_data_percentage) / 100)
            indices_test = torch.randperm(len(test_data))[:num_included_samples_test].tolist()
            test_data = test_data.select(indices_test)

            return train_data, test_data
            
                
                
        