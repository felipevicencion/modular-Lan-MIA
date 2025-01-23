# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# from torchtext.datasets import IMDB
import datasets
import torch

class Dataset:

        def __init__(self, dataset_name):
            self.dataset_name = dataset_name
            self.train_data = None
            self.indices_train = None
            self.test_data = None
            self.indices_test = None

        def get_train_test(self):
            return self.train_data, self.test_data

        def train_test_split(self, use_data_percentage):
             """
            Load a specified dataset, split it into training and testing sets, and subset 
            the data based on a given percentage.
        
            Parameters:
                use_data_percentage (int): The percentage of the dataset to be include in 
                                             both the training and testing splits. 
                                             This value should be between 0 and 100.
            Raises:
                ValueError: If the provided dataset name is not supported.
        
            Attributes Modified:
                - self.train_data: Subset of the member data based on the specified percentage.
                - self.test_data: Subset of the non-member data based on the specified percentage.
                - self.indices_train: Collection of randomly selected indices for the training subset.
                - self.indices_test: Collection of randomly selected indices for the testing subset.
            """

            if self.dataset_name == 'IMDb':

                self.dataset_name_internal = 'imdb'

            elif self.dataset_name == 'Yelp':

                self.dataset_name_internal = 'yelp_review_full'
                
            else:
                
                raise ValueError(f"{self.dataset_name} dataset is not available!")

            train_data, test_data = datasets.load_dataset(self.dataset_name_internal, split =['train', 'test'])

            # subsetting train and test data to 'use_data_percentage'
            num_included_samples_train = int(len(train_data) * float(use_data_percentage) / 100)
            self.indices_train = torch.randperm(len(train_data))[:num_included_samples_train].tolist()
            self.train_data = train_data.select(self.indices_train)

            num_included_samples_test = int(len(test_data) * float(use_data_percentage) / 100)
            self.indices_test = torch.randperm(len(test_data))[:num_included_samples_test].tolist()
            self.test_data = test_data.select(self.indices_test)

        def split_into_k_avoid_ind(k):
            """
            Splits the dataset into `k` subsets while avoiding indices already used 
            in the training and testing split. This method creates shadow datasets 
            for membership inference attacks (MIA).
        
            Parameters:
                k (int): The number of subsets to divide the remaining data into.
        
            Attributes Modified:
                - self.shadow_datasets: A list of `k` shadow datasets, each represented 
                                        as a tuple `(train_data_subset, test_data_subset)`.
            """
            all_indices = np.arange(len(self.train_data))
            valid_indices = np.setdiff1d(all_indices, self.indices_train + self.indices_test)
            shuffled_indices = np.random.permutation(valid_indices)
            subset_size = len(shuffled_indices) // k  # Size of each subset
            splits = [shuffled_indices[i * subset_size: (i + 1) * subset_size] for i in range(k)]
            remainder = shuffled_indices[k * subset_size:]
            for i, idx in enumerate(remainder):
                splits[i] = np.append(splits[i], idx)
            self.shadow_datasets = [(self.train_data[split], self.test_data[split]) for split in splits]

            
        
                
        