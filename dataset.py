import datasets
import torch
import argparse
import numpy as np

class Dataset:

        def __init__(self, input_args):
            dataset_options_parser = argparse.ArgumentParser(description="dataset options parser")
            dataset_options_parser.add_argument("-datatset_path", "--dataset-path", type=str, help="Path to the HF dataset, either a local directory or a dataset repository on the HF hub (list all available datasets with huggingface_hub.list_datasets)", default='imdb')
            dataset_options_parser.add_argument("-dataset_name", "--dataset-name", type=str, help="Defining the name of the dataset configuration", default=None)
            dataset_options_parser.add_argument("-use_data_percentage", "--use-data-percentage", type=int, help="Defining the percentage of the data to be used", default=100)
            dataset_args, _ = dataset_options_parser.parse_known_args(input_args)
            self.dataset_path = dataset_args.dataset_path
            self.dataset_name = dataset_args.dataset_name
            self.use_data_percentage = dataset_args.use_data_percentage
            self.train_data = None # same as member data
            self.indices_train = None
            self.test_data = None # same as non-member data
            self.indices_test = None

        def get_train_test(self):
            return self.train_data, self.test_data

        def add_sample(self, target_dataset, sample):
            """
            Add a sample to the specified dataset.
    
            Parameters:
                sample: The sample to add.
                target_dataset: The dataset to add the sample to train.
            """
            combined_dataset = datasets.concatenate_datasets([target_dataset, sample])
            return combined_dataset
            
        def train_test_split(self):
            """
            Load a specified dataset, split it into training and testing sets, and subset 
            the data based on a given percentage.
        
            Raises:
                ValueError: If the provided dataset name is not supported.
        
            Attributes Modified:
                - self.train_data: Subset of the member data based on the specified percentage.
                - self.test_data: Subset of the non-member data based on the specified percentage.
                - self.indices_train: Collection of randomly selected indices for the training subset.
                - self.indices_test: Collection of randomly selected indices for the testing subset.
            """

            print(f"..loading dataset {self.dataset_path}")
            if self.dataset_name is not None:
                train_data, test_data = datasets.load_dataset(path = self.dataset_path, name = self.dataset_name, split =['train', 'test'])
            else:
                train_data, test_data = datasets.load_dataset(self.dataset_path, split =['train', 'test'])
            # subsetting train and test data to 'use_data_percentage'
            num_included_samples_train = int(len(train_data) * float(self.use_data_percentage) / 100)
            self.indices_train = torch.randperm(len(train_data))[:num_included_samples_train].tolist()
            self.train_data = train_data.select(self.indices_train)
            num_included_samples_test = int(len(test_data) * float(self.use_data_percentage) / 100)
            self.indices_test = torch.randperm(len(test_data))[:num_included_samples_test].tolist()
            self.test_data = test_data.select(self.indices_test)

        def split_into_k_avoid_ind(self, k):
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

        def sample_n_in_k(self, k, samples_percentage):
            """
            Samples `n` data points for each of the `k` subsets. The subsets are not necessarily disjoint.
    
            Parameters:
                k (int): The number of subsets to generate.
                samples_percentage (int): Percentage of samples in each subset.
    
            Attributes Modified:
                - self.shadow_datasets: A list of `k` shadow datasets.
            """
            combined_dataset = datasets.concatenate_datasets([self.train_data, self.test_data])
            self.shadow_datasets = []
            for _ in range(k):
                sampled_indices = np.random.choice(np.arange(len(combined_dataset)), size=int( float(len(combined_dataset) * samples_percentage) / 100 ), replace=False)
                self.shadow_datasets.append(combined_dataset.select(sampled_indices)) 


        def deduplicate_dataset(self, dataset):
            """
            Removes duplicate rows from a Hugging Face datasets.Dataset object.

            Args:
                dataset (Dataset): The input dataset to deduplicate.

            Returns:
                Dataset: A new dataset with duplicates removed.
            """
            seen = set()

            def is_unique(example):
                example_tuple = tuple(tuple(example[key]) if isinstance(example[key], list) else example[key] for key in dataset.column_names)
                
                if example_tuple in seen:
                    return False  # Duplicate, remove
                seen.add(example_tuple)
                return True  # Unique, keep

            unique_dataset = dataset.filter(is_unique)
            return unique_dataset


        def distribute_to_shadow_datasets(self, attack_dataset):
            """
            Distribute each sample in the attack dataset to exactly half of the shadow datasets (stored in self.shadow_datasets as a list).
            
            Parameters:
                attack_dataset (list): dataset object containing samples on which membership inference attack is being performed.
                
            Returns:
                in_indices (list): Indices of the datasets that were chosen for a sample.
                out_indices (list): Indices of the datasets that were not chosen for a sample.
            """
            num_shadow_datasets = len(self.shadow_datasets)
            print(num_shadow_datasets)
            if num_shadow_datasets <= 1:
                raise ValueError("Number of shadow datasets should be >= 2.")
            in_indices = []  
            out_indices = []  
            print(f"attack_dataset type = {type(attack_dataset)}")
            for _ in range(len(attack_dataset)):
                half_datasets = np.random.choice(num_shadow_datasets, size=num_shadow_datasets // 2, replace=False)
                out_indices_sample = [idx for idx in range(num_shadow_datasets) if idx not in half_datasets]
                in_indices.append(half_datasets.tolist())
                out_indices.append(out_indices_sample)
            dataset_indices = [[] for _ in range(num_shadow_datasets)]  
            for sample_id, dataset_list in enumerate(in_indices):  
                for dataset_id in dataset_list:  
                    dataset_indices[dataset_id].append(sample_id)  
            dataset_indices = [sorted(indices) for indices in dataset_indices]
            for idx in half_datasets:
                samples = attack_dataset.select(dataset_indices[idx])
                self.shadow_datasets[idx] = self.add_sample(self.shadow_datasets[idx], samples)
            for idx in half_datasets:
                self.shadow_datasets[idx] = self.deduplicate_dataset(self.shadow_datasets[idx])
            return in_indices, out_indices



            
        
                
        