import torch
import numpy as np
from transformers import set_seed
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class FlexibleBinaryClassifier(nn.Module):
    def __init__(self, layer_sizes):
        super(FlexibleBinaryClassifier, self).__init__()
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])) 
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        
        layers.append(torch.nn.Sigmoid())
    
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_binary_classifier(layer_sizes, X_train, y_train, num_epochs=10, batch_size=16, learning_rate=0.001):
    model = FlexibleBinaryClassifier(layer_sizes)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    return model

    # example of usage of train_binary_classifier:
    # layer_sizes = [5, 10, 5, 1] 
    # model = train_binary_classifier(layer_sizes, X_train, y_train, num_epochs=10, batch_size=16, learning_rate=0.001)
    # model.eval()
    # new_sample = torch.randn(1, 5) 
    # with torch.no_grad():
    #     prediction = model(new_sample)
    #     print(f"Predicted probability: {prediction.item():.4f}")
    #     predicted_class = (prediction >= 0.5).float()
    #     print(f"Predicted class: {predicted_class.item()}")


# function that sets seed for all the libraries used
def fix_seed(seed: int = 0):
    """
    Sets a fixed seed for reproducibility in random number generation for PyTorch, NumPy, 
    and other libraries that depend on randomization.

    Parameters:
        seed (int): The seed value to use. Default is 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

def get_all_prob(input_ids, loss, logits):
    """
    Calculates various probabilities, log-likelihoods, and perplexities based on the model's outputs.

    Parameters:
        input_ids (torch.Tensor): Tensor of input token IDs.
        loss (torch.Tensor): Loss tensor from the model's output.
        logits (torch.Tensor): Logits tensor from the model's output.

    Returns:
        tuple: A tuple containing:
            - prob (float): Exponentiated negative loss (probability of the sequence).
            - ll (float): Log-likelihood of the sequence.
            - ppl (float): Perplexity of the sequence.
            - all_prob (list): List of probabilities for each token.
            - loss (float): The raw loss value.
    """
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()

def get_ll(input_ids, model):
    """
    Calculates the log-likelihood, probabilities, and perplexity for a given sentence.

    Parameters:
        inputs (torch.tensor): The input sequence for which to compute the metrics.
        model: The pre-trained language model.
        tokenizer: Tokenizer associated with the model.
        device: The device (CPU/GPU) for computation.

    Returns:
        tuple: See `get_all_prob` for details.
    """
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return get_all_prob(input_ids, loss, logits)

# function to tokenize train and test data
def tokenize_data(tokenizer, data, max_sequence_length):
    """
    Tokenizes the dataset only if it has not been tokenized.

    Parameters:
        tokenizer: Tokenizer to convert text into input IDs, attention masks, etc.
        data: Dataset to be tokenized.
        max_sequence_length (int): Maximum token length for truncation and padding.

    Returns:
        Dataset: Tokenized dataset.
    """
    # Check if dataset is already tokenized
    if all(col in data.column_names for col in ['input_ids', 'attention_mask']):
        return data  # Return as is if already tokenized

    tokenize_data = lambda batch: tokenizer(batch['text'], max_length=max_sequence_length, padding=True, truncation=True)
    data = data.map(tokenize_data, batched=True, batch_size=len(data))
    # data = data.remove_columns(['text']) # some of the attacks need access to original text
    data.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    return data

def capitalize_after_underscore(input_string):
    """
    Capitalizes each word in a string that is separated by underscores and afterwards removes all underscores.

    Parameters:
        input_string (str): The input string containing words separated by underscores.

    Returns:
        str: A string with each word capitalized and the underscores removed.
    """
    # Split the string by underscores, capitalize each part, and join them back
    return ''.join(word.capitalize() for word in input_string.split('_'))

def parse_thresholds(thr_str):
    """Parses threshold input from command line."""
    if "," in thr_str:
        thr_list = []
        for part in thr_str.split(","):
            if ":" in part:
                a, b, c = map(float, part.split(":"))
                thr_list.extend(np.arange(a, b, c))
            else:
                thr_list.append(float(part))
        return thr_list
    else:
        if ":" in thr_str:
            thr_list = []
            a, b, c = map(float, thr_str.split(":"))
            thr_list.extend(np.arange(a, b, c))
            return thr_list
        else:
            return float(thr_str)


