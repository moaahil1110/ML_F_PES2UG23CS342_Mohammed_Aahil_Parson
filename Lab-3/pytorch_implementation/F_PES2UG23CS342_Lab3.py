# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    # Get the target column (last column)
    target_column = tensor[:, -1]
    
    # Get unique classes and their counts
    unique_classes, counts = torch.unique(target_column, return_counts=True)
    
    # Calculate total number of samples
    total_samples = tensor.shape[0]
    
    # Calculate probabilities for each class
    probabilities = counts.float() / total_samples
    
    # Calculate entropy: -Σ(p_i * log2(p_i))
    # Handle log(0) case by using torch.where
    log_probs = torch.where(probabilities > 0, torch.log2(probabilities), torch.tensor(0.0))
    entropy = -torch.sum(probabilities * log_probs)
    
    return entropy.item()

    raise NotImplementedError


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    # Get the attribute column
    attribute_column = tensor[:, attribute]
    
    # Get unique values in the attribute
    unique_values = torch.unique(attribute_column)
    
    # Total number of samples
    total_samples = tensor.shape[0]
    
    avg_info = 0.0
    
    # For each unique value in the attribute
    for value in unique_values:
        # Get subset of data where attribute has this value
        mask = attribute_column == value
        subset = tensor[mask]
        
        # Calculate weight (proportion of total data)
        weight = subset.shape[0] / total_samples
        
        # Calculate entropy of this subset
        subset_entropy = get_entropy_of_dataset(subset)
        
        # Add weighted entropy to average info
        avg_info += weight * subset_entropy
    
    return avg_info

    raise NotImplementedError


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    # Calculate entropy of the entire dataset
    dataset_entropy = get_entropy_of_dataset(tensor)
    
    # Calculate average information of the attribute
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    
    # Calculate information gain
    information_gain = dataset_entropy - avg_info
    
    # Round to 4 decimal places
    return round(information_gain, 4)

    raise NotImplementedError


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    # Number of attributes (excluding target column which is last)
    num_attributes = tensor.shape[1] - 1
    
    # Dictionary to store information gains
    information_gains = {}
    
    # Calculate information gain for each attribute
    for attribute in range(num_attributes):
        info_gain = get_information_gain(tensor, attribute)
        information_gains[attribute] = info_gain
    
    # Find attribute with highest information gain
    selected_attribute = max(information_gains, key=information_gains.get)
    
    return (information_gains, selected_attribute)
    
    raise NotImplementedError
