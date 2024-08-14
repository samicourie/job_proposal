
def apply_weights(dataset, weights):
    """
    Apply weights to the dataset.

    Parameters:
    - dataset (2d array-like): dataset to apply weights on.
    - weights (array-like or dict): features weights.

    Returns:
    - weighted dataset.
    """

    if isinstance(weights, dict):
        for key in weights.keys():
            dataset[key] *= weights[key]

    else:
        for ind in range(len(weights)):
            dataset[:, ind] *= weights[ind]

    return dataset
