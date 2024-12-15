import numpy as np

def find_optimal_subsequence(vector):
    """
    Finds the optimal subsequence of a vector that minimizes variance.
    Returns the indices of the subsequence.
    """
    best_variance = float('inf')
    best_indices = []

    for start in range(len(vector)):
        for end in range(start + 1, len(vector) + 1):
            subsequence = vector[start:end]
            variance = np.var(subsequence)
            if variance < best_variance:
                best_variance = variance
                best_indices = list(range(start, end))

    return best_indices

def compute_exact_signs(vector):
    """
    Computes the exact sign for a vector based on the optimal subsequence
    that minimizes variance.
    """
    optimal_indices = find_optimal_subsequence(vector)
    optimal_subsequence = vector[optimal_indices]
    subsequence_mean = np.mean(optimal_subsequence)
    return np.sign(subsequence_mean)

def compute_signs_for_dataset(data):
    """
    Computes exact signs for each vector in a dataset.
    """
    return np.array([compute_exact_signs(vector) for vector in data])
