import numpy as np

def generate_large_scale_data(num_samples, num_vectors, dimensions):
    return [np.random.randn(num_vectors, dimensions) for _ in range(num_samples)]
