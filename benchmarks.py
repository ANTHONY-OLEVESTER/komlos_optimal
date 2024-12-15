import time
from .core import compute_signs_for_dataset, compute_exact_signs
import numpy as np

def validate_algorithms(data_samples):
    """
    Validates that the optimal subsequence approach computes correct signs.
    """
    all_match = True

    for data in data_samples:
        # Compute signs using the optimal subsequence algorithm
        optimal_signs = compute_signs_for_dataset(data)

        # Compute brute force signs (for validation)
        brute_force_signs = np.array([compute_exact_signs(vector) for vector in data])

        # Validate results
        if not np.array_equal(optimal_signs, brute_force_signs):
            all_match = False
            break

    return all_match

def benchmark_algorithms(num_samples, num_vectors, dimensions):
    """
    Benchmarks the optimal subsequence algorithm.
    """
    from .utils import generate_large_scale_data

    data_samples = generate_large_scale_data(num_samples, num_vectors, dimensions)
    total_time = 0

    for data in data_samples:
        # Benchmark the optimal subsequence algorithm
        start_time = time.time()
        compute_signs_for_dataset(data)
        total_time += time.time() - start_time

    print(f"Total Time: {total_time:.2f} seconds")
    return total_time
