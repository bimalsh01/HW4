"""Scaled dot-product attention using NumPy."""

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Apply softmax independently to each row."""
    # Subtracting each row's maximum prevents exponential overflow.
    shifted_values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention weights and the weighted value output."""
    # The last dimension of each key vector is the scaling dimension.
    key_dimension = key.shape[-1]

    # Compare every query with every key by multiplying Q by the transpose of K.
    # Scaling keeps attention scores well-behaved as the key size grows.
    scores = query @ key.T / np.sqrt(key_dimension)

    # Convert each row of scores into probabilities that sum to one.
    attention_weights = softmax(scores)

    # Use the probabilities to compute a weighted combination of value vectors.
    output = attention_weights @ value
    return attention_weights, output


if __name__ == "__main__":
    # These matrices are the test inputs specified in the assignment.
    query = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    key = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    # Run the complete attention pipeline for the test matrices.
    weights, output = scaled_dot_product_attention(query, key, value)

    # Display both required results from the assignment.
    print("Attention Weights:")
    print(weights)
    print("\nOutput:")
    print(output)
