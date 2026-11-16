import numpy as np

def scaled_dot_product_attention(Q, K, V):

    # Get the key dimension
    d = K.shape[1]

    # Calculate dot product of Q and K transpose
    scores = np.dot(Q, K.T)

    # Scale the attention scores
    scores = scores / np.sqrt(d)

    # Apply softmax
    exp_scores = np.exp(
        scores - np.max(scores, axis=1, keepdims=True)
    )

    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Multiply attention weights by V
    output = np.dot(attention_weights, V)

    return attention_weights, output


# Given input matrices
Q = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

K = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

V = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])

# Calculate attention
weights, output = scaled_dot_product_attention(Q, K, V)

# Display results
print("Attention Weights:")
print(weights)

print("\nFinal Output:")
print(output)