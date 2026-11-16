
import numpy as np

# Define the softmax function
def softmax(x):
    # Subtract the maximum value for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    # Normalize the exponential values
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Implement scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):

    # Get the key dimension
    d_k = K.shape[1]

    # Step 1: Calculate the dot product of Q and K transpose
    scores = np.matmul(Q, K.T)

    # Step 2: Scale the attention scores by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply softmax to obtain attention weights
    attention_weights = softmax(scaled_scores)

    # Step 4: Multiply attention weights by V
    output = np.matmul(attention_weights, V)

    return attention_weights, output


# Given test inputs
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

# Display the results
print("Attention Weights:")
print(weights)

print("\nFinal Output:")
print(output)
