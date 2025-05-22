
import numpy as np
def  compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X ,W_q)
    K = np.dot(X ,W_k)
    V = np.dot(X ,W_v)
    return Q, K, V


def self_attention(Q, K, V):
    s = np.dot(Q ,K.T ) /np.sqrt(K.shape[0])
    exp_values = np.exp(s)
    softmax_probabilities = exp_values /np.sum(exp_values ,axis=1 ,keepdims=True)
    # np.sum(exp_values) sums over all elements in the matrix, but softmax should be applied row-wise (for each query's attention scores).
    # Use np.sum(exp_values, axis=1, keepdims=True) to normalize each row independently.
    attention_output = np.dot(softmax_probabilities ,V)
    return attention_output



#running the self attention mechanism:
X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output1 = self_attention(Q, K, V)
# output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
print('output1\n',output1)

X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
W_q = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_k = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output2 = self_attention(Q, K, V)
#output:
# [[8.0, 10.0, 12.0],
# [8.61987385, 10.61987385, 12.61987385],
# [7.38012615, 9.38012615, 11.38012615]]
print('output2\n',output2)
