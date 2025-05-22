import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    Q = np.dot(X ,W_q)
    K = np.dot(X ,W_k)
    V = np.dot(X ,W_v)
    return Q, K, V



def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # The mask: (a causal mask)
    # mask = np.triu(np.ones((4, 4))*(-np.inf), k=1)
    # Upper triangular:
    # [[  0., -inf, -inf, -inf],
    # [  0.,   0., -inf, -inf],
    # [  0.,   0.,   0., -inf],
    # [  0.,   0.,   0.,   0.]]
    # k=1 means we keep only the strictly upper triangular part (above the main diagonal).
    # Everything on or below the diagonal is set to 0.
    # -inf ensures softmax probabilities become 0 for masked positions.
    # k=1 ensures the diagonal is kept (allows positions to attend to themselves).

    score = np.dot(Q, K.T) / np.sqrt(K.shape[0])
    masked_score = score + mask
    s = masked_score - np.max(masked_score, axis=1, keepdims=True)
    exp_values = np.exp(s)
    sofmax_score = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    output = np.dot(sofmax_score, V)
    return output

#running the masked self attention mechanism:
np.random.seed(42)
X = np.arange(16).reshape(4,4)
X = np.random.permutation(X.flatten()).reshape(4, 4)
mask = np.triu(np.ones((4, 4))*(-np.inf), k=1)
W_q = np.random.randint(0,4,size=(4,4))
W_k = np.random.randint(0,5,size=(4,4))
W_v = np.random.randint(0,6,size=(4,4))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
#expected output:
# [[ 52. 63. 48. 71.] [103. 109. 46. 99.] [103. 109. 46. 99.] [103. 109. 46. 99.]]
print(masked_attention(Q, K, V, mask))