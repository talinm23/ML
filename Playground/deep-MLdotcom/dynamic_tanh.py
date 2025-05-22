
# Dynamic Tanh: Normalization-Free Transformer Activation:
# A new study (https://arxiv.org/pdf/2503.106220) demonstrates that layer normalization, that is ubiquitous in transformers,
# produces Tanh-like S-shapes. By incorporating a new layer replacement for normalization
# called "Dynamic Tanh" (DyT for short), Transformers without normalization can match or
# exceed the performance of their normalized counterparts, mostly without hyperparameter tuning.
# Read more: https://www.deep-ml.com/problems/128
import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    tanh_calc=(np.exp(alpha*x)-np.exp(-alpha*x))/(np.exp(alpha*x)+np.exp(-alpha*x))
    dynamic_tanh = gamma * tanh_calc + beta
    return dynamic_tanh


x = np.array([[[0.94378259]],[[0.97754654]],[[0.36168351]],[[0.51821078]],[[0.76961589]]])
gamma = np.ones((1,))
beta = np.zeros((1,))
# Expected output:
# [[[0.4397]], [[0.4532]], [[0.1789]], [[0.2535]], [[0.3669]]]
print(np.round(dynamic_tanh(x, 0.5, gamma, beta),4))
