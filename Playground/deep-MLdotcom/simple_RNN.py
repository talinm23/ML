
import numpy as np
def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx,x)+np.dot(Wh,h)+b)
    final_hidden_state = np.round(h,4)
    return final_hidden_state #.tolist()

print(rnn_forward([[0.5], [0.1], [-0.2]], [0.0], [[1.0]], [[0.5]], [0.1]))
#expected output:
#[0.118]
print(rnn_forward( [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0.0, 0.0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0]], [0.1, 0.2] ))
#expected output:
#[0.7474, 0.9302]