import numpy as np


def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
    input_height_padded, input_width_padded = padded_input.shape

    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
            output_matrix[i, j] = np.sum(region * kernel)
    return output_matrix

#running the code:
input_matrix = np.array([ [1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], ])
kernel = np.array([ [1., 2.], [3., -1.], ])
padding, stride = 0, 1
#expected output:
#expected = np.array([ [ 16., 21., 26., 31.], [ 41., 46., 51., 56.], [ 66., 71., 76., 81.], [ 91., 96., 101., 106.], ])
output1 = simple_conv2d(input_matrix, kernel, padding, stride)
print('output1\n',output1)

input_matrix = np.array([ [1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], ])
kernel = np.array([ [.5, 3.2], [1., -1.], ])
padding, stride = 2, 2
expected = np.array([[-1., 1., 3., 5., 7., 15.], [ -4., 16., 21., 26., 31., 35.], [ 1., 41., 46., 51., 56., 55.], [ 6., 66., 71., 76., 81., 75.], [ 11., 91., 96., 101., 106., 95.], [ 42., 65., 68., 71., 74., 25.], ])
output2 = simple_conv2d(input_matrix, kernel, padding, stride)
print('output2\n',output2)


