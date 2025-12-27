from Layers.Base import BaseLayer
import numpy as np
from scipy.signal import correlate, correlate2d, convolve, convolve2d


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        self.input_channels = convolution_shape[0]
        self.kernel_shape = convolution_shape[1:]

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer_weights = None
        self._optimizer_bias = None

        self.input_tensor = None
        self.input_shape = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer_obj):
        self._optimizer_weights = optimizer_obj
        import copy
        self._optimizer_bias = copy.deepcopy(optimizer_obj)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.kernel_shape)
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape

        if len(self.convolution_shape) == 2:
            b, c, y = input_tensor.shape
            stride = self.stride_shape[0]
            padding_full = self.kernel_shape[0] - 1
            padding_l = padding_full // 2
            padding_r = padding_full - padding_l
            output_length = (y + padding_full) // stride

            input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (padding_l, padding_r)), mode='constant')

            output = np.zeros((b, self.num_kernels, output_length))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        corr = correlate(input_padded[batch, channel], self.weights[kernel, channel], mode='valid')
                        output[batch, kernel] += corr[::stride]
                    output[batch, kernel] += self.bias[kernel]

        else:
            b, c, y, x = input_tensor.shape
            stride_y, stride_x = self.stride_shape
            _, kernel_y, kernel_x = self.convolution_shape

            padding_x_total = kernel_x - 1
            padding_l = padding_x_total // 2
            padding_r = padding_x_total - padding_l
            padding_y_total = kernel_y - 1
            padding_t = padding_y_total // 2
            padding_b = padding_y_total - padding_t
            input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (padding_t, padding_b), (padding_l, padding_r)), mode='constant')

            output_y = int(np.ceil(y / stride_y))
            output_x = int(np.ceil(x / stride_x))
            output = np.zeros((b, self.num_kernels, output_y, output_x))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        corr = correlate2d(input_padded[batch, channel], self.weights[kernel, channel], mode='valid')
                        output[batch, kernel] += corr[::stride_y, ::stride_x]
                    output[batch, kernel] += self.bias[kernel]

        return output

    def backward(self, error_tensor):

        self._gradient_weights.fill(0)
        self._gradient_bias = np.sum(error_tensor, axis=(0,) + tuple(range(2, error_tensor.ndim)))

        if len(self.convolution_shape) == 2:
            b, c, y = self.input_tensor.shape
            stride = self.stride_shape[0]
            padding_full = self.kernel_shape[0] - 1
            padding_l = padding_full // 2
            padding_r = padding_full - padding_l

            input_padded = np.pad(self.input_tensor, ((0, 0), (0, 0), (padding_l, padding_r)), mode='constant')
            grad_input_padded = np.zeros_like(input_padded)

            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
            self._gradient_weights.fill(0)

            upsampled_error = np.zeros((error_tensor.shape[2] - 1) * stride + 1)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        upsampled_error[::stride] = error_tensor[batch, kernel]
                        self._gradient_weights[kernel, channel] += correlate(input_padded[batch, channel], upsampled_error, mode='valid')
                        grad_input_padded[batch, channel] += convolve(upsampled_error, self.weights[kernel, channel], mode='full')

            grad_input = grad_input_padded[:, :, padding_l:padding_l + y]

        else:
            b, c, y, x = self.input_tensor.shape
            stride_y, stride_x = self.stride_shape
            _, kernel_y, kernel_x = self.convolution_shape

            padding_x_total = kernel_x - 1
            padding_l = padding_x_total // 2
            padding_r = padding_x_total - padding_l
            padding_y_total = kernel_y - 1
            padding_t = padding_y_total // 2
            padding_b = padding_y_total - padding_t
            input_padded = np.pad(self.input_tensor, ((0, 0), (0, 0), (padding_t, padding_b), (padding_l, padding_r)), mode='constant')
            grad_input_padded = np.zeros_like(input_padded)

            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            self._gradient_weights.fill(0)

            up_height = (error_tensor.shape[2] - 1) * stride_y + 1
            up_width = (error_tensor.shape[3] - 1) * stride_x + 1

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    upsampled_error = np.zeros((up_height, up_width))
                    upsampled_error[::stride_y, ::stride_x] = error_tensor[batch, kernel]
                    for channel in range(c):
                        corr = correlate2d(input_padded[batch, channel], upsampled_error, mode='valid')
                        self._gradient_weights[kernel, channel] += corr[:self._gradient_weights.shape[2], :self._gradient_weights.shape[3]]

                        conv = convolve2d(upsampled_error, self.weights[kernel, channel], mode='full')
                        target_h, target_w = grad_input_padded[batch, channel].shape
                        conv_h, conv_w = conv.shape
                        padding_h = max(0, target_h - conv_h)
                        padding_w = max(0, target_w - conv_w)
                        if padding_h > 0 or padding_w > 0:
                            conv = np.pad(conv, ((0, padding_h), (0, padding_w)), mode='constant')
                        conv = conv[:target_h, :target_w]
                        grad_input_padded[batch, channel] += conv

            grad_input = grad_input_padded[:, :, padding_t:padding_t + y, padding_l:padding_l + x]

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return grad_input

