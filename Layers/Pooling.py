from Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.maxima = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b, c, y, x = input_tensor.shape
        pool_h, pool_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        pools = np.lib.stride_tricks.sliding_window_view(input_tensor, window_shape=(pool_h, pool_w), axis=(2, 3))
        pools = pools[:, :, ::stride_y, ::stride_x, :, :]
        flat_pool = pools.reshape(*pools.shape[:4], -1)
        output = np.max(flat_pool, axis=-1)

        indices = np.argmax(flat_pool, axis=-1)
        maxima = np.zeros_like(flat_pool)
        np.put_along_axis(maxima, indices[..., None], 1, axis=-1)
        self.maxima = maxima.reshape(*pools.shape)

        return output

    def backward(self, error_tensor):
        b, c, y, x = self.input_tensor.shape
        pool_h, pool_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        out_y = (y - pool_h) // stride_y + 1
        out_x = (x - pool_w) // stride_x + 1
        grad_input = np.zeros_like(self.input_tensor)
        grad_pools = self.maxima * error_tensor[:, :, :, :, None, None]

        for i in range(out_y):
            for j in range(out_x):
                y = i * stride_y
                x = j * stride_x
                grad_input[:, :, y:y + pool_h, x:x + pool_w] += grad_pools[:, :, i, j, :, :]

        return grad_input
