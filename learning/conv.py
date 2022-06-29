from turtle import shape
from cv2 import _OUTPUT_ARRAY_DEPTH_MASK_16U
import numpy as np

class Conv(object):
    def __init__(self, inputs, weights, stride=1, padding=1) -> None:
        '''
        inputs: b * c_in * h * w
        weights: c_in * k * k * c_out
        '''
        self.inputs = np.asarray(inputs, np.float32)
        self.weights = np.asarray(weights, np.float32)
        self.stride = stride
        self.padding = padding

    def conv2d(self):
        b, c_in, h, w = self.inputs.shape
        c_out, _, k, _  = self.weights.shape
        outputs = []
        for i in range(b):
            output = []
            for j in range(c_out):
                fea = self.inputs[i]
                kernel = self.weights[j]
                out = self.compute(fea, kernel)
                output.append(out)
            outputs.append(output)
        return np.asarray(outputs, np.float32)

    def compute(self, fea, kernel):
        c_in, h, w = fea.shape
        *_, k = kernel.shape
        fea_pad = np.zeros([c_in, h + 2 * self.padding, w + 2 * self.padding], np.float32)
        fea_pad[:, self.padding:h+self.padding, self.padding:w+self.padding] = fea
        out = np.zeros([(h + 2 * self.padding - k) // self.stride + 1,
                        (w + 2 * self.padding - k) // self.stride + 1], np.float32)
        for i in range((h + 2 * self.padding - k) // self.stride + 1):
            for j in range((w + 2 * self.padding - k) // self.stride + 1):
                fea_win = fea_pad[:, i * self.stride : i * self.stride + k, j * self.stride : j * self.stride + k]
                out[i, j] = np.sum(fea_win * kernel)
        return out
    
    def conv2d_mat(self):
        b, c_in, h, w = self.inputs.shape
        c_out, _, k, _  = self.weights.shape

        fea_pad = np.zeros([b, c_in, h + 2 * self.padding, w + 2 * self.padding], np.float32)
        fea_pad[..., self.padding:h+self.padding, self.padding:w+self.padding] = self.inputs

        outputs = []
        for m in range(b):
            output = []
            for n in range(c_out):
                kernel = self.weights[n]
                kernel = kernel.reshape(c_in * k * k, 1)
                h_out = (h + 2 * self.padding - k) // self.stride + 1
                w_out = (w + 2 * self.padding - k) // self.stride + 1
                mat = np.zeros([h_out * w_out, c_in * k * k], np.float32)
                row = 0
                for i in range(h_out):
                    for j in range(w_out):
                        fea_win = fea_pad[m, :, i * self.stride : i * self.stride + k, j * self.stride : j * self.stride + k]
                        mat[row] = fea_win.flatten()
                        row += 1
                out = np.dot(mat, kernel).reshape(h_out, w_out)
                output.append(out)
            outputs.append(output)
        
        return np.asarray(outputs, np.float32)


if __name__=='__main__':
    inputs = np.array([[
        [
            [1, 0, 1, 2, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 0, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 0, 1, 2, 0],
        ],
        [
            [2, 0, 2, 1, 1],
            [0, 1, 0, 0, 2],
            [1, 0, 0, 2, 1],
            [1, 1, 2, 1, 0],
            [1, 0, 1, 1, 1],

        ],
    ]])
    print(inputs.shape)
    weights = np.array([[
        [
            [1, 0, 1],
            [-1, 1, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    ]])
    print(weights.shape)
    conv = Conv(inputs, weights)
    out_mat = conv.conv2d_mat()
    print(out_mat)
    print(out_mat.shape)
    out = conv.conv2d()
    print(out)
    print(out.shape)