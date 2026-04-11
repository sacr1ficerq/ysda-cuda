# Quantization

Quantization is the process of reducing the bit width of model parameters to decrease the memory required to store these parameters and reduce the amount of computation without significantly degrading the model's performance quality.

There are many different quantization methods that balance compression ratio, computation speed, and accuracy. The result of quantization is a quantized model - a set of quantized weights and scales necessary for dequantization.

In this task, you will implement one of the subtasks required for model quantization - the quantization of a linear layer's weight matrix.

You need to implement a method for quantizing a weight matrix `W` with additional weight balancing using an array `S` from the `float` data type to the `int8_t` data type according to the following formula:

$$
W'_{r'c'} = \operatorname{RoundNearest}(\frac{(W_{r'c'} + S_{c'}) * 127} {\max (mScale, \max_{r=r'}\|W_{rc'} + S_{c'}\|)})
$$

where `W'` is the quantized `int8_t` matrix, `W` is the original `float` matrix, `S` is an additional array of `float` coefficients, and `mScale` is the minimum possible scale (in this task it is equal to `1e-5`).

Example:
```
           W                    S                        W'                    Scales
      1.0 0.1 -0.08          0.2 0 1      =>             127 11 97                 105.83333
     -3.3 -0.3  3                                        -98 -10 127               31.75
```

## Required Functions and Data Format
You must implement the following functions with the following signatures:

```cpp
void Quantization(size_t rows, size_t cols, const float* d_inputMatrix, const float* d_balanceFactors, size_t inputStride, size_t outStride, int8_t* d_out, float* d_outScales);
```

where

`rows, cols` - the size of the input and output matrices. You can assume that the matrix is represented in **RowMajor** format. In the example above, rows = 2, cols = 3.

`inputStride, outputStride` - the distance **(in number of elements)** between rows. For the example above, `inputStride == outputStride == cols == 3`, however, in general, due to memory alignment, this number may be greater than `cols`.

`d_outScales` - an array of length `rows`.

`d_balanceFactors` - an array of length `cols`.

Inside the `Quantization` function, all **allocations and input copying have already been performed**. All the `Quantization` function should do is calculate the necessary launch parameters (at the student's discretion) to call `QuantizationKernel` and invoke the latter.

It is guaranteed that `cols % 4 == 0`
