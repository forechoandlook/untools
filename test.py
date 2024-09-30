#     # def test_Conv_backward(self):
#     #     class Model(torch.nn.Module):
#     #         def __init__(self):
#     #             super(Model, self).__init__()

#     #         def forward(self, x,y,z):
#     #             out0,out1,out2 = torch.ops.aten.convolution_backward(x, y, z, [0], [2,2], 
#     #                   [1, 1], [1, 1], False, [0,0], 1, [True, True, False])
#     #             out2 = None
#     #             return [out0,out1,out2]

#     #     self.trace_and_test([[8,64,9,9],[8,3,16,16],[64,3,2,2]], Model())

# import ctypes
# import numpy as np

# # 加载动态库
# lib = ctypes.cdll.LoadLibrary('./build/libtest_project.dylib')

# float_ptr = ctypes.POINTER(ctypes.c_float)


# # 定义 C 函数的参数类型
# lib.conv_bwd.argtypes = [float_ptr, 
#                          float_ptr,
#                          float_ptr,
#                          float_ptr,  # grad_input
#                          float_ptr,  # grad_weight
#                          float_ptr,  # grad_bias
#                          ctypes.c_int, ctypes.c_int, ctypes.c_int, 
#                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
#                          ctypes.c_int, ctypes.c_int]

# # 创建输入数据
# batch_size = 1
# in_channels = 3
# out_channels = 8
# height = 32
# width = 32
# kernel_h = 3
# kernel_w = 3
# out_h = 30
# out_w = 30

# # 初始化数据 (随机初始化)
# src_data = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
# weights_data = np.random.rand(out_channels, in_channels, kernel_h, kernel_w).astype(np.float32)
# dst_data = np.random.rand(batch_size, out_channels, out_h, out_w).astype(np.float32)

# # 初始化梯度数组
# grad_input = np.zeros_like(src_data).astype(np.float32)
# grad_weight = np.zeros_like(weights_data).astype(np.float32)
# grad_bias = np.zeros(out_channels, dtype=np.float32)

# # 调用C++的卷积反向传播
# lib.conv_bwd(src_data.ctypes.data_as(float_ptr),
#              weights_data.ctypes.data_as(float_ptr),
#              dst_data.ctypes.data_as(float_ptr),
#              grad_input.ctypes.data_as(float_ptr),
#              grad_weight.ctypes.data_as(float_ptr),
#              grad_bias.ctypes.data_as(float_ptr),
#              batch_size, in_channels, out_channels, height, width, kernel_h, kernel_w, out_h, out_w)


# # 定义 strides 和 paddings
# strides = (1, 1)
# padding = (0, 0)
# import torch
# # 调用 torch.ops.aten.convolution_backward 来计算梯度
# # 返回值为 (grad_input, grad_weight, grad_bias)
# grad_output_tensor = torch.from_numpy(src_data)
# src_tensor = torch.from_numpy(dst_data)
# weights_tensor = torch.from_numpy(weights_data)
# print(grad_output_tensor.shape, weights_tensor.shape, src_tensor.shape)
# grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
#     src_tensor,  # input
#     grad_output_tensor,  # grad_output
#     weights_tensor,  # weight
#     None,  # bias (None means no bias)
#     strides,  # stride
#     padding,  # padding
#     (1, 1),  # dilation
#     False,  # transposed
#     (0, 0),  # output_padding (only relevant for transposed convolutions)
#     1,  # groups
#     [True, True, False]
# )
# print(grad_input)
# print(grad_weight)
# print(grad_bias)
# # 输出计算的梯度
# print("Grad Input:\n", grad_input)
# print("Grad Weight:\n", grad_weight)
# print("Grad Bias:\n", grad_bias)


import ctypes
import numpy as np

# 加载动态链接库
lib = ctypes.CDLL('./build/libtest_project.dylib')

# 定义输入张量
src_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
weights_data = np.random.randn(8, 3, 3, 3).astype(np.float32)
dst_data = np.random.randn(1, 8, 30, 30).astype(np.float32)

# 定义输出梯度
grad_input = np.zeros_like(src_data)
grad_weight = np.zeros_like(weights_data)
grad_bias = np.zeros((8,), dtype=np.float32)

lib.conv_bwd(
    src_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    weights_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    dst_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    grad_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    grad_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    grad_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    1, 3, 8, 32, 32, 3, 3, 30, 30, 1, 1, 0, 0
)

# 输出结果
print("Grad Input:", grad_input)
print("Grad Weight:", grad_weight)
print("Grad Bias:", grad_bias)