import ctypes 
import numpy as np 
import time 
import os
import logging
import torch

MAX_DIMS          = 8
MAX_OP_TENSOR_NUM = 10
MAX_CHAR_NUM      = 256
MAX_SHAPE_NUM     = 8
MAX_TENSOR_NUM    = 64
MAX_CMD_GROUP_NUM = 64
MAX_STAGE_NUM     = 64

lib = ctypes.CDLL("./build/libshare.so")

int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_void_p
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool
null_ptr  = ctypes.c_void_p(None)
ref       = lambda x: ctypes.byref(x)

# # 将整数值转换为 void* 类型
# data_ptr = ctypes.cast(tensor.data, ctypes.c_void_p)

def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)

def make2_c_int_list(my_list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point):
    return ctypes.string_at(char_point).decode('utf-8')

def make_np2c(np_array:np.ndarray):
    if np_array.flags['CONTIGUOUS'] == False:
        # info users
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

def make_torch2c(tensor:torch.Tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    ptr = tensor.data_ptr()
    return ctypes.c_void_p(ptr)

def str2char_point(string):
    return ctypes.c_char_p(string.encode('utf-8'))

data_type = {
    np.float32:0,
    np.float16:1,
    np.int16:4,
    np.int32:6,
    np.dtype(np.float32):0,
    np.dtype(np.float16):1,
    np.dtype(np.int16):4,
    np.dtype(np.int32):6,
    np.int8:2,
    np.dtype(np.int8):2,
    np.uint8:3,
    np.dtype(np.uint8):3,
}

type_map = {
    0: np.float32,
    1: np.float16,
    4: np.int16,
    6: np.int32,
    2: np.int8,
    3: np.uint8,
}

dtype_ctype_map = {
    np.float32: ctypes.c_float,
    np.float16: ctypes.c_uint16,
    np.int8:   ctypes.c_int8,
}

def make_c2np(data_ptr, shape, dtype):
    num = np.prod(shape)
    array_type = ctypes.cast(data_ptr, ctypes.POINTER(dtype_ctype_map[dtype]))
    np_array = np.ctypeslib.as_array(array_type, shape=(num,))
    return np_array.view(dtype=dtype).reshape(shape)

