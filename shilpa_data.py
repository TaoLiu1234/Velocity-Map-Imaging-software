import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import abel
# 读取 .mat 文件
mat = sio.loadmat('shilpa_data.mat')

# 查看所有变量名
print("变量名:", mat.keys())

# 访问某个变量，例如 'data'
data = mat['data']

# 如果是结构体或嵌套数据，可能需要进一步处理
# 例如：mat['struct_var'][0,0]['field_name'][0,0]

print("数据类型:", type(data))
print("数据形状:", data.shape)
print("前5个元素:", data[:5])