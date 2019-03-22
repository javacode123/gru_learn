# -*- coding: utf-8 -*-
"""
1: 将数据集 10000×15×4×101×101 俺高度分离 40000×15×101×101  是否需要针对不同高度分开计算
   训练集，测试集，验证集 6：2：2
2: 特征提取, 额外特征
2: 训练模型 xgbmodel 有均方损失计算
3: 模型组合
3: 风向
4: 性能指标  mse  mae
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import math

x = np.load("./data/xgb_result.cvs.npy")
y = np.load("./data/result_bigru.cvs.npy")
z = np.load("./data/result_rf.cvs.npy")
print math.sqrt(mean_squared_error(x, y))
print math.sqrt(mean_squared_error(x, z))
print math.sqrt(mean_squared_error(y, z))

# Train Score: 6.18871 RMSE
# Test Score: 13.80024 RMSE

# ========== 会用到的公式 ==============
# X = X[:, :, 1:2, :, :] 将层数分离 10000×15×4×101×101 -> [:, :, 0, :, :] [:, :, 1, :, :]

