# -*- encoding=utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise


# gets some time slice
def slice_t(train_df, time_sum, time_slice, m, n, h):

    train_np = train_df.reshape((time_sum, h, m, n))
    slice = train_np[time_sum-time_slice:]
    train = slice.reshape((time_slice*h*m*n))
    return train


# gets some height slice
def slice_h(arr, time, m, n, h, asd):
    """
    :param arr:
    :param time: 时序一共 15
    :param m: 行数
    :param n: 列数
    :param h: 截取层数
    :param asd:
    :return:
    """
    train = np.zeros((time, h, m*n))

    # 高度向上移动
    for i in range(time):
        for j in range(h):
            train[i, j] = arr[i, j + asd]

    train = train.reshape((time, h, m, n))
    return train


# cleans trainning data,return data index
def pre_train(train_df, test_df, train_add, test_add):
    """
    :param train_df: 15*4*10*4
    :param test_df:
    :param train_add: 15*1*6*6
    :param test_add:
    :return:
    """
    train = train_df.values[:, 1:-1]  # 特征提取训练数据 10000×（15×4×10×4）
    # print np.array(train).shape (10000, 2400)
    t = train_add.values[:, 1:-1]  # 卷积获取的数据 10000×（12×1×6×6）
    # print np.array(t).shape (10000, 432)
    train = np.hstack((train, t))  # 横向拼接 10000×（15×4×10×4 + 15×4×10×10
    # print np.array(train).shape (10000, 2832)
    dtest = test_df.values[:, 1:]
    tA = test_add.values[:, 1:]
    dtest = np.hstack((dtest, tA))
    # print np.array(dtest).shape (2000, 2832)
    # 计算样本之间的距离
    cor_distance = pairwise.pairwise_distances(dtest, train)
    # print np.array(cor_distance).shape  (2000, 10000)
    resultset = set()
    # 按行遍历， 测试样本到训练集的距离
    for tmp in cor_distance:
        index = np.argsort(tmp)
        for i in range(10):
            resultset.add(index[i])

    index = []
    for i in resultset:
        index.append(i)
    # print len(index) 4730

    return index

