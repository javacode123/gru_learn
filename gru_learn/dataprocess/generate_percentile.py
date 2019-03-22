# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


# produces percentile data
def percentile(line, data_type):

    cate = line[0].split(',')
    # 标签名称 train_i
    id_label = [cate[0]]

    if data_type == 'train':
        # 降水量
        id_label.append(float(cate[1]))

    record = [int(cate[2])]
    length = len(line)

    for i in range(1, length):
        record.append(int(line[i]))

    mat = np.array(record).reshape(15, 4, 101, 101)

    # deals with -1， 将 -1 变为 0
    mat[mat == -1] = 0

    con_mat = np.zeros((15, 4, 10, 4))
    for i in range(15):
        for j in range(4):
            # 101*101
            temp_mat = mat[i, j]
            for m in range(1, 11):
                # 从中间区域开始范围逐渐扩大 10*10 20*20 30*30 ...... 90*90 100*100
                mt = temp_mat[50-5*m:50+5*m+1, 50-5*m:50+5*m+1]
                # 最大值
                con_mat[i, j, m-1, 0] = np.max(mt)
                con_mat[i, j, m-1, 1] = np.percentile(mt, 75, interpolation='lower')
                con_mat[i, j, m-1, 2] = np.percentile(mt, 50, interpolation='lower')
                con_mat[i, j, m-1, 3] = np.percentile(mt, 25, interpolation='lower')

    return id_label, con_mat.reshape(15*4*10*4)


# produces percentile data set
def data_process(filename, data_type):

    header_list = ['id']
    for i in range(15*4*10*4):
        feature = 'thxy_' + str(i+1)
        header_list.append(feature)

    if data_type == 'train':
        header_list += ['label']

    df = pd.DataFrame(columns=header_list)

    with open(filename) as fr:

        if data_type == 'train':
            sample_num = 10000
        elif data_type == 'testB':
            sample_num = 2000

        for i in range(1, sample_num+1):

            line = fr.readline().strip().split(' ')
            # 15*4*101*101 -> 15*4*10*4 将雷达数据转换为 10*4 的特征
            id_label, con_mat = percentile(line, data_type)
            simp = list(con_mat)
            temp = [id_label[0]]
            temp += simp

            if data_type == 'train':
                temp += [id_label[1]]

            print(temp)
            df_temp = pd.DataFrame([temp], columns=header_list)
            df = df.append(df_temp, ignore_index=True)

#        print(df.head())
        df.to_csv('../data/'+data_type+'_percentile.csv', index=False, float_format='%.3f')
    return df


