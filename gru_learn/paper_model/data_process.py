# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# 提取特征（10*10 20*20 ... 90*90 100*100 对应的的百分位数25% 50% 75% 1005）
# 101*101 -> 10*4
def percentile(line):

    cate = line[0].split(',')
    # 标签名称 train_i
    id_label = [cate[0]]
    # 降水量
    id_label.append(float(cate[1]))
    record = [int(cate[2])]
    length = len(line)

    for i in range(1, length):
        record.append(int(line[i]))

    #  一条数据
    mat = np.array(record).reshape(15, 4, 101, 101)
    # deals with -1， 将 -1 变为 0
    mat[mat == -1] = 0

    con_mat = np.zeros((15, 4, 10, 4))
    for i in range(15):
        for j in range(4):
            # temp_mat: 101*101
            temp_mat = mat[i, j]
            for m in range(1, 11):
                # 从中间区域开始范围逐渐扩大 10*10 20*20 30*30 ...... 90*90 100*100
                mt = temp_mat[50-5*m:50+5*m+1, 50-5*m:50+5*m+1]
                # 百分位数
                con_mat[i, j, m-1, 0] = np.max(mt)
                con_mat[i, j, m-1, 1] = np.percentile(mt, 75, interpolation='lower')
                con_mat[i, j, m-1, 2] = np.percentile(mt, 50, interpolation='lower')
                con_mat[i, j, m-1, 3] = np.percentile(mt, 25, interpolation='lower')

    return id_label, con_mat.reshape(15*4*10*4)


def process_data(filename):
    head_list = ['id']
    for i in range(15*4*10*4):  # 101*101 提取特征 10*4
        feature = 'thxy' + str(i)
        head_list.append(feature)

    head_list.append('label')
    df = pd.DataFrame(columns=head_list)

    with open(filename) as f:
        num = 10000  # 共一万组数据
        for i in range(1, num+1):
            print "process" + str(i)
            line = f.readline().strip().split(' ')
            # 15*4*101*101 -> 15*4*10*4 将雷达数据转换为 10*4 的特征
            id_label, con_mat = percentile(line)
            simple = list(con_mat)
            # 标签 id
            temp = [id_label[0]]
            # id 数据 10*4
            temp += simple
            # id 数据 降水量
            temp += [id_label[1]]
            # print id_label
            # print len(con_mat)
            # print temp
            df_temp = pd.DataFrame([temp], columns=head_list)
            df = df.append(df_temp, ignore_index=True)
            # print df.head()

        df.to_csv('../paper_data/train_percentile.csv', index=False, float_format='%.3f')
    return df


if __name__ == "__main__":
    process_data('../data/train.txt')
