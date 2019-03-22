# -×- coding:utf-8 -*-
import numpy as np
import pandas as pd
import dataprocess.FeatureSelect as fs


# get the wind direct, old version
def windDriectold(pooling_mat):
    """
    根据 4.5km 高度处的图像求解风向（图像最大值）, 根据风向裁剪 6×6
    :param pooling_mat: (15*4*10*10)
    :return: （15*4*6*6）
    """
    direct_mat = np.zeros((15, 4, 6, 6))
    xx = np.arange(15)
    yy = np.arange(15)
    # 分别存储四个方向
    windDrirct = np.zeros(4)  # [0. 0. 0. 0.]
    sumx = 0
    sumy = 0

    for i in range(15):
        # 获取高度为 3.5km 图像
        wind_mat = pooling_mat[i, 3]  # forth level
        raw, column = wind_mat.shape  # 10×10
        # 展成一维 返回最大数值的索引
        positon = np.argmax(wind_mat)
        # 最大值的坐标
        m, n = divmod(positon, column)  # position of the max value point
        xx[i] = m
        yy[i] = n

    for i in range(0, 3):  # 0,1,2
        sumx += xx[i]
        sumy += yy[i]
    # 前三个时刻横坐标平均值
    sumx = sumx/3
    # 前三个时刻纵坐标平均值
    sumy = sumy/3

    for i in range(3, 15):
        # 向右移动
        if xx[i] > sumx:  # north wind
            windDrirct[2] += 1
        if xx[i] < sumx:
            windDrirct[3] += 1
        # 向上移动
        if yy[i] > sumy:  # west wind
            windDrirct[0] += 1
        if yy[i] < sumy:
            windDrirct[1] += 1
    # 获取移动的最大距离
    direct = np.argmax(windDrirct)

    for i in range(15):
        for j in range(4):
            temp_mat = pooling_mat[i, j]  # 获取池化图像（10×10）
            # 最大移动范围 12
            if direct == 0:  # 上移
                if windDrirct[2] > 6:  # west north 右移动
                    direct_mat1 = temp_mat[1:7, 0:6]
                elif windDrirct[3] > 6:  # west south 左边移动
                    direct_mat1 = temp_mat[3:9, 0:6]
                else:
                    direct_mat1 = temp_mat[2:8, 0:6]

            elif direct == 1:  # 向下移动
                if windDrirct[2] > 6:  # east
                    direct_mat1 = temp_mat[1:7, 4:10]
                elif windDrirct[3] > 6:  # east south
                    direct_mat1 = temp_mat[3:9, 4:10]
                else:
                    direct_mat1 = temp_mat[2:8, 4:10]

            elif direct == 2:
                if windDrirct[0] > 6:  # west north
                    direct_mat1 = temp_mat[0:6, 1:7]
                elif windDrirct[1] > 6:  # east north
                    direct_mat1 = temp_mat[0:6, 3:9]
                else:
                    direct_mat1 = temp_mat[0:6, 2:8]

            elif direct == 3:
                if windDrirct[0] > 6:  # west south
                    direct_mat1 = temp_mat[4:10, 1:7]
                elif windDrirct[1] > 6:  # east south
                    direct_mat1 = temp_mat[4:10, 3:9]
                else:
                    direct_mat1 = temp_mat[4:10, 2:8]
            else:
                direct_mat1 = temp_mat[2:8, 2:8]

            direct_mat[i][j] = direct_mat1

    return direct_mat.reshape(2160)


# get wind direct, new version
def windDriect1ave(pooling_mat):
    """

    :param pooling_mat: 15*4*10*10
    :return:
    """
    direct_mat = np.zeros((15, 4, 6, 6))
    xx = np.arange(15)
    yy = np.arange(15)
    windDrirct = np.zeros(4)

    for i in range(15):
        wind_mat = pooling_mat[i, 3]  # 高度为 3.5km
        raw, column = wind_mat.shape  # 10*10
        sumx = 0
        sumy = 0
        temp = wind_mat.reshape(1, 100)
        paramsort = np.argsort(-temp)  # 返回从大到小排序后的索引

        for j in range(5):  # 取 top5 对应的下标
            sumx += paramsort[0][j] // column  # 最大值对应行
            sumy += paramsort[0][j] % column  # 最大值对应列

        xx[i] = sumx // 5
        yy[i] = sumy // 5

    for i in range(1, 15):
        if xx[i] > xx[0]:
            windDrirct[2] += 1
        if xx[i] < xx[0]:
            windDrirct[3] += 1
        if yy[i] > yy[0]:
            windDrirct[0] += 1
        if yy[i] < yy[0]:
            windDrirct[1] += 1

    direct = np.argmax(windDrirct)

    for i in range(15):
        for j in range(4):

            temp_mat = pooling_mat[i, j]
            if direct == 0:
                if windDrirct[2] > 7:
                    direct_mat1 = temp_mat[1:7, 0:6]
                elif windDrirct[3] > 7:
                    direct_mat1 = temp_mat[3:9, 0:6]
                else:
                    direct_mat1 = temp_mat[2:8, 0:6]

            elif direct == 1 :
                if windDrirct[2] > 7:
                    direct_mat1 = temp_mat[1:7, 4:10]
                elif windDrirct[3] > 7:
                    direct_mat1 = temp_mat[3:9, 4:10]
                else:
                    direct_mat1 = temp_mat[2:8, 4:10]

            elif direct == 2:
                if windDrirct[0] > 7:
                    direct_mat1 = temp_mat[0:6, 1:7]
                elif windDrirct[1] > 7:
                    direct_mat1 = temp_mat[0:6, 3:9]
                else:
                    direct_mat1 = temp_mat[0:6, 2:8]

            elif direct == 3:
                if windDrirct[0] > 7:
                    direct_mat1 = temp_mat[4:10, 1:7]
                elif windDrirct[1] > 7:
                    direct_mat1 = temp_mat[4:10, 3:9]
                else:
                    direct_mat1 = temp_mat[4:10, 2:8]

            else:
                direct_mat1 = temp_mat[2:8, 2:8]

            direct_mat[i][j] = direct_mat1

    return direct_mat


#extend data in eight directions
def extendData(pooling_mat):
    """
    :param pooling_mat: 15*4*6*6
    :return:
    """
    return_value = np.zeros((8, 15, 4, 6, 6))

    for i in range(15):

        for j in range(4):

            temp_mat = pooling_mat[i, j]

            #topdown
            temp_mat1 = np.flipud(temp_mat)
            #leftright
            temp_mat2 = np.fliplr(temp_mat)
            #topdown-leftright
            temp_mat3 = np.fliplr(temp_mat1)
            #turn 90
            temp_mat4 = np.rot90(temp_mat)
            #turn 270
            temp_mat5 = np.rot90(temp_mat, 3)
            #turn 90-topdown
            temp_mat6 = np.flipud(temp_mat4)
            #tun 90-leftright
            temp_mat7 = np.fliplr(temp_mat4)

            return_value[0][i][j] = temp_mat
            return_value[1][i][j] = temp_mat1
            return_value[2][i][j] = temp_mat2
            return_value[3][i][j] = temp_mat3
            return_value[4][i][j] = temp_mat4
            return_value[5][i][j] = temp_mat5
            return_value[6][i][j] = temp_mat6
            return_value[7][i][j] = temp_mat7

    return return_value


# convolution 5*5
def train_convolution(line, data_type):
    """
    :param line: 数据集的每一行     train_i, y, data
    :param data_type: train or test
    :return: id_lable(train_i, y)  data(15*4*20*20)
    """
    cate = line[0].split(',')
    id_label = [cate[0]]

    if data_type == 'train':
        id_label.append(float(cate[1]))  # 获取降水值

    record = [int(cate[2])]  # 雷达图像素
    length = len(line)

    for i in range(1, length):
        record.append(int(line[i]))
    # 数据格式 101×101 分张， 101×101×4 四张属于同一个高度， 15×4×10×101 15个属于一个样本
    mat = np.array(record).reshape(15, 4, 101, 101)  # 转互成 [15×[4×[101×101]]]矩阵
    con_mat = np.zeros((15, 4, 20, 20))

    # 使用卷积核 5×5 将 100×100 图像变换成 20×20
    for i in range(15):
        for j in range(4):
            temp_mat = mat[i, j]  # 获取时序为 i 高度为 j 的图像  101*101
            temp_mat = np.delete(temp_mat, 0, axis=0)  # 删除第一行
            temp_mat = np.delete(temp_mat, 0, axis=1)  # 删除第一列 100*100
            for m in range(20):
                for n in range(20):
                    # 用 5*5 的矩阵逐行扫描
                    avg_mat = temp_mat[m*5:m*5+5, n*5:n*5+5]  # 5×5
                    # 卷积求平均值
                    con_mat[i, j, m, n] = np.average(avg_mat)

    return id_label, con_mat


# max pooling
def max_pooling(con_mat):

    pooling_mat = np.zeros((15, 4, 10, 10))  # 10,10

    for i in range(15):
        for j in range(4):
            temp_mat = con_mat[i, j]
            for m in range(10):  # 10
                for n in range(10):  # 10
                    max_mat = temp_mat[2*m:2*m+2, n*2:n*2+2]
                    pooling_mat[i, j, m, n] = np.max(max_mat)

    return pooling_mat


# process wind data
def dataprocess(filename, data_type, windversion):

    header_list = ['id']

    for i in range(432):  # 6000
        feature = 'thxy_' + str(i+1)
        header_list.append(feature)

    if data_type == 'train':
        header_list += ['label']

    # header_list : id thxy_1 thxy_2  hxy_3 ... thxy_431 thxy_432 label
    df = pd.DataFrame(columns=header_list)

    with open(filename) as fr:
        if data_type == 'train':
            sample_num = 10000
        elif data_type == 'testB':
            sample_num = 2000

        for i in range(1, sample_num + 1):

            line = fr.readline().strip().split(' ')
            # 得到标签和降水数值 数据卷积处理（15*4*101*101)->(15×4×20×20）
            id_label, con_mat = train_convolution(line, data_type)
            # 数据进行池化 2×2 （15×4×20×20)->(15*4*10*10)
            pooling_mat = max_pooling(con_mat)

            if windversion == 'new' and data_type == 'train':
                # 8*15*3*6*6
                eightValue = extendData(windDriect1ave(pooling_mat))
                for j in range(8):
                    value = eightValue[j].reshape((15, 4, 36))
                    # value (15, 1, 36) 截取 1.5km
                    value = fs.slice_h(value, time=15, m=6, n=6, h=1, asd=1)
                    # value(1, 12*1*6*6) 截取后 12 张
                    value = fs.slice_t(value, time_sum=15, time_slice=12, m=6, n=6, h=1)
                    # 雷达数据 10000×12×1×6×6
                    simp = list(value)
                    # 标签
                    temp = [id_label[0]]
                    temp += simp

                    if data_type == 'train':
                        temp += [id_label[1]]

                    # print(temp)
                    df_temp = pd.DataFrame([temp], columns=header_list)
                    df = df.append(df_temp, ignore_index=True)

            else:

                if windversion == 'old':
                    # 根据风向进行裁剪
                    value = windDriectold(pooling_mat)
                else:  # new test
                    value = windDriect1ave(pooling_mat).reshape(2160)

                value = value.reshape((15, 4, 36))  # value 15*4*6*6
                value = fs.slice_h(value, time=15, m=6, n=6, h=1, asd=1)  # value (15, 1, 36) 截取 1.5km 的高度的图像
                value = fs.slice_t(value, time_sum=15, time_slice=12, m=6, n=6, h=1)  # value(1, 12*1*6*6) 截取后 12 张
                simp = list(value)  # 12*6*6 = 432
                temp = [id_label[0]]  # train_i
                temp += simp  # train_i   value

                if data_type == 'train':
                    temp += [id_label[1]]  # train_i   value   label

                #print(temp)
                df_temp = pd.DataFrame([temp], columns=header_list)
                df = df.append(df_temp, ignore_index=True)

        print(df.head())

        if windversion == 'old':
            df.to_csv('../data/'+data_type+'_'+windversion+'_wind_4240.csv', index=False, float_format='%.3f')
        else:
            df.to_csv('../data/'+data_type+'_'+windversion+'_wind_1ave_8extend.csv', index=False, float_format='%.3f')

    return df


#the path of train set & the path of testB set
#trainfile = '../data/train.txt'
#testBfile = '../data/testB.txt'

#produces the train set of 'old' wind
#dataprocess(trainfile, data_type='train',windversion='old')
#proceces the testB set of 'old' wind
#dataprocess(testBfile, data_type='testB',windversion='old')

#produces the extended train set
#dataprocess(trainfile, data_type='train',windversion='new')
#produces the extended testB set
#dataprocess(testBfile, data_type='testB',windversion='new')
