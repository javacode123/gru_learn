# -*- encoding:utf-8 -*-
import sys
sys.path.append('../dataprocess')
import rfmodel as rf
import pandas as pd
import numpy as np
import FeatureSelect as fs
import data_process8 as dp
import generate_percentile as gp
import xgbmodel as xgbm
import bigrumodel as bigru


def check_code(mode, gru_mode):

    if(mode == 'simple'):
        train_df = pd.read_csv('../data/train_percentile.csv')
        test_df = pd.read_csv('../data/testB_percentile.csv')
        train_add = pd.read_csv('../data/train_old_wind_4240.csv')
        testA_add = pd.read_csv('../data/testB_old_wind_4240.csv')
        train_1ave8extend = pd.read_csv('../data/train_new_wind_1ave_8extend.csv')
        test_1ave = pd.read_csv('../data/testB_new_wind_1ave_8extend.csv')
    else:
        trainfile = '../data/train.txt'
        testBfile = '../data/testB.txt'

        train_add = pd.read_csv('../data/train_old_wind_4240.csv')
        # 处理数据集 10000×15×4×101×101 卷积 10000×15×4×20×20 池化 10000×15×4×10×10
        # 根据风向裁剪 10000×15×4×6×6 抽取高度为 1.5km 的图像 1000×15×1×6×6 抽取 时序后 12 张 10000×12×1*6×6
#        train_add = dp.dataprocess(trainfile, data_type='train', windversion='old')

        testA_add = pd.read_csv('../data/testB_old_wind_4240.csv')
#        testA_add = dp.dataprocess(testBfile, data_type='testB', windversion='old')

        # 生成训练集数据,1ave8extend
        train_1ave8extend = pd.read_csv('../data/train_new_wind_1ave_8extend.csv')
        # 根据风的八个方向，每个样本生成 8 张  10000*12*1*6*6
        # train_1ave8extend = dp.dataprocess(trainfile, data_type='train', windversion='new')

        # 生成测试集B数据,1ave
        test_1ave = pd.read_csv('../data/testB_new_wind_1ave_8extend.csv')
#        test_1ave = dp.dataprocess(testBfile, data_type='testB', windversion='new')

        # 生成训练集数据
        train_df = pd.read_csv('../data/train_percentile.csv')
        # 101*101 -> 25*25 50*50 75*75 100*100
        # 1000*15*4*101*101 -> 10000*15*4*10*4 特征提取 101*101 -> 10*4
#        train_df = gp.data_process(trainfile, data_type='train')

        # 生成测试集B数据
        test_df = pd.read_csv('../data/testB_percentile.csv')
#       test_df = gp.data_process(testBfile, data_type='testB')

    print('#data process has been done')
    # result_xgb = xgbm.xgbmodeltrain(train_1ave8extend, test_1ave)
    # np.save("../data/xgb_result.cvs", result_xgb)
    print('#xgb model has been done')

    # 15*4*10*4: train_df  12*1*6*6: train_add  计算距离测试集得出最近的几个样本
    # index = fs.pre_train(train_df=train_df, test_df=test_df, train_add=train_add, test_add=testA_add)

    # 按照模型训练, 将拟合度的索引按顺序返回
    # valid = rf.rf_model(train_df, test_df, 'train', train_add, testA_add, ne=100)  # 10000
    valid = np.load("../data/valid_result.npy")
    # np.save("../data/valid_result", valid)
    ne = 1100

    # index 表示距离 样本最近的样本
    # result_rf = rf.rf_model(train_df, test_df, 'trai', train_add, testA_add, ne, index=index)
    # np.save("../data/result_rf.cvs", result_rf)

    print('#rf model has been done')
    result_bigru = bigru.BiGRU_train(train_df, test_df, valid, gru_mode).reshape(2000)
    np.save("../data/result_bigru.cvs", result_bigru)
    print('#bigru model has been done')
    # ensemble = (result_xgb + result_rf + result_bigru)/3.0
    # np.savetxt("../data/submit_Team4.csv", ensemble)

#check_code('simple', 'online')

check_code('all','online')
