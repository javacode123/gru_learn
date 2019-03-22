# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gru_model as gru


if __name__ == "__main__":
    # run gru model
    # 1000*15*4*101*101 预处理 10000*15*4*10*4 百分位特征提取
    train_data = pd.read_csv('../paper_data/train_percentile.cvs')
    result_bigru = gru.BiGRU_train(train_data, test_df, valid, gru_mode).reshape(2000)
    # np.save("../data/result_bigru", result_bigru)
