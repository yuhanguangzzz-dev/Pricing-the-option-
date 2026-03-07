import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

#模拟主程序逻辑
from data_processor import DataProcessor
from parametric_models import AHBSModel
from nn_models import PyramidNN
from two_step import MachineCorrectionEngine

# 1. 加载并过滤数据
raw_data = pd.read_csv("sp500_options.csv")
clean_data = DataProcessor.apply_filters(raw_data)

# 2. 截面循环 (按天处理) [cite: 301]
for date, daily_df in clean_data.groupby('date'):
    # A. 划分训练集/测试集 (同日插值法) [cite: 308]
    train_df, test_df = DataProcessor.split_same_day(daily_df)

    # B. 初始化模型 (以 AHBS + NN3 为例) [cite: 351]
    engine = MachineCorrectionEngine(AHBSModel(), PyramidNN("NN3"))

    # C. 两步法训练 [cite: 196]
    engine.train_step(train_df)

    # D. 评估测试集性能 [cite: 320]
    predictions = engine.predict(test_df)
    rmse = np.sqrt(mean_squared_error(test_df['iv'], predictions))
    print(f"Date: {date}, IVRMSE: {rmse:.4%}")