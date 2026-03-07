import pandas as pd
import numpy as np


class DataProcessor:
    @staticmethod
    def apply_filters(df):
        """执行论文 Section 4 的标准化过滤 [cite: 258-261]"""
        df = df.copy()
        df['m'] = df['S'] / df['K']  # 计算值度 (Moneyness) [cite: 131]
        df['tau'] = df['days_to_expiry'] / 365.0  # 到期时间年化 [cite: 255]

        # 过滤标准: 到期时间 20-240 天, 值度 0.8-1.6 [cite: 258]
        mask = (df['days_to_expiry'] >= 20) & (df['days_to_expiry'] <= 240) & \
               (df['m'] >= 0.8) & (df['m'] <= 1.6)
        return df[mask].reset_index(drop=True)

    @staticmethod
    def split_same_day(df):
        """执行 Section 5.1 的同日插值划分：Strike 能被 10 整除的为训练集 [cite: 308]"""
        train_idx = df[df['K'] % 10 == 0].index
        test_idx = df[df['K'] % 10 != 0].index
        return df.loc[train_idx], df.loc[test_idx]