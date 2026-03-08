import pandas as pd
import numpy as np


class DataProcessor:
    class DataProcessor:

        @staticmethod
        def estimate_dividend_yield(df):
            """对于每天和每个到期日，使用最接近平值 (ATM) 的
            看涨和看跌期权对，从 Put-Call Parity (看跌-看涨期权平价关系) 中推导股息率 (q)。
            公式: C - P = S * e^{-q * tau} - K * e^{-r * tau}
            """
            df_q = df.copy()

            # 为了计算方便，预先计算年化时间
            if 'tau' not in df_q.columns:
                df_q['tau'] = df_q['days_to_expiry'] / 365.0

            df_q['q'] = 0.0  # 初始化隐含股息率列

            # 按照交易日和到期时间分组
            for (date, days), group in df_q.groupby(['date', 'days_to_expiry']):
                calls = group[group['option_type'] == 'C']
                puts = group[group['option_type'] == 'P']

                # 找到当天该到期日下，同时有 Call 和 Put 报价的交集行权价
                common_strikes = set(calls['K']).intersection(set(puts['K']))

                if not common_strikes:
                    continue  # 如果没有配对报价，则保持为 0.0

                S = group['S'].iloc[0]
                r = group['r'].iloc[0]
                tau = group['tau'].iloc[0]

                # 寻找最接近平值 (ATM，即 S/K 最接近 1.0) 的行权价 K
                best_K = min(common_strikes, key=lambda k: abs(S / k - 1.0))

                C = calls[calls['K'] == best_K]['price'].values[0]
                P = puts[puts['K'] == best_K]['price'].values[0]

                # 根据平价公式反推隐含股息率 q
                # e^{-q * tau} = (C - P + K * e^{-r * tau}) / S
                val = (C - P + best_K * np.exp(-r * tau)) / S
                if val > 0:
                    q = -np.log(val) / tau
                else:
                    q = 0.0

                # 将计算出的 q 赋值给该分组下（同日、同到期日）的所有期权
                df_q.loc[group.index, 'q'] = q

            return df_q
    @staticmethod
    def apply_filters(df):
        """
        [完善功能 1]：标准化过滤与保留虚值(OTM)逻辑
        执行论文 Section 4 的标准化过滤
        """
        df = df.copy()
        df['m'] = df['S'] / df['K']  # 计算值度 (Moneyness)
        df['tau'] = df['days_to_expiry'] / 365.0  # 到期时间年化

        # 1. 基础过滤标准: 到期时间 20-240 天, 值度 0.8-1.6
        mask_basic = (df['days_to_expiry'] >= 20) & (df['days_to_expiry'] <= 240) & \
                     (df['m'] >= 0.8) & (df['m'] <= 1.6)
        df = df[mask_basic]

        # 2. 剔除实值，仅保留虚值 (OTM) 期权
        mask_otm = ((df['option_type'] == 'C') & (df['m'] <= 1.0)) | \
                   ((df['option_type'] == 'P') & (df['m'] > 1.0))

        return df[mask_otm].reset_index(drop=True)

    def categorize_options(df):
        """
        期权类别的划分和评估标签
        用于生成论文 Table 3 格式的分组跑分结果
        """
        df = df.copy()

        # 划分 Moneyness (值度) 类别
        conditions_m = [
            (df['m'] >= 0.80) & (df['m'] < 0.90),
            (df['m'] >= 0.90) & (df['m'] < 0.97),
            (df['m'] >= 0.97) & (df['m'] < 1.03),
            (df['m'] >= 1.03) & (df['m'] < 1.10),
            (df['m'] >= 1.10) & (df['m'] <= 1.60)
        ]
        choices_m = ['DOTMC', 'OTMC', 'ATM', 'OTMP', 'DOTMP']
        df['moneyness_category'] = np.select(conditions_m, choices_m, default='Unknown')

        # 划分 Maturity (到期时间) 类别
        conditions_tau = [
            (df['days_to_expiry'] >= 20) & (df['days_to_expiry'] <= 60),
            (df['days_to_expiry'] > 60) & (df['days_to_expiry'] <= 240)
        ]
        choices_tau = ['Short', 'Long']
        df['maturity_category'] = np.select(conditions_tau, choices_tau, default='Unknown')

        return df

    @staticmethod
    def split_same_day(df):
        """执行 Section 5.1 的同日插值划分：Strike 能被 10 整除的为训练集"""
        train_idx = df[df['K'] % 10 == 0].index
        test_idx = df[df['K'] % 10 != 0].index
        return df.loc[train_idx], df.loc[test_idx]
