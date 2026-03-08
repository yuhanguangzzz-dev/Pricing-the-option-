import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from parametric_models import BSModel, AHBSModel, HestonModel, CWModel, BatesModel
from nn_models import PyramidNN
from two_step import MachineCorrectionEngine


def evaluate_ivrmse(y_true, y_pred):
    """计算隐含波动率均方根误差 (IVRMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    print("=" * 50)
    print("Step 1: Data Preparation (Section 4)")
    print("=" * 50)

    # 必需列: date, days_to_expiry, option_type, K, S, r, price, iv
    # 宏观特征列 (针对 Panel): VIX, LTV, LTP, RVOL, EPU, ADS, TMS, CRS
    try:
        raw_data = pd.read_csv("sp500_options_with_macro.csv")
        print(f"Loaded {len(raw_data)} raw option records.")
    except FileNotFoundError:
        print("未找到真实的期权数据文件，将生成极简模拟数据以保证流程跑通...")
        # 模拟数据生成逻辑 (仅作流程测试用)
        dates = pd.date_range('2016-01-04', '2019-06-28', freq='B')
        raw_data = pd.DataFrame({
            'date': np.random.choice(dates, 5000),
            'days_to_expiry': np.random.randint(20, 240, 5000),
            'option_type': np.random.choice(['C', 'P'], 5000),
            'K': np.random.uniform(3000, 5000, 5000),
            'S': 4000,
            'r': 0.02,
            'price': np.random.uniform(1, 100, 5000),
            'iv': np.random.uniform(0.05, 0.40, 5000),
            'VIX': np.random.uniform(10, 30, 5000),
            'LTV': np.random.uniform(0, 5, 5000),
            'LTP': np.random.uniform(0, 1, 5000),
            'RVOL': np.random.uniform(10, 30, 5000),
            'EPU': np.random.uniform(50, 150, 5000),
            'ADS': np.random.uniform(-2, 2, 5000),
            'TMS': np.random.uniform(0, 2, 5000),
            'CRS': np.random.uniform(0, 2, 5000)
        })

    # 2. 数据清洗流水线
    print("Estimating dividend yields from Put-Call Parity...")
    data_with_q = DataProcessor.estimate_dividend_yield(raw_data)

    print("Applying standard filters and dropping ITM options...")
    clean_data = DataProcessor.apply_filters(data_with_q)

    print("Categorizing options (Moneyness & Maturity)...")
    final_data = DataProcessor.categorize_options(clean_data)
    print(f"Data preparation complete. Retained {len(final_data)} OTM options.\n")

    print("=" * 50)
    print("Step 2: Cross-Sectional Prediction (Section 5)")
    print("=" * 50)

    # 为了演示，我们只取第一天的数据进行横截面测试
    unique_dates = final_data['date'].unique()
    test_date = unique_dates[0]
    daily_df = final_data[final_data['date'] == test_date]
    print(f"Running Cross-Sectional tests for date: {test_date.date() if hasattr(test_date, 'date') else test_date}")

    # 同日插值划分 (Strikes divisible by 10)
    train_df, test_df = DataProcessor.split_same_day(daily_df)

    if len(train_df) > 0 and len(test_df) > 0:
        # 测评模型列表：BS, AHBS, Heston, CW 以及 Bates
        models_to_test = {
            "AHBS": AHBSModel(),
            "Heston": HestonModel(),
            "Bates (Novelty)": BatesModel()
        }

        for model_name, p_model in models_to_test.items():
            print(f"\n--- Training {model_name} + NN3 ---")
            # 初始化基础神经网络 NN3 (无额外特征)
            nn3_model = PyramidNN(arch_type="NN3", num_features=0)
            engine = MachineCorrectionEngine(p_model, nn3_model, feature_cols=None)

            # 训练两步法
            engine.train_step(train_df)

            # 预测与评估
            predictions = engine.predict(test_df)
            rmse = evaluate_ivrmse(test_df['iv'], predictions)
            print(f"[{model_name} + NN3] Same-day Out-of-Sample IVRMSE: {rmse:.4%}")
    else:
        print("Not enough data to split on this day for the demo.")

    print("\n" + "=" * 50)
    print("Step 3: Panel Data Estimation (Section 6)")
    print("=" * 50)

    # 面板数据按时间划分：假设前 70% 的日期作为训练集 (In-sample)，后 30% 作为测试集 (Out-of-sample)
    sorted_dates = sorted(unique_dates)
    split_idx = int(len(sorted_dates) * 0.7)
    train_dates = sorted_dates[:split_idx]

    panel_train_df = final_data[final_data['date'].isin(train_dates)]
    panel_test_df = final_data[~final_data['date'].isin(train_dates)]

    print(f"Panel Training Set: {len(panel_train_df)} options.")
    print(f"Panel Testing Set: {len(panel_test_df)} options.")

    if len(panel_train_df) > 0 and len(panel_test_df) > 0:
        macro_features = ['VIX', 'LTV', 'LTP', 'RVOL', 'EPU', 'ADS', 'TMS', 'CRS']

        print("\n--- Training Pooled Heston + NN3F (with Macro Features) ---")
        # 1. 结构化参数模型的全局估计 (Pooled Estimation)
        # 注意：由于非线性最小二乘法在极大样本上极度耗时，工业界回测中往往会在面板中
        # 随机抽取一个具有代表性的子集（比如每周三的数据）来估计固定结构参数，
        # 然后固定这些参数去算全样本的残差。这里演示直接丢入引擎。

        p_model_panel = HestonModel()
        nn3f_model = PyramidNN(arch_type="NN3", num_features=len(macro_features))

        engine_panel = MachineCorrectionEngine(p_model_panel, nn3f_model, feature_cols=macro_features)

        # 训练面板模型
        print("Fitting parametric model and neural network on Panel Data... (This may take a while)")
        engine_panel.train_step(panel_train_df)

        # 面板样本外预测
        print("Predicting on out-of-sample Panel Data...")
        panel_predictions = engine_panel.predict(panel_test_df)
        panel_rmse = evaluate_ivrmse(panel_test_df['iv'], panel_predictions)

        print(f"[Heston + NN3F] Out-of-Sample Panel IVRMSE: {panel_rmse:.4%}")

        # 你可以进一步按照 category 输出表现，对标 Table 4
        panel_test_df['pred_iv'] = panel_predictions
        print("\nPanel Performance by Moneyness:")
        for category, group in panel_test_df.groupby('moneyness_category'):
            cat_rmse = evaluate_ivrmse(group['iv'], group['pred_iv'])
            print(f"  {category}: {cat_rmse:.4%}")


if __name__ == "__main__":
    main()