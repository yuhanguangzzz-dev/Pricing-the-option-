import torch
import torch.nn as nn


class MachineCorrectionEngine:
    def __init__(self, parametric_model, nn_model, feature_cols=None):
        """
        :param parametric_model: 基础参数模型 (如 BSModel, HestonModel)
        :param nn_model: 修正网络 (如 PyramidNN("NN3") 或 PyramidNN("NN3", num_features=8))
        :param feature_cols: 宏观特征列名列表 (如果为 None，则退化为横截面模式)
        """
        self.p_model = parametric_model
        self.nn_corrector = nn_model

        # 为了严格复现论文基准，其实论文使用的是 Moller(1993) 的 SCG (缩放共轭梯度算法)
        # 但 PyTorch 没有内置 SCG，我们可以保留 Adam，或者换成 LBFGS 来近似二阶优化器的效果。这里暂保留 Adam。
        self.optimizer = torch.optim.Adam(self.nn_corrector.parameters(), lr=0.01)
        self.feature_cols = feature_cols if feature_cols is not None else []
        self.input_cols = ['m', 'tau'] + self.feature_cols

    def train_step(self, df_train):
        # 1. 提取基础变量
        m = df_train['m'].values
        tau = df_train['tau'].values
        iv = df_train['iv'].values

        # 注意：如果你之前引入了 Heston 或 Bates 模型，这里需要传入 S 和 r
        # 假设我们修改 parametric_model 的 fit 接口兼容 **kwargs
        if hasattr(self.p_model, 'params'):  # 粗略判断是否是复杂结构化模型
            S, r = df_train['S'].values, df_train['r'].values
            self.p_model.fit(m, tau, iv, S=S, r=r)
            iv_fitted_p = self.p_model.predict(m, tau, S=S, r=r)
        else:
            self.p_model.fit(m, tau, iv)
            iv_fitted_p = self.p_model.predict(m, tau)

        # 模型定价误差 epsilon_p
        epsilon_p = iv - iv_fitted_p

        # 2. 训练神经网络拟合误差表面 f(y_t, m, tau)
        # X 现在不仅包含 m, tau, 还包含了可能的宏观特征
        X = torch.FloatTensor(df_train[self.input_cols].values)
        y = torch.FloatTensor(epsilon_p).view(-1, 1)

        self.nn_corrector.train()
        for epoch in range(500):
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.nn_corrector(X), y)
            loss.backward()
            self.optimizer.step()

    def predict(self, df_test):
        m = df_test['m'].values
        tau = df_test['tau'].values

        # 获取参数模型预测
        if hasattr(self.p_model, 'params'):
            S, r = df_test['S'].values, df_test['r'].values
            iv_p = self.p_model.predict(m, tau, S=S, r=r)
        else:
            iv_p = self.p_model.predict(m, tau)

        # 获取 NN 修正预测
        X_test = torch.FloatTensor(df_test[self.input_cols].values)
        self.nn_corrector.eval()
        with torch.no_grad():
            f_hat = self.nn_corrector(X_test).numpy().flatten()

        return iv_p + f_hat  # 被机器修正后的最终 IV