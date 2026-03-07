import torch
import torch.nn as nn

class MachineCorrectionEngine:
    def __init__(self, parametric_model, nn_model):
        self.p_model = parametric_model
        self.nn_corrector = nn_model
        self.optimizer = torch.optim.Adam(self.nn_corrector.parameters(), lr=0.01)

    def train_step(self, df_train):
        # 1. 拟合参数模型并获取残差 epsilon_p [cite: 199]
        m, tau, iv = df_train['m'].values, df_train['tau'].values, df_train['iv'].values
        self.p_model.fit(m, tau, iv)
        iv_fitted_p = self.p_model.predict(m, tau)
        epsilon_p = iv - iv_fitted_p  # 模型定价误差 [cite: 195, 199]

        # 2. 训练神经网络拟合误差表面 f(m, tau) [cite: 200-201]
        X = torch.FloatTensor(df_train[['m', 'tau']].values)
        y = torch.FloatTensor(epsilon_p).view(-1, 1)

        for epoch in range(500):
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.nn_corrector(X), y)
            loss.backward()
            self.optimizer.step()

    def predict(self, df_test):
        # 最终预测 = 参数模型预测 + NN 修正预测 [cite: 204]
        m, tau = df_test['m'].values, df_test['tau'].values
        iv_p = self.p_model.predict(m, tau)

        X_test = torch.FloatTensor(df_test[['m', 'tau']].values)
        with torch.no_grad():
            f_hat = self.nn_corrector(X_test).numpy().flatten()

        return iv_p + f_hat  # 被机器修正后的 IV [cite: 204, 325]