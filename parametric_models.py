import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, least_squares, newton


# ==========================================
# 辅助工具：Black-Scholes 基础计算 [cite: 113, 125]
# ==========================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    """标准 Black-Scholes 期权定价公式 [cite: 113]"""
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_iv(price, S, K, T, r, option_type='call'):
    """使用 Newton-Raphson 方法反推隐含波动率 [cite: 125, 159]"""
    if price <= 0: return 0.0
    try:
        # 初始值设为 0.2
        return newton(lambda x: bs_price(S, K, T, r, x, option_type) - price, 0.2, tol=1e-5)
    except:
        return 0.0


# ==========================================
# 1. Black-Scholes (BS) 模型 [cite: 105, 142]
# ==========================================

class BSModel:
    """
    基础 Black-Scholes 模型。
    在截面估计中，该模型预测一个常数的隐含波动率表面 [cite: 119, 142]。
    """

    def __init__(self):
        self.avg_iv = 0.0

    def fit(self, m, tau, iv):
        # 估计方式：计算当天所有观测期权的平均隐含波动率 
        self.avg_iv = np.mean(iv)
        return self

    def predict(self, m, tau):
        # 预测结果为平面的常数 [cite: 144]
        return np.full_like(m, self.avg_iv)


# ==========================================
# 2. Ad-hoc Black-Scholes (AHBS) 模型 [cite: 126]
# ==========================================

class AHBSModel:
    """
    从业者 Black-Scholes 模型 (Practitioner BS)。
    将 IV 建模为值度 (m) 和时间 (tau) 的二次多项式 。
    """

    def __init__(self):
        self.params = None

    def _get_X(self, m, tau):
        # 矩阵构造: [1, m, m^2, tau, tau^2, m*tau] [cite: 132, 139]
        return np.column_stack([np.ones_like(m), m, m ** 2, tau, tau ** 2, m * tau])

    def fit(self, m, tau, iv):
        X = self._get_X(m, tau)
        # 通过普通最小二乘法 (OLS) 最小化 IVMSE [cite: 135, 136]
        self.params, _, _, _ = np.linalg.lstsq(X, iv, rcond=None)
        return self

    def predict(self, m, tau):
        X = self._get_X(m, tau)
        return X @ self.params


# ==========================================
# 3. Heston 模型 (COS 方法实现) [cite: 145, 157]
# ==========================================

class HestonModel:
    """
    Heston 随机波动率模型。
    使用 Fourier-cosine 系列展开法进行定价 [cite: 157]。
    """

    def __init__(self):
        self.params = [0.04, 0.04, 2.0, 0.3, -0.7]  # [v0, v_bar, kappa, sigma_v, rho] [cite: 155]

    def _char_func(self, u, T, r, v0, v_bar, kappa, sigma_v, rho):
        """Heston 特征函数 [cite: 156]"""
        d = np.sqrt((rho * sigma_v * u * 1j - kappa) ** 2 + sigma_v ** 2 * (u * 1j + u ** 2))
        g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)
        C = (kappa * v_bar / sigma_v ** 2) * (
                    (kappa - rho * sigma_v * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v ** 2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        return np.exp(C + D * v0 + u * 1j * r * T)

    def cos_price(self, S, K, T, r, v0, v_bar, kappa, sigma_v, rho, N=128):
        """基于 COS 方法的定价逻辑 [cite: 157]"""
        x, k = np.log(S / K), np.arange(N)
        a, b = -10, 10
        u = k * np.pi / (b - a)
        cf = self._char_func(u, T, r, v0, v_bar, kappa, sigma_v, rho)
        # 简化计算 Chi 和 Psi 的级数求和
        term = np.exp(1j * k * np.pi * (x - a) / (b - a))
        return S * np.real(np.sum(cf * term)) / N

    def fit(self, m, tau, iv, S, r):
        # 估计结构参数：最小化隐含波动率误差 (IVMSE) [cite: 158, 161]
        def objective(p):
            # p = [v0, v_bar, kappa, sigma_v, rho]
            v0, v_bar, kappa, sigma_v, rho = p
            err = 0
            for i in range(len(m)):
                K = S / m[i]
                price = self.cos_price(S, K, tau[i], r, v0, v_bar, kappa, sigma_v, rho)
                h_iv = bs_iv(price, S, K, tau[i], r)
                err += (iv[i] - h_iv) ** 2
            return err / len(m)

        res = minimize(objective, self.params, bounds=[(1e-4, 1), (1e-4, 1), (0.1, 10), (1e-4, 1), (-0.99, 0.99)])
        self.params = res.x
        return self

    def predict(self, m, tau, S, r):
        v0, v_bar, kappa, sigma_v, rho = self.params
        preds = []
        for i in range(len(m)):
            K = S / m[i]
            price = self.cos_price(S, K, tau[i], r, v0, v_bar, kappa, sigma_v, rho)
            preds.append(bs_iv(price, S, K, tau[i], r))
        return np.array(preds)


# ==========================================
# 4. Carr and Wu (CW) 模型 [cite: 167]
# ==========================================

class CWModel:
    """
    Carr and Wu 模型。
    通过求解无套利限制下的二次方程来获得 IV [cite: 170, 189]。
    """

    def __init__(self):
        self.theta = [0.04, 0.0, 0.1, 0.5, -0.5]  # [v, m_drift, w, eta, rho] [cite: 184]

    def _solve_sigma_sq(self, k, tau, v, m_drift, w, eta, rho):
        """求解公式 (7) 的二次方程 [cite: 180, 189]"""
        exp_eta = np.exp(-eta * tau)
        A = 0.25 * (exp_eta ** 2) * (w ** 2) * (tau ** 2)
        B = (1 - 2 * exp_eta * m_drift * tau - exp_eta * w * rho * np.sqrt(v) * tau)
        C = -(v + 2 * exp_eta * w * rho * np.sqrt(v) * k + (exp_eta ** 2) * (w ** 2) * (k ** 2))

        # 求解 Ax^2 + Bx + C = 0，取正根
        delta = B ** 2 - 4 * A * C
        return (-B + np.sqrt(delta)) / (2 * A)

    def fit(self, m, tau, iv):
        # 最小化公式 (8) [cite: 186, 187]
        def objective(p):
            v, m_drift, w, eta, rho = p
            k = np.log(1.0 / m)  # 相对执行价 [cite: 179]
            err = 0
            for i in range(len(m)):
                sigma_sq_pred = self._solve_sigma_sq(k[i], tau[i], v, m_drift, w, eta, rho)
                err += (iv[i] ** 2 - sigma_sq_pred) ** 2
            return err

        res = minimize(objective, self.theta)
        self.theta = res.x
        return self

    def predict(self, m, tau):
        v, m_drift, w, eta, rho = self.theta
        k = np.log(1.0 / m)
        sigma_sq = [self._solve_sigma_sq(k[i], tau[i], v, m_drift, w, eta, rho) for i in range(len(m))]
        return np.sqrt(np.maximum(0, sigma_sq))

    # ==========================================
    # 5. Bates 模型 (Heston + Jump)
    # ==========================================

    class BatesModel(HestonModel):
        """
        Bates (1996/2000) 跳跃-扩散模型。
        在 Heston 随机波动率的基础上，加入泊松对数正态跳跃过程以捕捉极端左尾风险。
        继承 HestonModel 以复用 COS 方法的核心定价逻辑。
        """

        def __init__(self):
            super().__init__()
            # 参数: Heston参数 + Jump参数 [v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J]
            # lambda_ : 每年跳跃的平均次数 (跳跃强度)
            # mu_J    : 跳跃幅度的均值 (通常为负，代表左尾暴跌)
            # sigma_J : 跳跃幅度的波动率
            self.params = [0.04, 0.04, 2.0, 0.3, -0.7, 0.1, -0.05, 0.1]

        def _char_func_bates(self, u, T, r, v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J):
            """Bates 模型的特征函数"""
            # 1. 计算风险中性测度下的跳跃漂移补偿项 (Jump Compensator)
            omega = lambda_ * (np.exp(mu_J + 0.5 * sigma_J ** 2) - 1)

            # 2. 调用父类 Heston 的特征函数，注意将无风险利率 r 修正为 (r - omega) 以满足无套利条件
            cf_heston = super()._char_func(u, T, r - omega, v0, v_bar, kappa, sigma_v, rho)

            # 3. 计算纯跳跃过程 (Merton Jump) 的特征函数
            cf_jump = np.exp(lambda_ * T * (np.exp(1j * u * mu_J - 0.5 * (u * sigma_J) ** 2) - 1))

            # 4. 独立过程的特征函数相乘
            return cf_heston * cf_jump

        def cos_price_bates(self, S, K, T, r, v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J, N=128):
            """复写 COS 定价逻辑以使用 Bates 特征函数"""
            x, k = np.log(S / K), np.arange(N)
            a, b = -10, 10
            u = k * np.pi / (b - a)
            cf = self._char_func_bates(u, T, r, v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J)
            term = np.exp(1j * k * np.pi * (x - a) / (b - a))
            return S * np.real(np.sum(cf * term)) / N

        def fit(self, m, tau, iv, S, r):
            """估计 8 个结构参数：最小化隐含波动率误差 (IVMSE)"""

            def objective(p):
                v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J = p
                err = 0
                for i in range(len(m)):
                    K = S / m[i]
                    price = self.cos_price_bates(S, K, tau[i], r, v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J,
                                                 sigma_J)
                    # 使用基础的 Newton-Raphson 反解 IV
                    h_iv = bs_iv(price, S, K, tau[i], r)
                    err += (iv[i] - h_iv) ** 2
                return err / len(m)

            # 在 Heston Bounds 的基础上，增加跳跃参数的边界限制
            # lambda_ \in [0, 5], mu_J \in [-1, 1], sigma_J \in [1e-4, 1]
            bounds = [(1e-4, 1), (1e-4, 1), (0.1, 10), (1e-4, 1), (-0.99, 0.99),
                      (0.0, 5.0), (-1.0, 1.0), (1e-4, 1.0)]

            res = minimize(objective, self.params, bounds=bounds)
            self.params = res.x
            return self

        def predict(self, m, tau, S, r):
            """Bates 模型的预测逻辑"""
            v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J = self.params
            preds = []
            for i in range(len(m)):
                K = S / m[i]
                price = self.cos_price_bates(S, K, tau[i], r, v0, v_bar, kappa, sigma_v, rho, lambda_, mu_J, sigma_J)
                preds.append(bs_iv(price, S, K, tau[i], r))
            return np.array(preds)