import numpy as np
import pandas as pd
from scipy.stats import norm, t

class EWMAVaRCalculator:
    def __init__(self, returns, lambda_, alpha, window, nu=None):
        self.returns = returns
        self.lambda_ = lambda_
        self.alpha = np.ndarray([alpha]) if isinstance(alpha, float) or isinstance(alpha, np.ndarray) else np.array(alpha)
        self.window = window
        self.nu = nu

    def EWMA_VaR(self):
        VaR_series, ES_series = {}, {}
        ewma_volatility = np.zeros(len(self.returns))
        ewma_volatility[self.window-1] = np.sqrt(np.var(self.returns[:self.window]))

        for i in range(self.window, len(self.returns)):
            current_index = self.returns.index[i]
            ewma_volatility[i] = np.sqrt(self.lambda_ * ewma_volatility[i-1]**2 + (1 - self.lambda_) * self.returns.iloc[i-1]**2)
            z_score = norm.ppf(1 - self.alpha)
            z_pdf = norm.pdf(z_score)
            VaR_series[current_index] = -z_score * ewma_volatility[i]
            ES_series[current_index] = ewma_volatility[i] / self.alpha * z_pdf

        return pd.concat([
            pd.Series(ewma_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)

    def DCS_EWMA_VaR(self):
        VaR_series, ES_series = {}, {}
        ewma_volatility = np.zeros(len(self.returns))
        ewma_volatility[self.window-1] = np.sqrt(np.var(self.returns[:self.window]))
        epsilon = 1e-8

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns.iloc[i-self.window:i]
            current_index = self.returns.index[i]

            for t in range(1, self.window):
                score = 1 + (window_returns.iloc[t] ** 2 - ewma_volatility[i-1] ** 2) / ewma_volatility[i-1] ** 2
                ewma_volatility_squared = self.lambda_ * ewma_volatility[i-1] ** 2 + (1 - self.lambda_) * score * window_returns.iloc[t] ** 2
                ewma_volatility[i] = np.sqrt(max(ewma_volatility_squared, epsilon))

            z_score = norm.ppf(1 - self.alpha)
            z_pdf = norm.pdf(z_score)
            VaR_series[current_index] = -z_score * ewma_volatility[i]
            ES_series[current_index] = ewma_volatility[i] / self.alpha * z_pdf

        return pd.concat([
            pd.Series(ewma_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)

    def SD_EWMA_VaR(self):
        if self.nu is None:
            raise ValueError("Degrees of freedom 'nu' must be provided for SD_EWMA_VaR method.")
        VaR_series, ES_series = {}, {}
        ewma_volatility = np.zeros(len(self.returns))
        initial_volatility = np.sqrt(np.var(self.returns[:self.window]))
        ewma_volatility[self.window-1] = initial_volatility
        epsilon = 1e-8
        A = (1 - self.lambda_) / (1 + 3 * self.nu**-1)

        for i in range(self.window, len(self.returns)):
            y_t = self.returns.iloc[i]
            current_index = self.returns.index[i]
            f_t = ewma_volatility[i-1] ** 2
            f_t = max(f_t, epsilon)
            score = ((self.nu + 1) / (self.nu - 2 + (y_t**2 / f_t))) * y_t**2 - f_t
            ewma_volatility_squared = f_t + A * (1 + 3 * self.nu**-1) * score
            ewma_volatility[i] = np.sqrt(max(ewma_volatility_squared, epsilon))
            t_score = t.ppf(1 - self.alpha, df=self.nu)
            t_ES = t.pdf(t_score, df=self.nu) / self.alpha
            t_ES *= (self.nu + t_score ** 2) / (self.nu - 1)
            VaR_series[current_index] = -t_score * ewma_volatility[i]
            ES_series[current_index] = ewma_volatility[i] * t_ES

        return pd.concat([
            pd.Series(ewma_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)
