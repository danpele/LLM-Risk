import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t
from scipy.special import gamma

class GARCHVaRCalculator:
    def __init__(self, returns, alpha, window, nu=None):
        self.returns = returns
        self.alpha = np.ndarray([alpha]) if isinstance(alpha, float) or isinstance(alpha, np.ndarray) else np.array(alpha)
        self.window = window
        self.nu = nu

    def GARCH_VaR(self):
        def garch_log_likelihood(params, returns):
            omega, garch_alpha, garch_beta = params
            n = len(returns)
            var = np.zeros(n)
            var[0] = np.var(returns) + 1e-6
            for t in range(1, n):
                var[t] = omega + garch_alpha * returns.iloc[t-1]**2 + garch_beta * var[t-1]
                if var[t] < 1e-6 or np.isnan(var[t]) or np.isinf(var[t]):
                    var[t] = 1e-6
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + returns**2 / var)
            return -log_likelihood

        VaR_series, ES_series = {}, {}
        garch_volatility = np.zeros(len(self.returns))

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns[i-self.window:i]
            current_index = self.returns.index[i]
            initial_params = [0.1, 0.1, 0.8]
            bounds = [(1e-6, None), (0, 1), (0, 1)]
            result = minimize(garch_log_likelihood, initial_params, args=(window_returns,), bounds=bounds, method='L-BFGS-B')

            if result.success:
                omega, garch_alpha, garch_beta = result.x
                var_forecast = omega + garch_alpha * window_returns.iloc[-1]**2 + garch_beta * np.var(window_returns)
                if var_forecast < 1e-6:
                    var_forecast = 1e-6
                forecast_var = np.sqrt(var_forecast)
                garch_volatility[i] = forecast_var
                z_score = norm.ppf(1 - self.alpha)
                z_pdf = norm.pdf(z_score)
                VaR_series[current_index] = -z_score * forecast_var
                ES_series[current_index] = forecast_var / self.alpha * z_pdf

        return pd.concat([
            pd.Series(garch_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)

    def GARCH_t_VaR(self):
        def garch_log_likelihood(params, returns, nu):
            omega, garch_alpha, garch_beta = params
            n = len(returns)
            var = np.zeros(n)
            var[0] = np.var(returns) + 1e-6
            for t in range(1, n):
                var[t] = omega + garch_alpha * returns.iloc[t-1]**2 + garch_beta * var[t-1]
                if var[t] < 1e-6 or np.isnan(var[t]) or np.isinf(var[t]):
                    var[t] = 1e-6
            log_likelihood_constant_term = np.log(gamma((nu + 1) / 2)) - np.log(gamma(nu / 2)) - 0.5 * np.log((nu - 2) * np.pi)
            log_likelihood_variable_term = - 0.5 * np.log(var) - (nu + 1) / 2 * np.log(1 + (returns ** 2) / ((nu - 2) * var))
            log_likelihood = np.sum(log_likelihood_constant_term + log_likelihood_variable_term)
            return log_likelihood

        VaR_series, ES_series = {}, {}
        garch_volatility = np.zeros(len(self.returns))

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns[i-self.window:i]
            current_index = self.returns.index[i]
            initial_params = [0.1, 0.1, 0.8]
            bounds = [(1e-6, None), (0, 1), (0, 1)]
            result = minimize(garch_log_likelihood, initial_params, args=(window_returns, self.nu), bounds=bounds, method='L-BFGS-B')

            if result.success:
                omega, garch_alpha, garch_beta = result.x
                var_forecast = omega + garch_alpha * window_returns.iloc[-1]**2 + garch_beta * np.var(window_returns)
                if var_forecast < 1e-6:
                    var_forecast = 1e-6
                forecast_var = np.sqrt(var_forecast)
                garch_volatility[i] = forecast_var
                t_score = t.ppf(1 - self.alpha, df=self.nu)
                t_ES = t.pdf(t_score, df=self.nu) / self.alpha
                t_ES *= (self.nu + t_score ** 2) / (self.nu - 1)
                VaR_series[current_index] = -t_score * forecast_var
                ES_series[current_index] = forecast_var * t_ES

        return pd.concat([
            pd.Series(garch_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)

    def GAS_GARCH_VaR(self):
        def score_function(returns, var):
            return (returns**2 / var - 1)

        def gas_log_likelihood(params, returns):
            omega, garch_alpha, garch_beta = params
            n = len(returns)
            var = np.zeros(n)
            var[0] = np.var(returns) + 1e-6  # Initialize with sample variance
            score = np.zeros(n)
            
            for t in range(1, n):
                score[t-1] = score_function(returns.iloc[t-1], var[t-1])
                var[t] = omega + garch_alpha * score[t-1] * var[t-1] + garch_beta * var[t-1]
                if var[t] < 1e-6 or np.isnan(var[t]) or np.isinf(var[t]):
                    var[t] = 1e-6
            
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + returns**2 / var)
            return -log_likelihood

        VaR_series, ES_series = {}, {}
        garch_volatility = np.zeros(len(self.returns))

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns[i-self.window:i]
            current_index = self.returns.index[i]
            initial_params = [0.01, 0.01, 0.9]
            bounds = [(1e-6, None), (0, 1), (0, 1)]
            result = minimize(gas_log_likelihood, initial_params, args=(window_returns,), bounds=bounds, method='L-BFGS-B')

            if result.success:
                omega, garch_alpha, garch_beta = result.x
                last_return = window_returns.iloc[-1]
                last_var = np.var(window_returns)
                last_score = score_function(last_return, last_var)
                var_forecast = omega + garch_alpha * last_score * last_var + garch_beta * last_var
                if var_forecast < 1e-6:
                    var_forecast = 1e-6
                forecast_var = np.sqrt(var_forecast)
                garch_volatility[i] = forecast_var
                z_score = norm.ppf(1 - self.alpha)
                z_pdf = norm.pdf(z_score)
                VaR_series[current_index] = -z_score * forecast_var
                ES_series[current_index] = forecast_var / self.alpha * z_pdf

        return pd.concat([
            pd.Series(garch_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)

    def GAS_GARCH_t_VaR(self):
        def score_function(returns, var, nu):
            term1 = 1 + (returns**2 / ((nu - 2) * var))
            term2 = (returns**2 * (nu + 1) / (var * (nu - 2)) - var)
            scaling_term = (nu + 3) / nu
            return scaling_term * term1**(-1) * term2

        def gas_log_likelihood(params, returns, nu):
            omega, garch_alpha, garch_beta = params
            n = len(returns)
            var = np.zeros(n)
            var[0] = np.var(returns) + 1e-6  # Initialize with sample variance
            score = np.zeros(n)
            
            for t in range(1, n):
                score[t-1] = score_function(returns.iloc[t-1], var[t-1], nu)
                var[t] = omega + garch_alpha * score[t-1] * var[t-1] + garch_beta * var[t-1]
                if var[t] < 1e-6 or np.isnan(var[t]) or np.isinf(var[t]):
                    var[t] = 1e-6
            
            log_likelihood = -0.5 * np.sum(np.log(var) + (nu + 1) * np.log(1 + returns**2 / (nu * var)))
            return -log_likelihood

        VaR_series, ES_series = {}, {}
        garch_volatility = np.zeros(len(self.returns))

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns[i-self.window:i]
            current_index = self.returns.index[i]
            initial_params = [0.01, 0.01, 0.9]
            bounds = [(1e-6, None), (0, 1), (0, 1)]
            options = {'maxiter': 100}
            result = minimize(gas_log_likelihood, initial_params, args=(window_returns, self.nu), bounds=bounds, method='L-BFGS-B', options=options)

            if result.success:
                omega, garch_alpha, garch_beta = result.x
                last_return = window_returns.iloc[-1]
                last_var = np.var(window_returns)
                last_score = score_function(last_return, last_var, self.nu)
                var_forecast = omega + garch_alpha * last_score * last_var + garch_beta * last_var
                if var_forecast < 1e-6:
                    var_forecast = 1e-6
                forecast_var = np.sqrt(var_forecast)
                garch_volatility[i] = forecast_var
                t_score = t.ppf(1 - self.alpha, df=self.nu)
                t_ES = t.pdf(t_score, df=self.nu) / self.alpha
                t_ES *= (self.nu + t_score ** 2) / (self.nu - 1)
                VaR_series[current_index] = -t_score * forecast_var
                ES_series[current_index] = forecast_var * t_ES

        return pd.concat([
            pd.Series(garch_volatility[self.window:], name="volatility", index=self.returns.index[self.window:]),
            pd.DataFrame.from_dict(VaR_series, orient="index", columns=[f"VaR_{alpha_:.4f}" for alpha_ in self.alpha]), 
            pd.DataFrame.from_dict(ES_series, orient="index", columns=[f"ES_{alpha_:.4f}" for alpha_ in self.alpha])
        ], axis=1)
