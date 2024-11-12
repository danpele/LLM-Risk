import os, json, time
import datetime as dt

import pandas as pd
from arch.univariate import arch_model

from scipy import stats
from joblib import dump

from pylpa.data import get_returns_from_prices
from pylpa.logger import get_logger, LOGGER

from pylpa.lpa import find_largest_homogene_interval
from pylpa.constant import MULTIPLIER, N_0

import numpy as np

from pylpa.utils import default_config

# Config

config = {
  "model": {
    "name": "garch",
    "params": {"p":  1, "q":  1}
  },
  "data": {
    "feature": "log_returns",
    "preprocessing": {
      "name": "StandardScaler"
    }
  },
  "bootstrap": {
    "generate": "exponential",
    "num_sim": 100,
    "njobs": 32
  },
  "min_steps": 5,
  "maxtrial": 1,
  "maxiter": 1000
}

files = [
    "cact.xlsx", "cbu.xlsx", "CRIX.xlsx", "djci.xlsx", "ftse.xlsx", 
    "gdaxi.xlsx", "SP500.xlsx", 
    "SPGTCLTR.xlsx", "stoxx.xlsx"
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            prog="Backtest",
            description="Run the LPA",
        )
    parser.add_argument(
        "--level",
        type=str,
        help="Logger level",
        default="info",
    )
    parser.add_argument(
        "--quantiles",
        help="VaR levels",
        default=[0.01, 0.025, 0.01875, 0.01250, 0.00625],
    )
    args = parser.parse_args()

    if 'seed' in config.keys():
        np.random.seed(seed=config['seed'])

    for f in files:
        config["data"]["path"] = f"data/{f}"
        config = default_config(config)

        # Save directory
        save_dir = config.get("save_dir", f.split(".")[0])
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{save_dir}_{now}"

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        else:
            LOGGER.warning("Overwriting save_dir")
        LOGGER = get_logger(
            "LPA-main", level=args.level, save_path=f"{save_dir}/run.log"
        )

        # dump config
        json.dump(config, open('%s/config.json' % save_dir, 'w'))

        tc1 = time.time()
        fit_window = {}

        dates, returns = get_returns_from_prices(
            path=config["data"]["path"], feature_name=config["data"]["feature"]
        )
        min_size = int(np.round(N_0 * MULTIPLIER ** 2))
        # Get previous N_k
        # If len(window) < previous N_k do not perform test and take last
        # breakpoint
        # Otherwise perform test
        # Fit GARCH on interval
        # Predict
        indices = list(range(min_size, len(returns)))
        value_at_risk = np.zeros((len(indices), len(args.quantiles)))
        expected_shortfall = np.zeros((len(indices), len(args.quantiles)))
        window_means, window_var = {}, {}
        window_cond_means, window_cond_var = {}, {}
        breakpoints = []
        c = 0
        for i in indices:
            window = returns[:i,:]
            intervals = []
            breaks = []
            if config["data"].get("preprocessing") is not None:
                if config["data"]["preprocessing"]["name"] == "StandardScaler":
                    LOGGER.info(
                        'Centering and reducing to mean 0 and variance 1')
                    # normalize test set with train
                    mean_ = np.mean(window)
                    std_ = np.std(window)
                    window = (window - mean_) / std_
                else:
                    raise NotImplementedError(config["preprocessing"])
            else:
                window = window

            interval, index = find_largest_homogene_interval(
                window, config["model"], K=config["K"],
                interval_step=config["interval_step"],
                min_steps=config['min_steps'],
                solver=config['solver'], maxiter=config['maxiter'],
                maxtrial=config['maxtrial'],
                generate=config["bootstrap"]["generate"],
                num_sim=config["bootstrap"]['num_sim'],
                njobs=config["bootstrap"]['njobs']
            )
            if index != -1:
                breakpoints.append([dates[index], index])

            # Fit garch on interval
            am = arch_model(
                interval,
                mean="AR" if config["model"]["name"] == "ARMAGARCH" else "Zero",
                dist="normal", **config["model"]["params"]
            )
            res = am.fit()
            forecasts = res.forecast(horizon=1)
            cond_mean = forecasts.mean
            cond_var = forecasts.variance
            q = am.distribution.ppf(args.quantiles)
            VaR = cond_mean.values + np.sqrt(cond_var).values * q[None, :]
            if am.distribution.name == "Normal":
                ES = stats.norm.pdf(q) * 1/args.quantiles
                ES = (cond_mean.values + np.sqrt(cond_var).values * ES[None, :])
            else:
                raise NotImplementedError
            if config["data"].get("preprocessing") is not None:
                if config["data"]["preprocessing"]["name"] == "StandardScaler":
                    VaR *= std_
                    VaR += mean_
                    ES *= std_
                    ES += mean_
                else:
                    raise NotImplementedError(config["preprocessing"])
            value_at_risk[c, :] = VaR
            expected_shortfall[c, :] = ES
            window_means[i] = mean_
            window_var[i] = std_
            window_cond_means[i] = cond_mean
            window_cond_var[i] = cond_var
            if c % 10 == 0:
                res_VaR = pd.DataFrame(
                    value_at_risk, columns=[f"VaR_{q}" for q in args.quantiles],
                    index=dates[min_size:],
                )
                res_ES = pd.DataFrame(
                    expected_shortfall,
                    columns=[f"ES_{q}" for q in args.quantiles],
                    index=dates[min_size:],
                )
                results = pd.concat([res_VaR, res_ES], axis=1)
                results.to_csv(f"{save_dir}/res_{c}.csv")
                pd.DataFrame(
                    breakpoints, columns=["dates", "index"]
                ).to_csv(f"{save_dir}/res_breakpoints_{c}.csv", index=False)
                dump(window_means, f"{save_dir}/res_means_{c}.pkl")
                dump(window_var, f"{save_dir}/res_var_{c}.pkl")
                dump(window_cond_means, f"{save_dir}/res_cond_means_{c}.pkl")
                dump(window_cond_var, f"{save_dir}/res_cond_var_{c}.pkl")
            c += 1

        # Save forecasts
        try:
            res_VaR = pd.DataFrame(
                value_at_risk, columns=[f"VaR_{q}" for q in args.quantiles],
                index=dates[min_size:i+1],
            )
            res_ES = pd.DataFrame(
                expected_shortfall, columns=[f"ES_{q}" for q in args.quantiles],
                index=dates[min_size:i+1],
            )
            results = pd.concat([res_VaR, res_ES], axis=1)
            results.to_csv(f"{save_dir}/results.csv")
            pd.DataFrame(
                breakpoints, columns=["dates", "index"]
            ).to_csv(f"{save_dir}/breakpoints.csv", index=False)
            dump(window_means, f"{save_dir}/res_means.pkl")
            dump(window_var, f"{save_dir}/res_vars.pkl")
            dump(window_cond_means, f"{save_dir}/res_cond_means.pkl")
            dump(window_cond_var, f"{save_dir}/res_cond_var.pkl")
    
            # Clean temp files
            for i in range(0, c + 1, 10):
                os.remove(f"{save_dir}/res_{i}.csv")
                os.remove(f"{save_dir}/res_breakpoints_{i}.csv")
                os.remove(f"{save_dir}/res_means_{i}.pkl")
                os.remove(f"{save_dir}/res_var_{i}.pkl")
                os.remove(f"{save_dir}/res_cond_means_{i}.pkl")
                os.remove(f"{save_dir}/res_cond_var_{i}.pkl")
        except Exception as e:
            print(f"Encountered error when saving final results: {e}. Use partial results. A few data points will be missing.")
            
