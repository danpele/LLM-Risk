# LLM VaR

## Installation

### Create environment

Run the following command to install all dependencies in a conda environment named `llmtime`. Change the cuda version for torch if you have NVIDIA drivers with cuda < 11.8. 

#### Mac OS/Linux

```
source install.sh
```

#### Windows

From CMD:
```
install.bat
```

### Add OpenAI API key

Create a file called `.env` in the main folder and add the following line:

```
OPENAI_API_API_KEY=<your api key>
```

## Run LLM Simulations

Use the `LLM Simulations.ipynb` notebook. By default, all assets and only windows of sizes 30, 45 are chosen. Check all available parameters.

Each LLM must be simulated separately. For GPT 3.5, it takes around 8 hours to finish the full simulation, with much higher times for GPT 4. GPT 4o sits in the middle.

## Benchmarks

1. To run GARCH/EWMA models, use the `Run EWMA GARCH.ipynb` notebook. Currently running GARCH takes around 40 minutes with parallel processing because of CRIX. You can cut CRIX to start from the overall `starting_date` in `LLMSimulations.ipynb` notebook minus your prefered window size to make it run faster. 
2. If run GARCH LPA, use the script `multiple_backtest.py` from this project's root. After you clone <a href="https://github.com/QuantLet/pylpa">`pylpa`</a>, replace it in `pylpa/scripts`. Parameters are available. On our 9 assets this runs in around 7 - 8 hours with 32 CPUs and 200 simulations. The same trick with CRIX should also work here. Copy/Move the resulting directories into the `GARCH/LPA` directory.

## Backtesting

The notebook `Backtest VaR & ES.ipynb` includes all backtests. Read through the configuration at the beginning and change paths according to your LLM simulations (benchmarks should not be modified).

Be advised that $Z_3$ backtest runs rather slow, around 20 minutes.


## Long-running notebooks

Keeping notebooks open for a long period of time may not be the best solution (working remotely can create problems when closing the browser tab, e.g. in VS Code). We avoided this issue because we use Jupyter Lab.

Transforming notebooks to scripts (e.g. using `Import Notebook to Script` option in Visual Studio Code) might be an good alternative. Both <a href="https://www.scaler.com/topics/how-to-run-process-in-background-linux/">Mac OS/Linux</a> and <a href="https://medium.com/@sakethvrudraraju/how-to-run-a-python-script-in-the-background-on-windows-95987864ef3e">Windows</a> provide easy solutions to run scripts in the background.

However, unless intermediary results are saved, one must ensure the script treats every error possible (not assured for benchmarking models & backtesting), or the whole execution would have to be started from scratch in case of runtime errors.
