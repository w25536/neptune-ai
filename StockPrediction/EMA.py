import pandas as pd
import matplotlib.pyplot as plt
import neptune
import numpy as np
from train_test_split import train_test_split
from data_processing import calculate_rmse, calculate_mape, calculate_perf_metrics
from visualization import plot_stock_trend


df = pd.read_csv('./Datasets/AAPL_historic_stock_data.csv')

df.columns = df.columns.str.strip()

stockprices = df[["Date", "Low", "High", "Close", "Open"]]

test_ratio = 0.2

train, test, train_size, test_size = train_test_split(stockprices, test_ratio)

window_size = 50


# Initialize a Neptune run
run = neptune.init_run(
    project="common/fbprophet-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["prophet", "additional regressors", "script"],  # optional
)

###### Exponential MA
window_ema_var = f"{window_size}_EMA"

# Calculate the 50-day exponentially weighted moving average
stockprices[window_ema_var] = (
    stockprices["Close"].ewm(span=window_size, adjust=False).mean()
)
stockprices["200day"] = stockprices["Close"].rolling(200).mean()

### Plot and performance metrics for EMA model
plot_stock_trend(var=window_ema_var, 
                 cur_title="Exponetial Moving Averages", 
                 stockprices=stockprices, 
                 run_instance=run, 
                 neptune_instance=neptune)
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var, run=run, stockprices=stockprices, train_size=train_size)



# Print the evaluation metrics
print(f"RMSE EMA: {rmse_ema}")
print(f"MAPE EMA: {mape_ema}")


### Stop the run
run.stop()
