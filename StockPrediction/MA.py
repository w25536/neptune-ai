import pandas as pd
import matplotlib.pyplot as plt
import neptune
import numpy as np
from train_test_split import train_test_split
from data_processing import calculate_rmse, calculate_mape, calculate_perf_metrics
from visualization import plot_stock_trend


df = pd.read_csv('./Datasets/TSLA.csv')

df.columns = df.columns.str.strip()

stockprices = df[["Date", "Low", "High", "Close", "Open"]]

test_ratio = 0.2

train, test, train_size, test_size = train_test_split(stockprices, test_ratio)

window_size = 50

# Initialize a Neptune run
run = neptune.init_run(
    project="w25536/StockPrediction",
    name="SMA",
    description="stock-prediction-machine-learning",
    tags=["stockprediction", "MA_Simple", "neptune"],
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZGUxZWY5Yi01YTdjLTQ5YjEtYjQ5YS02OGYyMjA1NDMyMzkifQ==", # your credentials

    
)

window_var = f"{window_size}day"

stockprices[window_var] = stockprices["Close"].rolling(window_size).mean()

### Include a 200-day SMA for reference
stockprices["200day"] = stockprices["Close"].rolling(200).mean()

### Plot and performance metrics for SMA model
plot_stock_trend(var=window_var, 
                 cur_title="Simple Moving Averages", 
                 stockprices=stockprices, 
                 run_instance=run, 
                 neptune_instance=neptune)

rmse_sma, mape_sma = calculate_perf_metrics(var=window_var, 
                                            run=run, 
                                            stockprices=stockprices, 
                                            train_size=train_size)
# Print the evaluation metrics
print(f"RMSE SMA: {rmse_sma}")
print(f"MAPE SMA: {mape_sma}")

### Stop the run
run.stop()




