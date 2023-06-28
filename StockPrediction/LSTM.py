import pandas as pd
import matplotlib.pyplot as plt
import neptune
import numpy as np
from train_test_split import train_test_split
from data_processing import calculate_rmse, calculate_mape, calculate_perf_metrics,train_and_evaluate_lstm
from visualization import plot_stock_trend, plot_stock_trend_lstm


df = pd.read_csv('./Datasets/TSLA.csv')

df.columns = df.columns.str.strip()


stockprices = df[["Date", "Low", "High", "Close", "Open"]]

test_ratio = 0.3

train, test, train_size, test_size = train_test_split(stockprices, test_ratio)



window_size = 50
cur_epochs = 10
cur_batch_size = 1



run = neptune.init_run(
    project="common/fbprophet-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
    tags=["prophet", "additional regressors", "script"],  # optional
)


# Train and evaluate the LSTM model
rmse_lstm, mape_lstm = train_and_evaluate_lstm(
    stockprices=df,
    train_size=train_size,
    window_size=window_size,
    cur_epochs=cur_epochs,
    cur_batch_size=cur_batch_size,
    test=test,
    run=run
)

# Print the evaluation metrics
print(f"RMSE LSTM: {rmse_lstm}")
print(f"MAPE LSTM: {mape_lstm}")

plot_stock_trend_lstm(train=train, test=test, run_instance=run, neptune_instance=neptune)

# Stop the Neptune run
run.stop()
