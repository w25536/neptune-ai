import matplotlib.pyplot as plt
import numpy as np


def plot_stock_trend(var, cur_title, stockprices, run_instance, neptune_instance):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    ## Log to Neptune
    run_instance["Plot of Stock Predictions"].upload(
        neptune_instance.types.File.as_image(ax.get_figure())
    )
    
    
    
def plot_stock_trend_lstm(train, test, run_instance, neptune_instance):
    fig = plt.figure(figsize = (20,10))
    plt.plot(np.asarray(train["Date"]), np.asarray(train["Close"]), label = "Train Closing Price")
    plt.plot(np.asarray(test["Date"]), np.asarray(test["Close"]), label = "Test Closing Price")
    plt.plot(np.asarray(test["Date"]), np.asarray(test["Predictions_lstm"]), label = "Predicted Closing Price")
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper right")

    ## Log image to Neptune
    run_instance["Plot of Stock Predictions"].upload(
        neptune_instance.types.File.as_image(fig)
    )

