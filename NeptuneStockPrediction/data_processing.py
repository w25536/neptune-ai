import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from neptune.integrations.tensorflow_keras import NeptuneCallback


## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N : i])
        y.append(data[i])

    return np.array(X), np.array(y)




#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def calculate_perf_metrics(var, run, stockprices, train_size):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )

    ## Log to Neptune
    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape


def run_lstm_model(X_train, y_train, layer_units):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def train_lstm_model(X_train, y_train, cur_epochs, cur_batch_size, run):
    layer_units = 50
    optimizer = "adam"

    model = run_lstm_model(X_train, y_train, layer_units)

    neptune_callback = NeptuneCallback(run=run)

    history = model.fit(
        X_train,
        y_train,
        epochs=cur_epochs,
        batch_size=cur_batch_size,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
        callbacks=[neptune_callback]
    )

    return model, history

def calculate_perf_metrics_lstm(model, X_test, test, scaler):
    predicted_price_ = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_)

    test["Predictions_lstm"] = predicted_price

    rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
    mape_lstm = calculate_mape(np.array(test["Close"]), np.array(test["Predictions_lstm"]))

    return rmse_lstm, mape_lstm


def preprocess_test_data(stockprices, scaler, window_size, test):
    raw = stockprices["Close"][len(stockprices) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


def train_and_evaluate_lstm(stockprices, train_size, window_size, cur_epochs, cur_batch_size, test, run):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stockprices[["Close"]])
    scaled_data_train = scaled_data[:train_size]

    X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)

    layer_units = 50

    model = run_lstm_model(X_train, y_train, layer_units)

    neptune_callback = NeptuneCallback(run=run)

    history = model.fit(
        X_train,
        y_train,
        epochs=cur_epochs,
        batch_size=cur_batch_size,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
        callbacks=[neptune_callback]
    )
    
    scaled_data_test = scaled_data[train_size - window_size:]
    X_test = np.array([scaled_data_test[i-window_size:i, 0] for i in range(window_size, scaled_data_test.shape[0])])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    predicted_price_ = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_)

    test["Predictions_lstm"] = predicted_price

    rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
    mape_lstm = calculate_mape(np.array(test["Close"]), np.array(test["Predictions_lstm"]))

    return rmse_lstm, mape_lstm