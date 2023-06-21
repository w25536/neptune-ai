import pandas as pd


def train_test_split(stockprices, test_ratio):
    training_ratio = 1 - test_ratio

    train_size = int(training_ratio * len(stockprices))
    test_size = int(test_ratio * len(stockprices))
    
    print(f"train_size: {train_size}")
    print(f"test_size: {test_size}")

    train = stockprices[:train_size][["Date", "Close"]]
    test = stockprices[train_size:][["Date", "Close"]]

    return train, test, train_size, test_size