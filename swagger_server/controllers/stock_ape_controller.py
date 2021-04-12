import connexion
import six
import sqlite3
import time
import requests
import json
import datetime
import os
from swagger_server.models.history_data import HistoryData  # noqa: E501
from swagger_server.models.predict_data import PredictData  # noqa: E501
from swagger_server.models.quote_data import QuoteData  # noqa: E501
from swagger_server import util
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import numpy as np
import pandas as pd
import random
import threading
# import matplotlib.pyplot as plt


api_key = 'ROPCJ3JV78ML1ZMD'
time_valid = 3600
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 100



def predict_stock(ticker):  # noqa: E501
    """Returns

    By passing in the appropriate symbol you can get prediction  for ticker  # noqa: E501

    :param ticker: pass stock ticker for prediction
    :type ticker: str

    :rtype: PredictData
    """
    ticker = ticker.upper()
    conn = sqlite3.connect('stock.db')
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    prediction = get_prediction(ticker, ticker_data_filename, model_name, conn)
    conn.close()
    return json.dumps(prediction)


def stock_history(ticker):  # noqa: E501
    """Returns

    By passing a stock ticker you will get returned data 100 days back  # noqa: E501

    :param ticker: pass stock ticker for history
    :type ticker: str

    :rtype: HistoryData
    """
    conn = sqlite3.connect('stock.db')
    stock_h = get_stock_history(ticker, conn)
    conn.close()
    return json.loads(stock_h)


def stock_quote(ticker):  # noqa: E501
    """Returns

    By passing a stock ticker you will get returned latest quote data from ALPHAVANTAGE or Database  # noqa: E501

    :param ticker: pass stock ticker for quote
    :type ticker: str

    :rtype: QuoteData
    """

    conn = sqlite3.connect('stock.db')
    stock = get_stock(ticker, conn)
    conn.close()
    return json.loads(stock)










def update_stock_history(symbol, json_history, last_called, conn):
    database_command = f"UPDATE STOCK_HISTORY " \
                       f"SET JSON = '{json_history}'," \
                       f"LAST_CALLED ='{last_called}'" \
                       f"WHERE SYMBOL = '{symbol}';"
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.OperationalError:
        print("error while updating")
        return False


def update_stock(symbol, valid, json_stock, conn):
    database_command = f"UPDATE STOCK " \
                       f"SET VALID = {valid}," \
                       f"JSON ='{json_stock}'" \
                       f"WHERE SYMBOL = '{symbol}';"
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print("error while updating")
        return False


def get_old_stock_history(symbol, conn):
    database_command = f"SELECT * FROM STOCK WHERE SYMBOL = '{symbol}'"
    cursor = conn.execute(database_command)
    r = [dict((cursor.description[i][0], value)
              for i, value in enumerate(row)) for row in cursor.fetchall()]

    if r:
        return r[0]
    else:
        return None


def get_old_stock(symbol, conn):
    database_command = f"SELECT * FROM STOCK WHERE SYMBOL = '{symbol}'"
    cursor = conn.execute(database_command)
    r = [dict((cursor.description[i][0], value)
              for i, value in enumerate(row)) for row in cursor.fetchall()]

    if r:
        return r[0]
    else:
        return None


def call_api_tim_series_daily_adjusted(symbol, exists, conn):
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&"
                            f"symbol={symbol}&outputsize=compact&apikey={api_key}")
    ts = 'Time Series (Daily)'
    data_json = response.text
    if ts in response.json() and len(response.json()[ts]) > 0:
        if exists:
            update_stock_history(symbol, data_json, datetime.datetime.utcnow().date(), conn)
        else:
            insert_stock_history(symbol, data_json, datetime.datetime.utcnow().date(), conn)
        return data_json
    else:
        return get_old_stock_history(symbol, conn)


def call_api_global_quote(symbol, exists, conn):
    response = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}")
    gq = 'Global Quote'
    data_json = response.text
    # kontola zda API vraci nejake data a nebo neni vycerpany pocet callu
    if gq in response.json() and len(response.json()[gq]) > 0:
        if exists:
            update_stock(symbol, time.time() + time_valid, data_json, conn)
        else:
            insert_stock(symbol, time.time() + time_valid, data_json, conn)
        return data_json
    else:
        return get_old_stock(symbol, conn)


def get_stock(symbol, conn):
    """
    | Function searches DB of TABLE STOCK
    | - Symbol is in DB and is *valid returns JSON
    | - Symbol is in DB and is not *valid calls  API updates to DB and returns JSON
    | - Symbol is not in DB calls API inserts data to DB and returns JSON
    | - Calls for API are depleted returns old data JSON
    | - Calls for API are depleted and there are no data in DB returns None
    | *valid - data are valid for 3600 second
    :type symbol: str
    :return JSON or None
    """

    database_command = f"SELECT * FROM STOCK WHERE SYMBOL = '{symbol}'"
    try:
        cursor = conn.execute(database_command)
        r = [dict((cursor.description[i][0], value)
                  for i, value in enumerate(row)) for row in cursor.fetchall()]
    except sqlite3.IntegrityError:
        return None

    if r:
        if (r[0]['VALID'] - time.time()) > 0:
            # vraceni pokud existuje a je validni
            return r[0]['JSON']
        else:
            # tady se vola pokud uz existuje ale neni validni
            return call_api_global_quote(symbol, True, conn)
    else:
        # tady vola pokud neexistuje
        return call_api_global_quote(symbol, False, conn)


def insert_stock(symbol, valid, json_stock, conn):
    database_command = f"INSERT INTO STOCK (SYMBOL,VALID,JSON) " \
                       f"VALUES('{symbol}',{valid},'{json_stock}') "
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def insert_stock_history(symbol, json_history, last_called, conn):
    database_command = f"INSERT INTO STOCK_HISTORY (SYMBOL,JSON,LAST_CALLED) " \
                       f"VALUES('{symbol}','{json_history}','{last_called}') "
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_stock_history(symbol, conn):
    """
    | Function searches TABLE of DB STOCK_HISTORY
    | - the symbol is in DB and is *valid returns JSON
    | - the symbol is in DB and is not v*alid calls API updates to DB and returns JSON
    | - the symbol is not in DB calls API inserts data to DB and returns JSON
    | - calls for API are depleted returns old data JSON
    | - calls for API are depleted and there are no data in Database returns None
    | *valid - data are valid for 1800s
    :param symbol: symbol is ticker of stock
    :return JSON or None
    """
    database_command = f"SELECT * FROM STOCK_HISTORY WHERE SYMBOL = '{symbol}'"
    try:
        cursor = conn.execute(database_command)
        r = [dict((cursor.description[i][0], value)
                  for i, value in enumerate(row)) for row in cursor.fetchall()]
    except sqlite3.IntegrityError:
        return None

    if r:
        if str(r[0]['LAST_CALLED']) == str(datetime.datetime.utcnow().date()):
            # vraceni pokud existuje a je validni
            return r[0]['JSON']
        else:
            # tady se vola pokud uz existuje ale neni validni
            return call_api_tim_series_daily_adjusted(symbol, True, conn)
    else:
        # tady vola pokud neexistuje
        return call_api_tim_series_daily_adjusted(symbol, False, conn)


def train_stock(ticker,ticker_data_filename, model_name,conn):
    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")

    # load the data
    data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                     shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS)

    # save the dataframe
    data["df"].to_csv(ticker_data_filename)

    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
                                   save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)
    test(ticker, model_name, conn)


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=False,
              test_size=0.1, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, true_future, pred_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, true_future, pred_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


def update_prediction(symbol, state, future_price, mean_absolute_error, accuracy, conn):

    dt = datetime.datetime.now()
    td = datetime.timedelta(days=LOOKUP_STEP)
    future_date = dt + td
    database_command = f"UPDATE PREDICTION " \
                       f"SET STATE = '{state}'," \
                       f"PRICE ={future_price}," \
                       f"MEAN_ABSOLUTE_ERROR ={mean_absolute_error}," \
                       f"ACCURACY ={accuracy}," \
                       f"PRICE_DATE ='{future_date}'" \
                       f"WHERE SYMBOL = '{symbol}';"
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def insert_prediction_empty(symbol,state,conn):
    database_command = f"INSERT INTO PREDICTION (SYMBOL,STATE) " \
                       f"VALUES('{symbol}','{state}') "
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def test(ticker,model_name,conn):
    # load the data
    data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                     shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS)

    # construct the model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # get the final dataframe for the testing set
    final_df = get_final_df(model, data)
    # predict the future price
    future_price = predict(model, data)
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(
        final_df)

    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)
    update_prediction(ticker, "DONE", future_price, mean_absolute_error, accuracy_score, conn)


def check_date(prediction_date, time_back):

    date_time_obj = datetime.datetime.strptime(prediction_date, '%Y-%m-%d %H:%M:%S.%f')
    date_valid = datetime.datetime.now() + datetime.timedelta(time_back)
    print(f"prediction date :{prediction_date}\ntime back :{time_back}\n datetimeobj:{date_time_obj}\ndate_valid{date_valid}")
    if date_time_obj > date_valid:
        return True
    else:
        return False


def new_prediction(ticker, ticker_data_filename, model_name, conn):
    insert_prediction_empty(ticker, "PREDICTING", conn)
    th = threading.Thread(train_stock(ticker, ticker_data_filename, model_name,conn))
    th.start()


def get_prediction(ticker, ticker_data_filename, model_name, conn):
    database_command = f"SELECT * FROM PREDICTION WHERE SYMBOL = '{ticker}'"
    try:
        cursor = conn.execute(database_command)
        r = [dict((cursor.description[i][0], value)
                  for i, value in enumerate(row)) for row in cursor.fetchall()]
    except sqlite3.IntegrityError:
        return "ERROR"

    if r:
        if r[0]['STATE'] == "PREDICTING":
            print(1)
            return "predicting"

        elif r[0]['STATE'] == "DONE":
            print(2)
            if check_date(str(r[0]['PRICE_DATE']),5):
                print(3)
                return str(r[0])
            else:
                print(4)
                new_prediction(ticker,ticker_data_filename,model_name,conn)
                return "predicting"
    else:
        print(5)
        new_prediction(ticker,ticker_data_filename,model_name,conn)
        return "predicting"
