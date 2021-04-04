import connexion
import six
import sqlite3
import time
import requests
import json
import datetime


from swagger_server.models.history_data import HistoryData  # noqa: E501
from swagger_server.models.predict_data import PredictData  # noqa: E501
from swagger_server.models.quote_data import QuoteData  # noqa: E501
from swagger_server import util

api_key = 'ROPCJ3JV78ML1ZMD'
time_valid = 1800

def predict_stock(ticker):  # noqa: E501
    """Returns

    By passing in the appropriate symbol you can get prediction data for ticker  # noqa: E501

    :param ticker: pass stock ticker for prediction
    :type ticker: str

    :rtype: PredictData
    """
    return 'do some magic!'


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
    return stock_h


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
    return stock


def update_stock_history(symbol, json_history, last_called,conn):
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


def update_stock(symbol, valid, json_stock,conn):
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


def get_old_stock_history(symbol,conn):
    database_command = f"SELECT * FROM STOCK WHERE SYMBOL = '{symbol}'"
    cursor = conn.execute(database_command)
    r = [dict((cursor.description[i][0], value)
              for i, value in enumerate(row)) for row in cursor.fetchall()]

    if r:
        return r[0]
    else:
        return None


def get_old_stock(symbol,conn):
    database_command = f"SELECT * FROM STOCK WHERE SYMBOL = '{symbol}'"
    cursor = conn.execute(database_command)
    r = [dict((cursor.description[i][0], value)
              for i, value in enumerate(row)) for row in cursor.fetchall()]

    if r:
        r[0]
    else:
        return None


def call_api_tim_series_daily_adjusted(symbol, exists,conn):
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&"
                            f"symbol={symbol}&outputsize=compact&apikey={api_key}")
    ts = 'Time Series (Daily)'
    data_json = response.text
    if ts in response.json() and len(response.json()[ts]) > 0:
        if exists:
            update_stock_history(symbol, data_json, datetime.datetime.utcnow().date(),conn)
        else:
            insert_stock_history(symbol, data_json, datetime.datetime.utcnow().date(),conn)
        return data_json
    else:
        return get_old_stock_history(symbol)


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
        return get_old_stock(symbol)


def get_stock(symbol, conn):
    """
    | Function searches DB of TABLE STOCK
    | - Symbol is in DB and is *valid returns JSON
    | - Symbol is in DB and is not *valid calls  API updates to DB and returns JSON
    | - Symbol is not in DB calls API inserts data to DB and returns JSON
    | - Calls for API are depleted returns old data JSON
    | - Calls for API are depleted and there are no data in DB returns None
    | *valid - data are valid for 1800 second
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


def insert_stock(symbol, valid, json_stock,conn):
    database_command = f"INSERT INTO STOCK (SYMBOL,VALID,JSON) " \
                       f"VALUES('{symbol}',{valid},'{json_stock}') "
    try:
        conn.execute(database_command)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def insert_stock_history(symbol, json_history, last_called,conn):
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
        if r[0]['LAST_CALLED'] == datetime.datetime.utcnow().date():
            # vraceni pokud existuje a je validni
            return r[0]['JSON']
        else:
            # tady se vola pokud uz existuje ale neni validni
            return call_api_tim_series_daily_adjusted(symbol, True, conn)
    else:
        # tady vola pokud neexistuje
        return call_api_tim_series_daily_adjusted(symbol, False, conn)
