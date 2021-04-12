import sqlite3
import os

def create_stock_db():
    try:
        conn.execute('''CREATE TABLE STOCK
         ( SYMBOL       VARCHAR(10) PRIMARY KEY     NOT NULL,
         VALID          INTEGER     NOT NULL,
         JSON           TEXT        NOT NULL);''')
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False

def create_stock_history_db():
    try:
        conn.execute('''CREATE TABLE STOCK_HISTORY
         ( SYMBOL        VARCHAR(10) PRIMARY KEY     NOT NULL,
          JSON          TEXT       NOT NULL,
          LAST_CALLED   DATETIME NOT NULL);''')
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False

def create_stock_prediction_db():
    try:
        conn.execute('''CREATE TABLE PREDICTION
         ( SYMBOL        VARCHAR(10) PRIMARY KEY     NOT NULL,
          STATE          TEXT NOT NULL,
          PRICE          REAL,
          MEAN_ABSOLUTE_ERROR    REAL,
          ACCURACY  REAL,
          PRICE_DATE   DATETIME);''')
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False

conn = sqlite3.connect('stock.db')
create_stock_history_db()
create_stock_db()
create_stock_prediction_db()
conn.close()