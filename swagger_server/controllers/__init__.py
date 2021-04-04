import sqlite3

conn = sqlite3.connect('stock.db')
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


create_stock_history_db()
create_stock_db()
conn.close()