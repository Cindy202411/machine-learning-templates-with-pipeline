import sqlite3
import pandas as pd

def load_data(sqlite_path, table_name):
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

