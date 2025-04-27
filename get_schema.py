import sqlite3
import sys

db_path = 'data/processed/bible.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schema = cursor.fetchall()
    for table_schema in schema:
        print(table_schema[0])
    conn.close()
except Exception as e:
    print(f"Error accessing database: {e}", file=sys.stderr)
