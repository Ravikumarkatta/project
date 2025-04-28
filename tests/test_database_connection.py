import unittest
import sqlite3
import os
from pathlib import Path

class TestDatabaseConnection(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_db.db"
        self.conn = sqlite3.connect(self.test_db)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def test_connection(self):
        """Test database connection and table creation"""
        self.cursor.execute("INSERT INTO test_table (name) VALUES ('test')")
        self.conn.commit()
        self.cursor.execute("SELECT * FROM test_table")
        result = self.cursor.fetchall()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'test')

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

if __name__ == '__main__':
    unittest.main()
