import sqlite3

connection = sqlite3.connect("database/ecg_database.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(

id INTEGER PRIMARY KEY AUTOINCREMENT,

patient_name TEXT,

age INTEGER,

gender TEXT,

prediction TEXT,

confidence REAL,

date_time TEXT

)
""")

connection.commit()

connection.close()

print("Database Created Successfully")