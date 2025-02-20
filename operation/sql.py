import sqlite3
import os
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/university.db"))
conn=sqlite3.connect(db_path)
#conn=sqlite3.connect("university.db")
mycursor=conn.cursor()

mycursor.execute("select * from student_mark_details")
res=mycursor.fetchall()
for i in res:
    print(i)