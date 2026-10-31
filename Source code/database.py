
import mysql.connector
from getpass import getpass

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password=getpass("Enter MySQL root password: "),
        database="customer_churn_db"
    )

    if connection.is_connected():
        print("MySQL Database Connected Successfully!")

        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")

        print("\nAvailable Tables:")
        for table in cursor.fetchall():
            print(table[0])

        cursor.close()
        connection.close()

except mysql.connector.Error as error:
    print("Database Connection Failed:", error)
