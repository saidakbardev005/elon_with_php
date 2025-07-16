import mysql.connector
from mysql.connector import Error
from config import Config

def get_connection():
    """
    mysql.connector bilan MySQL serveriga ulanadi.
    Ulanish davomida xatolik bo‘lsa, None qaytaradi.
    """
    try:
        conn = mysql.connector.connect(
            host     = Config.DB_HOST,
            port     = Config.DB_PORT,
            user     = Config.DB_USER,
            password = Config.DB_PASSWORD,
            database = Config.DB_NAME,
            charset  = "utf8mb4",
            use_pure = True
        )
        # Ulanish jonliligini tekshirish, uzilish bo‘lsa qayta urinadi
        conn.ping(reconnect=True, attempts=3, delay=5)
        print("[db] MySQL serveriga muvaffaqiyatli ulanildi.")
        return conn

    except Error as e:
        print(f"[db] MySQL ulanish xatosi: {e}")
        return None


def get_cursor():
    """
    Faol ulanishdan cursor qaytaradi.
    Agar ulanmagan bo‘lsa, avval qayta ulanishga urinadi.
    """
    conn = get_connection()
    if conn:
        return conn.cursor(dictionary=True)
    return None


def close_connection(conn):
    """
    Berilgan connection va uning cursor’ini yopadi.
    """
    try:
        if conn:
            conn.close()
            print("[db] Ulanish yopildi.")
    except Error as e:
        print(f"[db] Ulanishni yopishda xato: {e}")
