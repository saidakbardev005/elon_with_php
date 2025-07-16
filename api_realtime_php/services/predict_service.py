import os
import joblib
import numpy as np
import pandas as pd
import googlemaps
import warnings

from config import Config
from services.db import get_connection
from transliteration.latin_to_cyrillic import latin_to_cyrillic
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# 📁 Model fayllari
MODEL_PATH = os.path.join(PROJECT_ROOT, "kmeans_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")
PRICE_MODEL_PATH = os.path.join(PROJECT_ROOT, "price_predictor.pkl")

# Google Maps API
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

def load_db_data():
    """
    MySQL bazadan ma'lumotlarni o'qiydi va to'rt DataFrame qaytaradi.
    Raises ConnectionError agar baza bilan bog'lanilmasa.
    """
    conn = get_connection()
    if conn is None:
        raise ConnectionError("MySQL bazaga ulanishda xatolik yuz berdi.")
    try:
        df_price = pd.read_sql_query(
            """
            SELECT
              SUBSTRING_INDEX(pick_up_address, ',', 1) AS raw_from,
              SUBSTRING_INDEX(shipping_address, ',', 1)  AS raw_to,
              price
            FROM announcements
            """, con=conn
        )
        df_price["from_city"] = (
            df_price["raw_from"].astype(str)
            .map(lambda x: latin_to_cyrillic(x).strip().lower())
        )
        df_price["to_city"] = (
            df_price["raw_to"].astype(str)
            .map(lambda x: latin_to_cyrillic(x).strip().lower())
        )
        df_price.drop(columns=["raw_from", "raw_to"], inplace=True)

        drivers_df = pd.read_sql_query(
            "SELECT user_id, latitude, longitude FROM driver_locations", con=conn
        )
        my_autos = pd.read_sql_query(
            "SELECT user_id, transport_model, transport_weight, transport_volume FROM my_autos", con=conn
        )
        users = pd.read_sql_query(
            "SELECT id AS user_id, fullname, phone, status FROM users", con=conn
        )
    finally:
        conn.close()

    return df_price, drivers_df, my_autos, users


def get_coordinates(location: str):
    try:
        result = gmaps.geocode(location)
        if result:
            loc = result[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None


def load_or_initialize_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        dummy_data = np.array([[1000, 10], [2000, 20]])
        scaler = StandardScaler().fit(dummy_data)
        model = MiniBatchKMeans(n_clusters=2, batch_size=2, random_state=42)
        model.partial_fit(scaler.transform(dummy_data))
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler

model, scaler = load_or_initialize_model()


def online_fit_and_predict(weight: float, volume: float):
    global model, scaler
    X = np.array([[weight, volume]])
    X_scaled = scaler.transform(X)
    model.partial_fit(X_scaled)
    joblib.dump(model, MODEL_PATH)
    return model.predict(X_scaled)[0]


def load_or_initialize_price_model():
    if os.path.exists(PRICE_MODEL_PATH):
        price_model = joblib.load(PRICE_MODEL_PATH)
    else:
        price_model = SGDRegressor()
        price_model.partial_fit([[0, 0]], [0])
        joblib.dump(price_model, PRICE_MODEL_PATH)
    return price_model


def online_fit_and_predict_price(f_enc: int, t_enc: int, actual_price: float = None) -> int:
    model = load_or_initialize_price_model()
    X = np.array([[f_enc, t_enc]])
    if actual_price is not None:
        model.partial_fit(X, [actual_price])
        joblib.dump(model, PRICE_MODEL_PATH)
    pred = model.predict(X)[0]
    return max(0, int(pred))


def train_price_model_from_db():
    """
    Batch tarzda narx modelini o'rganadi. Agar DB bilan bog'lanmasa, xatoni qayd qilib, chiqadi.
    """
    try:
        df_price, _, _, _ = load_db_data()
    except ConnectionError as e:
        print(f"❌ Train modeli yuklanmadi, DB xatosi: {e}")
        return

    from_list = df_price["from_city"].tolist()
    to_list = df_price["to_city"].tolist()
    le = LabelEncoder().fit(from_list + to_list)

    X, y = [], []
    for _, row in df_price.iterrows():
        try:
            f_enc = int(le.transform([row["from_city"]])[0])
            t_enc = int(le.transform([row["to_city"]])[0])
            price = float(row["price"])
            if price > 0:
                X.append([f_enc, t_enc])
                y.append(price)
        except Exception:
            continue

    if not X:
        print("❌ Narx modeli uchun ma'lumot topilmadi.")
        return

    price_model = SGDRegressor()
    price_model.partial_fit(X, y)
    joblib.dump(price_model, PRICE_MODEL_PATH)
    print(f"✅ Narx modeli {len(X)} namunada o‘qitildi.")


def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    """
    Eng mos haydovchilarni topadi:
    - Yo'nalish bo'yicha klasterlash
    - Uzunlik/engi bo'yicha masofa hisoblash
    """
    try:
        _, drivers_df, my_autos, users = load_db_data()
    except ConnectionError as e:
        print(f"❌ Haydovchilarni yuklab bo'lmadi, DB xatosi: {e}")
        return []

    # Matn tipidagi ustunlarni float tipiga aylantirish
    drivers_df['latitude'] = pd.to_numeric(drivers_df['latitude'], errors='coerce')
    drivers_df['longitude'] = pd.to_numeric(drivers_df['longitude'], errors='coerce')
    my_autos['transport_weight'] = pd.to_numeric(my_autos['transport_weight'], errors='coerce')
    my_autos['transport_volume'] = pd.to_numeric(my_autos['transport_volume'], errors='coerce')

    # Foydalanuvchi ma'lumotlarini birlashtirish va nomalardagi NaN qatorlarni tashlash
    drivers = (
        drivers_df
        .merge(my_autos, on='user_id')
        .merge(users[['user_id', 'fullname', 'phone', 'status']], on='user_id')
        .dropna(subset=['transport_weight', 'transport_volume', 'latitude', 'longitude'])
        .copy()
    )

    arr = drivers[['transport_weight', 'transport_volume']].values
    if arr.size == 0:
        return []

    scaler_local = StandardScaler().fit(arr)
    arr_s = scaler_local.transform(arr)

    km = MiniBatchKMeans(n_clusters=min(4, len(arr_s)), batch_size=6, random_state=42)
    km.partial_fit(arr_s)
    inp_s = scaler_local.transform([[weight, volume]])
    cid = int(km.predict(inp_s)[0])

    drivers['cluster_id'] = km.predict(arr_s)
    same = drivers[drivers['cluster_id'] == cid].copy()
    if same.empty:
        return []

    same['capacity_distance'] = np.sqrt(
        (same['transport_weight'] - weight) ** 2 +
        (same['transport_volume'] - volume) ** 2
    )
    same['distance_km'] = (
        np.sqrt(
            (same['latitude'] - lat) ** 2 +
            (same['longitude'] - lon) ** 2
        ) * 111
    )

    return (
        same
        .sort_values(by=['capacity_distance', 'distance_km'], ascending=[True, True])
        .head(5)[[
            'fullname', 'phone', 'transport_model',
            'transport_weight', 'transport_volume', 'distance_km'
        ]]
        .to_dict(orient='records')
    )
