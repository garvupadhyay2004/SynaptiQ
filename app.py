"""
SynaptiQ - FastAPI Backend
Models:
  - ImageGuard : adult_content_detector_new.keras
  - AutoMatch  : KNN on Automobile_dataset.csv
  - PredictIQ  : Random Forest on logistics_data.csv
"""
  
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle, os, io

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI(title="SynaptiQ API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── MODEL 1: ImageGuard ──
print("Loading ImageGuard model...")
try:
    image_model = tf.keras.models.load_model("adult_content_detector_new.keras", compile=False)
    print("✅ ImageGuard model loaded!")
except Exception as e:
    print(f"⚠️ ImageGuard model failed to load: {e}")
    image_model = None

# ── MODEL 2: PredictIQ ──
print("Loading PredictIQ model...")
rf_model = None
rf_feature_cols = []
if os.path.exists("rf_model.pkl") and os.path.exists("rf_feature_cols.pkl"):
    with open("rf_model.pkl","rb") as f: rf_model = pickle.load(f)
    with open("rf_feature_cols.pkl","rb") as f: rf_feature_cols = pickle.load(f)
    print("✅ PredictIQ model loaded from pickle!")
elif os.path.exists("logistics_data.csv"):
    df_log = pd.read_csv("logistics_data.csv")
    df_log["date"] = pd.to_datetime(df_log["date"])
    df_log["day"] = df_log["date"].dt.day
    df_log["month"] = df_log["date"].dt.month
    df_log["year"] = df_log["date"].dt.year
    df_log = df_log.drop(columns=["date"])
    df_log = pd.get_dummies(df_log, columns=["warehouse_id","region"], drop_first=True)
    y = df_log["orders"]
    X = df_log.drop(columns=["orders","workers"])
    rf_feature_cols = list(X.columns)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    with open("rf_model.pkl","wb") as f: pickle.dump(rf_model,f)
    with open("rf_feature_cols.pkl","wb") as f: pickle.dump(rf_feature_cols,f)
    print("✅ PredictIQ model trained and saved!")

# ── MODEL 3: AutoMatch KNN ──
print("Loading AutoMatch KNN model...")
car_knn = None; car_scaler = None; car_df = None
CAR_FEATURES = ['Engine HP','Engine Cylinders','highway MPG','city mpg','MSRP']

if os.path.exists("knn_car_model.pkl") and os.path.exists("knn_car_scaler.pkl") and os.path.exists("knn_car_df.pkl"):
    with open("knn_car_model.pkl","rb") as f: car_knn = pickle.load(f)
    with open("knn_car_scaler.pkl","rb") as f: car_scaler = pickle.load(f)
    with open("knn_car_df.pkl","rb") as f: car_df = pickle.load(f)
    print("✅ AutoMatch KNN loaded from pickle!")
elif os.path.exists("Automobile_dataset.csv"):
    car_df = pd.read_csv("Automobile_dataset.csv")
    car_df = car_df.dropna(subset=CAR_FEATURES).reset_index(drop=True)
    car_scaler = StandardScaler()
    data_scaled = car_scaler.fit_transform(car_df[CAR_FEATURES])
    car_knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    car_knn.fit(data_scaled)
    with open("knn_car_model.pkl","wb") as f: pickle.dump(car_knn,f)
    with open("knn_car_scaler.pkl","wb") as f: pickle.dump(car_scaler,f)
    with open("knn_car_df.pkl","wb") as f: pickle.dump(car_df,f)
    print("✅ AutoMatch KNN trained and saved!")
else:
    print("⚠️ Automobile_dataset.csv not found.")

# ── ENDPOINTS ──
@app.get("/")
def root():
    return {"status":"SynaptiQ API running","models":{"ImageGuard":"loaded" if image_model else "not loaded","PredictIQ":"loaded" if rf_model else "not loaded","AutoMatch":"loaded (KNN)" if car_knn else "not loaded"}}

@app.post("/predict/image")
async def predict_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224,224))
        img_array = np.expand_dims(keras_image.img_to_array(img)/255.0, axis=0)
        non_adult_prob = float(image_model.predict(img_array)[0][0])
        adult_prob = 1 - non_adult_prob
        label = "Adult Content" if adult_prob >= 0.5 else "Non-Adult Content"
        confidence = adult_prob if adult_prob >= 0.5 else non_adult_prob
        return {"label":label,"safe_prob":round(non_adult_prob,4),"unsafe_prob":round(adult_prob,4),"confidence":round(confidence,4)}
    except Exception as e:
        return {"error":str(e)}

class CarInput(BaseModel):
    hp: float
    cylinders: float
    highway_mpg: float
    city_mpg: float
    budget: float

@app.post("/recommend/car")
def recommend_car(data: CarInput):
    try:
        if car_knn is None:
            return {"error":"AutoMatch not loaded. Add Automobile_dataset.csv"}
        user_input = np.array([[data.hp, data.cylinders, data.highway_mpg, data.city_mpg, data.budget]])
        user_scaled = car_scaler.transform(user_input)
        distances, indices = car_knn.kneighbors(user_scaled)
        recs = car_df.iloc[indices[0]].copy()
        recs['sim'] = distances[0]
        filtered = recs[recs['MSRP'] <= data.budget]
        if filtered.empty:
            filtered = recs
        filtered = filtered.sort_values("sim").head(3)
        result = []
        for _, row in filtered.iterrows():
            match_pct = min(99, max(0, round((1 - row['sim']/10)*100, 1)))
            result.append({
                "name": f"{row.get('Make','?')} {row.get('Model','?')}",
                "year": int(row.get('Year',0)),
                "engine": f"{row.get('Engine Fuel Type','N/A')} | {int(row.get('Engine HP',0))} HP",
                "mileage": f"City: {int(row.get('city mpg',0))} | Hwy: {int(row.get('highway MPG',0))} MPG",
                "price": f"${int(row.get('MSRP',0)):,}",
                "match_score": match_pct
            })
        return {"cars":result,"source":"knn_model"}
    except Exception as e:
        return {"error":str(e)}

class PredictInput(BaseModel):
    month: int; day: int; year: int
    shipment_weight: float; processing_time: float
    scanner_used: int; warehouse_id: str; region: str

@app.post("/predict/analytics")
def predict_analytics(data: PredictInput):
    try:
        if rf_model is None:
            return {"error":"PredictIQ not loaded."}
        input_dict = {col:0 for col in rf_feature_cols}
        input_dict["day"] = data.day; input_dict["month"] = data.month; input_dict["year"] = data.year
        input_dict["shipment_weight"] = data.shipment_weight; input_dict["processing_time"] = data.processing_time
        input_dict["scanner_used"] = data.scanner_used
        if data.warehouse_id=="WH_2" and "warehouse_id_WH_2" in input_dict: input_dict["warehouse_id_WH_2"]=1
        elif data.warehouse_id=="WH_3" and "warehouse_id_WH_3" in input_dict: input_dict["warehouse_id_WH_3"]=1
        rc = f"region_{data.region}"
        if rc in input_dict: input_dict[rc]=1
        predicted_orders = float(rf_model.predict(pd.DataFrame([input_dict]))[0])
        return {"predicted_orders":round(predicted_orders,2),"estimated_workers_needed":int(predicted_orders/12),"month":data.month,"warehouse":data.warehouse_id,"region":data.region}
    except Exception as e:
        return {"error":str(e)}