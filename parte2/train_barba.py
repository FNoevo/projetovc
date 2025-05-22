import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

IMG_DIR = "dataceleba/archive/img_align_celeba"
ATTR_FILE = "dataceleba/archive/list_attr_celeba.csv"


df = pd.read_csv(ATTR_FILE)
df = df[:3000]  # limitar para testes rápidos

X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_DIR, row["image_id"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X.append(gray.flatten())

    # inverter: 1 = com barba
    y.append(1 if row["No_Beard"] == 0 else 0)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"🧔 Acurácia barba: {acc:.2%}")

joblib.dump(model, "modelo_barba.pkl")
print("💾 Modelo guardado como modelo_barba.pkl")
