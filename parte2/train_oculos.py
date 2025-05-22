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
df = df[:1000]  # limitar para testes rápidos

X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    nome_img = row["image_id"]

    # Corrigir caminho completo
    img_path = os.path.join("data", "dataceleba", "archive", "img_align_celeba", nome_img)
    img_path = os.path.normpath(img_path)

    if not os.path.exists(img_path):
        print(f"❌ Imagem não encontrada: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Falha ao ler imagem: {img_path}")
        continue

    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X.append(gray.flatten())
    y.append(row["Eyeglasses"])


X = np.array(X)
y = np.array(y)

print(f"🔍 Total de imagens carregadas: {len(X)}")
print(f"🔍 Total de labels carregadas: {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"👓 Acurácia óculos: {acc:.2%}")

joblib.dump(model, "modelo_oculos.pkl")
print("💾 Modelo guardado como modelo_oculos.pkl")
