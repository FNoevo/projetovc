import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Caminhos
IMG_DIR = "dataceleba/archive/img_align_celeba"
ATTR_FILE = "dataceleba/archive/list_attr_celeba.csv"

# ⚠️ Usar sep='\s+' em vez de delim_whitespace
df = pd.read_csv(ATTR_FILE, sep='\s+', skiprows=1)

# Criar coluna 'image_id' com os nomes reais das imagens
df["image_id"] = df.index

# Manter apenas imagens de 000001.jpg a 001000.jpg (que existem no teu disco)
df = df[df["image_id"].isin([f"{i:06d}.jpg" for i in range(1, 1001)])]

X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    nome_img = row["image_id"]
    img_path = os.path.normpath(os.path.join(IMG_DIR, nome_img))

    if not os.path.exists(img_path):
        print(f"❌ Imagem não encontrada: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Erro ao ler imagem: {img_path}")
        continue

    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    X.append(gray.flatten())

    y.append(1 if row["Male"] == 1 else 0)

X = np.array(X)
y = np.array(y)

print(f"✅ Total de imagens processadas: {len(X)}")

if len(X) == 0:
    print("❌ Nenhuma imagem válida foi carregada. Verifica os caminhos.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"👨‍🦰 Acurácia género: {acc:.2%}")

joblib.dump(model, "modelo_genero.pkl")
print("💾 Modelo guardado como modelo_genero.pkl")
