import cv2
import joblib
import numpy as np

# Carregar modelo
model = joblib.load("modelo_genero.pkl")

# Caminho da imagem nova a prever
image_path = "exemplo.jpeg"  # muda para a imagem que quiseres testar

# Ler imagem
img = cv2.imread(image_path)
if img is None:
    print("Erro: imagem não encontrada.")
    exit()

# Pré-processamento
img = cv2.resize(img, (64, 64))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flat = gray.flatten().reshape(1, -1)

# Previsão
pred = model.predict(flat)[0]
prob = model.predict_proba(flat)[0]

label = "Homem" if pred == 0 else "Mulher"
conf = max(prob) * 100

print(f"✅ Resultado: {label} ({conf:.2f}% de confiança)")
