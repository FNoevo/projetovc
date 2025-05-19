import cv2
import joblib
import numpy as np

# Carregar modelo
model = joblib.load("modelo_genero.pkl")

# Iniciar webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível aceder à câmara.")
    exit()

print("🟢 Câmara iniciada. Pressiona [ESPAÇO] para prever ou [Q] para sair.")

# Variável para guardar o último resultado
last_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da câmara.")
        break

    # Mostrar a última previsão na imagem
    if last_prediction:
        cv2.putText(frame, last_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar frame com texto
    cv2.imshow("Webcam - Pressiona [ESPAÇO] para prever ou [Q] para sair", frame)

    key = cv2.waitKey(1) & 0xFF

    # Se pressionar ESPAÇO, faz previsão
    if key == 32:  # Tecla ESPAÇO
        img = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten().reshape(1, -1)

        pred = model.predict(flat)[0]
        prob = model.predict_proba(flat)[0]

        label = "Homem" if pred == 0 else "Mulher"
        conf = max(prob) * 100
        last_prediction = f"{label} ({conf:.2f}%)"

        print(f"✅ Resultado: {last_prediction}")

    # Se pressionar Q, sai
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
