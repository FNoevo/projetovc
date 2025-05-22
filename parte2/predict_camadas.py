import cv2
import numpy as np
import joblib

# Carregar modelos treinados
modelo_barba = joblib.load("modelo_barba.pkl")
modelo_genero = joblib.load("modelo_genero.pkl")
modelo_oculos = joblib.load("modelo_oculos.pkl")

# Iniciar webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Não foi possível aceder à câmara.")
    exit()

print("📷 Webcam ativa. Pressiona 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Erro ao capturar imagem.")
        break

    # Pré-processar imagem
    img = cv2.resize(frame, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    flat = gray.flatten().reshape(1, -1)

    # Previsões
    genero = modelo_genero.predict(flat)[0]
    barba = modelo_barba.predict(flat)[0]
    oculos = modelo_oculos.predict(flat)[0]

    # Texto de previsão
    genero_str = "Homem" if genero == 1 else "Mulher"
    barba_str = "Com barba" if barba == 1 else "Sem barba"
    oculos_str = "Com óculos" if oculos == 1 else "Sem óculos"

    # Criar caixa preta semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Adicionar texto
    cv2.putText(frame, f"Género: {genero_str}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Barba: {barba_str}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Óculos: {oculos_str}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Mostrar resultado
    cv2.imshow("Previsão em tempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
