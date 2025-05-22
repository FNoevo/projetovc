import cv2
import joblib
import numpy as np

# Carregar modelos
modelo_oculos = joblib.load("modelo_oculos.pkl")
modelo_barba = joblib.load("modelo_barba.pkl")

def estimar_cor_pele(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return (0, 0, 0)

    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    h_, w_, _ = face_roi.shape
    central = face_roi[h_//4:3*h_//4, w_//4:3*w_//4]
    mean_color = cv2.mean(central)[:3]
    return tuple(int(c) for c in mean_color)

# Iniciar webcam
cap = cv2.VideoCapture(0)
print("🟢 Webcam iniciada - Pressiona ESPAÇO para prever ou Q para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    show_text = ""

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # ESPAÇO
        img = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten().reshape(1, -1)

        pred_oculos = modelo_oculos.predict(flat)[0]
        pred_barba = modelo_barba.predict(flat)[0]

        label_oculos = "Com Óculos" if pred_oculos == 1 else "Sem Óculos"
        label_barba = "Com Barba" if pred_barba == 1 else "Sem Barba"

        cor_pele = estimar_cor_pele(frame)
        cor_pele_str = f"RGB({cor_pele[0]}, {cor_pele[1]}, {cor_pele[2]})"

        show_text = f"{label_oculos} | {label_barba} | Pele: {cor_pele_str}"
        print(f"🔎 {show_text}")

    if show_text:
        cv2.putText(frame, show_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Camadas - Óculos, Barba e Pele", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
