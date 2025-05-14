import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Caminhos
base_dir = "data"
categories = ["men", "women"]

# Preparação dos dados
X = []
y = []

for label, category in enumerate(categories):
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))       # Redimensiona
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Cinzento
        X.append(img_gray.flatten())          # Flatten
        y.append(label)

X = np.array(X)
y = np.array(y)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo (SVM)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Avaliação simples
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"✅ Accuracy: {acc:.2%}")

# Guardar modelo
joblib.dump(model, "modelo_genero.pkl")
