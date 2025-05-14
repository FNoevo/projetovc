import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import os
import cv2

# Carregar modelo
model = joblib.load("parte2/modelo_genero.pkl")

# Repetir preparação dos dados (igual ao treino)
base_dir = "parte2/data"
categories = ["men", "women"]

X = []
y = []

for label, category in enumerate(categories):
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(gray.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# Separar conjunto de teste
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prever
y_pred = model.predict(X_test)

# Avaliação
print("✅ Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=categories))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()
