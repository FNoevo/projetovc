import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Preparação dos dados (igual)
base_dir = "data"
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
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img_gray.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    n_jobs=-1,          # usar todos os núcleos do CPU
    verbose=1
)

grid.fit(X_train, y_train)

# Melhor modelo
best_model = grid.best_estimator_

# Avaliação
pred = best_model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"🌟 Melhor modelo: {grid.best_params_}")
print(f"✅ Accuracy com melhores parâmetros: {acc:.2%}")

# Guardar modelo
joblib.dump(best_model, "modelo_genero_rf_grid.pkl")
print("💾 Modelo guardado como 'modelo_genero_rf_grid.pkl'")
