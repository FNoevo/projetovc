import cv2
import numpy as np
import os

image_path = "imagens/ponteee.jpg"
output_path = "resultados/geom_logic.jpg"

img = cv2.imread(image_path)

# Redimensionar
resized = cv2.resize(img, (300, 300))

# Rotacionar
rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

# Operação lógica: NOT
not_img = cv2.bitwise_not(rotated)

os.makedirs("../resultados", exist_ok=True)
cv2.imwrite(output_path, not_img)
