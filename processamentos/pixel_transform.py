import cv2
import os

input_folder = "../imagens"
output_folder = "resultados/pixel_transform"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        inverted = 255 - img  # negativo
        bright = cv2.convertScaleAbs(img, alpha=1, beta=50)

        name = filename.rsplit('.', 1)[0]
        cv2.imwrite(f"{output_folder}/{name}_inverted.jpg", inverted)
        cv2.imwrite(f"{output_folder}/{name}_bright.jpg", bright)
