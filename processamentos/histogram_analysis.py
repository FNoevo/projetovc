import cv2
import matplotlib.pyplot as plt
import os

input_folder = "imagens"
output_folder = "resultados/histograms"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path, 0)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        name = filename.rsplit('.', 1)[0]
        plt.figure()
        plt.title(f"Histograma de {filename}")
        plt.xlabel("Intensidade")
        plt.ylabel("Frequência")
        plt.plot(hist)
        plt.savefig(f"{output_folder}/{name}_hist.png")
        plt.close()
