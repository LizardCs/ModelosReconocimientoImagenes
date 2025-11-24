import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURACIÓN ===
ruta_base = "C:/IARECONCIMIENTOPLANTA/data"
ruta_balanceada = "C:/IARECONCIMIENTOPLANTA/data_balanceada"
img_exts = [".jpg", ".jpeg", ".png"]

# Crear carpeta de destino
if os.path.exists(ruta_balanceada):
    shutil.rmtree(ruta_balanceada)
os.makedirs(ruta_balanceada, exist_ok=True)

# === OBTENER CLASES Y CONTEOS ===
conteos = {}
for clase in os.listdir(ruta_base):
    carpeta_clase = os.path.join(ruta_base, clase)
    if os.path.isdir(carpeta_clase):
        imagenes = [img for img in os.listdir(carpeta_clase) if os.path.splitext(img)[-1].lower() in img_exts]
        conteos[clase] = len(imagenes)

# Mostrar distribución original
plt.figure(figsize=(8, 4))
sns.barplot(x=list(conteos.keys()), y=list(conteos.values()))
plt.title("Distribución original de imágenes por clase")
plt.ylabel("Cantidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === BALANCEO DE CLASES ===
max_clase = max(conteos.values())

def aumentar_imagen(imagen):
    # Aumentos simples: rotación, brillo, volteo
    if random.random() < 0.5:
        imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        imagen = imagen.rotate(random.randint(-20, 20))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(random.uniform(0.8, 1.2))
    return imagen

for clase, cantidad in conteos.items():
    origen = os.path.join(ruta_base, clase)
    destino = os.path.join(ruta_balanceada, clase)
    os.makedirs(destino, exist_ok=True)

    imagenes = [img for img in os.listdir(origen) if os.path.splitext(img)[-1].lower() in img_exts]

    # Copiar originales
    for img in imagenes:
        shutil.copy(os.path.join(origen, img), os.path.join(destino, img))

    # Generar más si hay menos que el máximo
    faltan = max_clase - cantidad
    for i in range(faltan):
        img_src = random.choice(imagenes)
        img_path = os.path.join(origen, img_src)
        with Image.open(img_path) as img:
            img_aug = aumentar_imagen(img)
            nuevo_nombre = f"aug_{i}_{img_src}"
            img_aug.save(os.path.join(destino, nuevo_nombre))

# === MOSTRAR DISTRIBUCIÓN FINAL ===
conteos_bal = {clase: len(os.listdir(os.path.join(ruta_balanceada, clase))) for clase in conteos.keys()}

plt.figure(figsize=(8, 4))
sns.barplot(x=list(conteos_bal.keys()), y=list(conteos_bal.values()))
plt.title("Distribución balanceada de imágenes por clase")
plt.ylabel("Cantidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()