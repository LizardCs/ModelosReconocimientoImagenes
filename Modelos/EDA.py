import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
from collections import Counter

# === Configuración inicial ===
ruta_base = "C:/IARECONCIMIENTOPLANTA/data"
clases = [d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))]

# === 1. DISTRIBUCIÓN DE IMÁGENES POR CLASE ===
conteos = {clase: len(os.listdir(os.path.join(ruta_base, clase))) for clase in clases}

plt.figure(figsize=(10, 5))
sns.barplot(x=list(conteos.keys()), y=list(conteos.values()))
plt.title("Distribución de imágenes por clase")
plt.ylabel("Cantidad de imágenes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# === 2. TAMAÑO PROMEDIO Y HISTOGRAMA DE TAMAÑOS ===
anchos, altos = [], []
for clase in clases:
    for img_name in os.listdir(os.path.join(ruta_base, clase))[:100]:  # muestreo
        path = os.path.join(ruta_base, clase, img_name)
        try:
            with Image.open(path) as img:
                anchos.append(img.width)
                altos.append(img.height)
        except:
            continue

print(f"Tamaño promedio: {np.mean(anchos):.1f} x {np.mean(altos):.1f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(anchos, bins=20, kde=True)
plt.title("Distribución de anchos de imagen")
plt.xlabel("Ancho")

plt.subplot(1, 2, 2)
sns.histplot(altos, bins=20, kde=True)
plt.title("Distribución de altos de imagen")
plt.xlabel("Alto")

plt.tight_layout()
plt.show()

# === 3. EJEMPLOS VISUALES ===
n_clases = len(clases)
cols = 3
rows = (n_clases + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
axs = axs.flatten()

for i, clase in enumerate(clases):
    try:
        img_path = os.path.join(ruta_base, clase, os.listdir(os.path.join(ruta_base, clase))[0])
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].set_title(clase)
        axs[i].axis('off')
    except:
        axs[i].axis('off')

for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.suptitle("Ejemplo representativo por clase", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# === 4. DISTRIBUCIÓN DE COLORES (RGB) POR CLASE ===
medias_rgb = {clase: [] for clase in clases}

for clase in clases:
    imgs = os.listdir(os.path.join(ruta_base, clase))[:10]
    for img_name in imgs:
        path = os.path.join(ruta_base, clase, img_name)
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            medias_rgb[clase].append(np.mean(arr, axis=(0, 1)))  # media R,G,B
        except:
            continue

# Promedios RGB por clase
for clase in clases:
    if medias_rgb[clase]:
        medias_rgb[clase] = np.mean(medias_rgb[clase], axis=0)
    else:
        medias_rgb[clase] = [0, 0, 0]

# Visualización RGB
plt.figure(figsize=(10, 5))
for i, color in enumerate(['R', 'G', 'B']):
    plt.bar([c + f' ({color})' for c in clases], [medias_rgb[c][i] for c in clases], label=color)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Intensidad media")
plt.title("Distribución promedio RGB por clase")
plt.legend()
plt.tight_layout()
plt.show()

# === 5. DETECCIÓN DE IMÁGENES DUPLICADAS Y DAÑADAS ===
hashes = Counter()
errores = 0

for clase in clases:
    for img_name in os.listdir(os.path.join(ruta_base, clase)):
        path = os.path.join(ruta_base, clase, img_name)
        try:
            with Image.open(path) as img:
                hashes[hash(img.tobytes())] += 1
        except:
            errores += 1

duplicados = sum([1 for v in hashes.values() if v > 1])
print(f"Imágenes duplicadas: {duplicados}")
print(f"Imágenes dañadas o ilegibles: {errores}")

# === 6. RESUMEN POR CLASE ===
print("\n RESUMEN DE CLASES:")
total = sum(conteos.values())
for clase in clases:
    porcentaje = conteos[clase] / total * 100
    print(f"{clase:<20}: {conteos[clase]:>4} imágenes ({porcentaje:5.2f}%)")