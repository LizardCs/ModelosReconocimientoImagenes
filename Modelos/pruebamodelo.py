import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import os

# === RUTAS ===
modelo_tflite_path = "C:/IARECONCIMIENTOPLANTA/modelotomate_efficientnetb0.tflite"
labels_path = "C:/IARECONCIMIENTOPLANTA/labels.txt"
img_path = "C:/IARECONCIMIENTOPLANTA/data/Tomate__saludable/image (1).jpg"

# === CARGAR LABELS ===
with open(labels_path, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f]

# === CARGAR MODELO TFLITE ===
interpreter = tf.lite.Interpreter(model_path=modelo_tflite_path)
interpreter.allocate_tensors()

# === DETALLES DE ENTRADA/SALIDA ===
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === PREPROCESAR IMAGEN ===
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)  # Usa el correcto de EfficientNet
input_data = np.expand_dims(img_array, axis=0).astype(np.float32)

# === INFERENCIA ===
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# === MOSTRAR TOP 3 ===
top3_indices = np.argsort(output)[-3:][::-1]
print("Top 3 resultados:")
for i in top3_indices:
    label = labels[i] if i < len(labels) else f"Etiqueta desconocida {i}"
    score = output[i]
    print(f"{label}: {score:.2%}")
