import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input

# === RUTAS ===
modelo_tflite_path = "C:/Proyectos/IA/IARECONCIMIENTOPLANTA/modelotomate_efficientnetb0.tflite"
labels_path = "C:/Proyectos/IA/IARECONCIMIENTOPLANTA/labels.txt"
img_path = "C:/Proyectos/IA/IARECONCIMIENTOPLANTA/data/Tomate__virus_mosaico/image (1).jpg"

# === CARGAR LABELS DESDE labels.txt ===
with open(labels_path, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# === CARGAR MODELO TFLITE ===
interpreter = tf.lite.Interpreter(model_path=modelo_tflite_path)
interpreter.allocate_tensors()

# === DETALLES DE ENTRADA/SALIDA ===
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === PREPROCESAR IMAGEN ===
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# === HACER PREDICCIÓN ===
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
pred_index = np.argmax(output_data)
pred_class = labels[pred_index]

print(f"✅ Clase predicha: {pred_class} (índice: {pred_index})")
