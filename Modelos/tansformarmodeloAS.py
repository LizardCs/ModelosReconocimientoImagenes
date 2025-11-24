import tensorflow as tf
import tensorflow_hub as hub  # IMPORTANTE para capas KerasLayer

# Ruta base
ruta_modelo = r'D:\IARECONCIMIENTOPLANMTA'

# Cargar el modelo .h5 con el parámetro custom_objects
modelo = tf.keras.models.load_model(
    ruta_modelo + r'\plantamodelo.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)

# Convertir a TFLite
conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = conversor.convert()

# Guardar el modelo .tflite en la misma carpeta
ruta_guardado = ruta_modelo + r'\plantamodelo.tflite'
with open(ruta_guardado, 'wb') as f:
    f.write(modelo_tflite)

print("✅ Conversión completada: guardado en", ruta_guardado)
