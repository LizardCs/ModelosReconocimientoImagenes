# 🌿 AppIARN – Reconocimiento de Hojas de Planta con Inteligencia Artificial

Aplicación móvil para **reconocer hojas de plantas en tiempo real**, utilizando un modelo de **visión por computadora entrenado en Python** y una **APK desarrollada en Android Studio** que usa la cámara del teléfono.  
El proyecto combina **Machine Learning**, **Computer Vision** y **desarrollo móvil** para ofrecer una herramienta práctica y accesible.

---

## 📌 Descripción del Proyecto
**AppIARN** permite identificar hojas de diferentes especies mediante una fotografía tomada con la cámara del dispositivo móvil.  
El modelo de IA fue entrenado en Python y convertido a **TensorFlow Lite (TFLite)** para ejecutarse de manera eficiente en Android.

La app funciona **sin conexión a Internet**, ya que toda la inferencia se realiza localmente en el teléfono.

---

## 🧠 Tecnologías Utilizadas

### 🔹 Entrenamiento del Modelo (Python)
- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib / Seaborn  
- Transfer Learning (MobileNet / EfficientNet)  
- Jupyter Notebook o Google Colab  

### 🔹 Aplicación Android
- Android Studio  
- Kotlin / Java  
- TensorFlow Lite  
- CameraX (captura en tiempo real)  
- XML para interfaz  
- Android 8.0+  

---

## 📂 Estructura del Proyecto

📦 AppIARN
│
├── dataset/ # Imágenes de las hojas
├── model/
│ ├── modelo.ipynb # Notebook de entrenamiento del modelo
│ ├── model.h5 # Modelo entrenado en Python
│ └── model.tflite # Modelo optimizado para Android
│
├── android_app/
│ ├── app/src/main/java/ # Código fuente Android
│ ├── app/src/main/res/ # Layouts, íconos, recursos y modelos
│ └── build.gradle

