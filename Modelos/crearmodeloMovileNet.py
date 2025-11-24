import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# === CONFIGURACIÓN ===
ruta_base = "C:/IARECONCIMIENTOPLANTA/data"
modelo_salida = "C:/IARECONCIMIENTOPLANTA/modelotomate_mobilenetv2.h5"
img_size = (224, 224)
batch_size = 32
epochs = 10

# === AUMENTO Y PREPROCESAMIENTO ===
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    ruta_base,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    ruta_base,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# === MODELO BASE ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model.trainable = False  # Congelar capas base

# === CAPAS PERSONALIZADAS ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === COMPILACIÓN Y ENTRENAMIENTO ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(modelo_salida, save_best_only=True, verbose=1)

model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint])
