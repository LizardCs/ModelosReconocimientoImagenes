import tensorflow as tf

# Cargar modelo
interpreter = tf.lite.Interpreter(model_path="D:/IARECONCIMIENTOPLANMTA/plantamodelo.tflite")
interpreter.allocate_tensors()

# Mostrar detalles del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== ENTRADAS ===")
for i, input_tensor in enumerate(input_details):
    print(f"[{i}] Nombre: {input_tensor['name']}")
    print(f"    Forma esperada: {input_tensor['shape']}")
    print(f"    Tipo de dato: {input_tensor['dtype']}")

print("\n=== SALIDAS ===")
for i, output_tensor in enumerate(output_details):
    print(f"[{i}] Nombre: {output_tensor['name']}")
    print(f"    Forma esperada: {output_tensor['shape']}")
    print(f"    Tipo de dato: {output_tensor['dtype']}")

# Leer labels.txt si existe
