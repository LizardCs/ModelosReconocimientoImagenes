package com.example.appiarn

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import androidx.media3.common.util.Log
import androidx.media3.common.util.UnstableApi
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivityres : Activity() {

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var captureButton: Button
    private lateinit var analyzeButton: Button
    private lateinit var pickButton: Button
    private lateinit var testAssetButton: Button
    private val REQUEST_IMAGE_CAPTURE = 1
    private val REQUEST_IMAGE_PICK = 2
    private val PERMISSION_REQUEST_READ_EXTERNAL_STORAGE = 100

    private lateinit var interpreter: Interpreter

    @UnstableApi
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.capturedImageView)
        resultTextView = findViewById(R.id.resultTextView)
        captureButton = findViewById(R.id.captureButton)
        analyzeButton = findViewById(R.id.analyzeButton)
        pickButton = findViewById(R.id.pickButton)
        testAssetButton = findViewById(R.id.testAssetButton)

        initInterpreter()

        testAssetButton.setOnClickListener {
            listarAssets()
            try {
                val bitmap = loadBitmapFromAssets("image_1.JPG")
                if (bitmap != null) {
                    imageView.setImageBitmap(bitmap)
                    analyzeImage(bitmap)
                } else {
                    resultTextView.text = "❌ No se pudo cargar la imagen desde assets"
                }
            } catch (e: Exception) {
                resultTextView.text = "❌ Error: ${e.message}"
            }
        }

        captureButton.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }

        pickButton.setOnClickListener {
            val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                Manifest.permission.READ_MEDIA_IMAGES
            } else {
                Manifest.permission.READ_EXTERNAL_STORAGE
            }

            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(permission), PERMISSION_REQUEST_READ_EXTERNAL_STORAGE)
            } else {
                openGallery()
            }
        }

        analyzeButton.setOnClickListener {
            val bitmap = imageView.drawable?.toBitmap()
            if (bitmap != null) {
                analyzeImage(bitmap)
            } else {
                resultTextView.text = "Primero captura o selecciona una imagen"
            }
        }
    }

    private fun openGallery() {
        val pickPhotoIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(pickPhotoIntent, REQUEST_IMAGE_PICK)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd("modelotomate_efficientnetb0.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun initInterpreter() {
        val model = loadModelFile()
        interpreter = Interpreter(model)
    }

    @OptIn(UnstableApi::class)
    private fun loadBitmapFromAssets(filename: String): Bitmap? {
        return try {
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val inputStream = assets.open(filename)
            val bitmap = BitmapFactory.decodeStream(inputStream, null, options)
            inputStream.close()
            bitmap
        } catch (e: Exception) {
            Log.e("CargaAsset", "No se pudo cargar el bitmap: ${e.message}", e)
            null
        }
    }

    @OptIn(UnstableApi::class)
    private fun listarAssets() {
        try {
            val archivos = assets.list("") ?: arrayOf()
            Log.d("AssetsDebug", "Contenido en /assets/:")
            for (archivo in archivos) {
                Log.d("AssetsDebug", archivo)
            }
        } catch (e: Exception) {
            Log.e("AssetsDebug", "Error al listar assets: ${e.message}", e)
        }
    }

    @OptIn(UnstableApi::class)
    private fun analyzeImage(bitmap: Bitmap) {
        try {
            // Redimensionar imagen a 224x224 (como lo requiere EfficientNetB0)
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

            // Preprocesar imagen como RGB puro, float32, normalizado a [0, 1]
            val floatArray = FloatArray(224 * 224 * 3)
            var index = 0
            for (y in 0 until 224) {
                for (x in 0 until 224) {
                    val pixel = resizedBitmap.getPixel(x, y)

                    val r = ((pixel shr 16) and 0xFF) / 255.0f
                    val g = ((pixel shr 8) and 0xFF) / 255.0f
                    val b = (pixel and 0xFF) / 255.0f

                    floatArray[index++] = r
                    floatArray[index++] = g
                    floatArray[index++] = b
                }
            }

            // Preparar input para el modelo
            val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            inputBuffer.loadArray(floatArray)

            // Crear buffer de salida
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32)
            interpreter.run(inputBuffer.buffer, outputBuffer.buffer.rewind())

            val outputArray = outputBuffer.floatArray

            // Mostrar todas las probabilidades
            val fullOutput = outputArray.mapIndexed { i, v -> "[$i] %.4f".format(v) }.joinToString("\n")
            Log.d("OutputProbabilidades", fullOutput)

            // Mostrar valor específico de 'Tomate__saludable' (índice 5)
            Log.d("SaludableScore", "Probabilidad clase 5 (Tomate__saludable): ${outputArray[5]}")

            // Leer etiquetas
            val labels = assets.open("labels.txt").bufferedReader().readLines()

            // Top 3 predicciones
            val predictions = outputArray.mapIndexed { i, score -> i to score }
                .sortedByDescending { it.second }
                .take(3)

            val resultText = StringBuilder("Top 3 resultados:\n")
            for ((i, score) in predictions) {
                val label = labels.getOrNull(i) ?: "Desconocido"
                resultText.append("• $label: %.2f%%\n".format(score * 100))
            }

            resultTextView.text = resultText.toString()

        } catch (e: Exception) {
            Log.e("Classifier", "Error al analizar imagen: ${e.message}", e)
            resultTextView.text = "Error al analizar imagen: ${e.message}"
        }
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    val imageBitmap = data?.extras?.get("data") as? Bitmap
                    if (imageBitmap != null) {
                        imageView.setImageBitmap(imageBitmap)
                        resultTextView.text = "Imagen capturada. Presiona 'Analizar'."
                    } else {
                        resultTextView.text = "Error al obtener la foto."
                    }
                }
                REQUEST_IMAGE_PICK -> {
                    val imageUri: Uri? = data?.data
                    if (imageUri != null) {
                        val inputStream = contentResolver.openInputStream(imageUri)
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        inputStream?.close()
                        if (bitmap != null) {
                            imageView.setImageBitmap(bitmap)
                            resultTextView.text = "Imagen seleccionada. Presiona 'Analizar'."
                        } else {
                            resultTextView.text = "No se pudo cargar la imagen seleccionada."
                        }
                    } else {
                        resultTextView.text = "No se seleccionó ninguna imagen."
                    }
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_READ_EXTERNAL_STORAGE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                resultTextView.text = "Permiso para acceder a la galería denegado"
            }
        }
    }
}
