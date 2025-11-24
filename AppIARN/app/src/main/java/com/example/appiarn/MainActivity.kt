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
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : Activity() {

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
        //testAssetButton = findViewById(R.id.testAssetButton)

        initInterpreter()

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

    /* Permisos para abrir la galeria*/
    private fun openGallery() {
        val pickPhotoIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(pickPhotoIntent, REQUEST_IMAGE_PICK)
    }

    /*Carga de modelo*/
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd("modelotomate_efficientnetb0.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @UnstableApi
    private fun initInterpreter() {
        try {
            val model = loadModelFile()

            val options = Interpreter.Options().apply {
                setNumThreads(2) // No mas de 2 si es un celular de gama media
            }

            interpreter = Interpreter(model, options)
        } catch (e: Exception) {
            Log.e("InterpreterInit", "Error al inicializar el modelo: ${e.message}", e)
            resultTextView.text = "Error al carga el modelo: ${e.message}"
        }
    }

    /*Cargar bits de imagenes*/
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

    /*Procesar imagenes para aplicar el modelo*/
    @OptIn(UnstableApi::class)
    private fun analyzeImage(bitmap: Bitmap) {
        try {
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0.5f, 0.5f)) // EfficientNet
                .build()
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            val output = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32)
            interpreter.run(tensorImage.buffer, output.buffer.rewind())
            val confidences = output.floatArray
            val labels = assets.open("labels.txt").bufferedReader().readLines()
            val top3 = confidences.mapIndexed { index, score -> index to score }
                .sortedByDescending { it.second }
                .take(3)
            val highestConfidence = top3.firstOrNull()?.second ?: 0f
            val resultText = StringBuilder()
            if (highestConfidence < 0.80f) {
                resultText.append("Planta no analizada en el modelo.")
            } else {
                val (index, score) = top3[0]
                val fullLabel = labels.getOrNull(index) ?: "Desconocido"
                val simplifiedLabel = fullLabel.substringAfter("__").replace('_', ' ').capitalize()
                resultText.append("$simplifiedLabel (%.2f%%)\n".format(score * 100))
                resultText.append("Recomendaciones:\n")
                resultText.append(getRecomendaciones(simplifiedLabel))
            }
            resultTextView.text = resultText.toString()
        } catch (e: Exception) {
            Log.e("Classifier", "Error: ${e.message}", e)
            resultTextView.text = "Error al analizar imagen: ${e.message}"
        }
    }

    private fun getRecomendaciones(label: String): String {
        return when (label.lowercase()) {
            "acaros" -> "• Aplicar acaricidas naturales o químicos.\n• Revisar humedad ambiental.\n• Rotar cultivos para evitar proliferación."
            "hoja moho" -> "• Evitar exceso de riego.\n• Mejorar ventilación entre plantas.\n• Usar fungicidas preventivos."
            "mancha bacteriana" -> "• Retirar hojas afectadas.\n• Desinfectar herramientas.\n• Aplicar cobre como tratamiento."
            "mancha diana" -> "• Usar fungicidas específicos.\n• Evitar salpicaduras de agua.\n• Eliminar residuos vegetales infectados."
            "mancha septoria foliar" -> "• Aplicar fungicidas a base de clorotalonil.\n• Evitar riego por aspersión.\n• Practicar rotación de cultivos."
            "saludable" -> "• Mantener monitoreo constante.\n• Aplicar fertilización equilibrada.\n• Vigilar plagas regularmente."
            "tizon tardio" -> "• Eliminar hojas afectadas inmediatamente.\n• Aplicar fungicidas sistémicos.\n• Evitar exceso de humedad."
            "tizon temprano" -> "• Evitar monocultivo.\n• Aplicar tratamiento con mancozeb.\n• Usar variedades resistentes."
            "virus mosaico" -> "• Eliminar plantas afectadas.\n• Controlar insectos vectores.\n• Desinfectar regularmente herramientas."
            "virus rizado amarillo" -> "• Controlar mosca blanca.\n• Usar barreras físicas.\n• Aplicar jabón potásico o aceites minerales."
            else -> "• No se encontraron recomendaciones específicas."
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
