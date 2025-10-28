package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class CtcOutputs(
	val logProbs: Array<Array<FloatArray>>, // [1, T, num_ctc]
	val catLogits: Array<Array<FloatArray>>? // [1, T, num_cat] or null
)

class CtcModelRunner(
    private val context: Context,
    private val tflitePath: String,
    private val metadataPath: String,
    private val preferGpu: Boolean = false
) {
	companion object {
		private const val TAG = "CtcModelRunner"
	}

	private var interpreter: Interpreter? = null
	private var gpuDelegate: GpuDelegate? = null
	lateinit var meta: CtcModelMetadata

	init {
		load()
	}

	private fun load() {
		meta = ModelMetadataLoader.loadFromAssets(context, metadataPath)
		val modelBuffer = loadModelFile(tflitePath)
        val options = Interpreter.Options()
        if (preferGpu) {
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU delegate enabled for CTC model")
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable; falling back to CPU", t)
                options.setNumThreads(4)
            }
        } else {
            options.setNumThreads(4)
        }
		interpreter = Interpreter(modelBuffer, options)
		Log.i(TAG, "CTC model loaded: $tflitePath with meta: $metadataPath")
		// Log I/O shapes
		val inShape = interpreter!!.getInputTensor(0).shape()
		Log.d(TAG, "Input shape: ${inShape.contentToString()}")
		val out0 = interpreter!!.getOutputTensor(0).shape()
		Log.d(TAG, "Output[0] shape: ${out0.contentToString()}")
		if (interpreter!!.outputTensorCount > 1) {
			val out1 = interpreter!!.getOutputTensor(1).shape()
			Log.d(TAG, "Output[1] shape: ${out1.contentToString()}")
		}
	}

	private fun loadModelFile(assetPath: String): MappedByteBuffer {
		val fd = context.assets.openFd(assetPath)
		FileInputStream(fd.fileDescriptor).use { fis ->
			val channel = fis.channel
			return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
		}
	}

	fun run(sequence: Array<FloatArray>): CtcOutputs {
		val interpreter = interpreter ?: throw IllegalStateException("Interpreter not loaded")
        val batch = 1
        val inputDim = meta.input_dim
        val baseShape = interpreter.getInputTensor(0).shape()
        val baseT = if (baseShape.size == 3 && baseShape[2] == inputDim) baseShape[1]
                    else if (baseShape.size == 2 && baseShape[1] == inputDim) baseShape[0]
                    else -1
        val targetT = if (baseT > 0) baseT else sequence.size
        var timeSteps = targetT

        // Try to resize input to [1, T, 178]. If resizing fails (static graph), pad/truncate.
        var resized = false
        if (baseT <= 0) {
            try {
                val inShape = interpreter.getInputTensor(0).shape()
                if (inShape.size == 3) {
                    if (inShape[1] != timeSteps || inShape[2] != inputDim) {
                        interpreter.resizeInput(0, intArrayOf(1, timeSteps, inputDim))
                        interpreter.allocateTensors()
                    }
                    resized = true
                } else if (inShape.size == 2 && inShape[1] == inputDim) {
                    if (inShape[0] != timeSteps) {
                        interpreter.resizeInput(0, intArrayOf(timeSteps, inputDim))
                        interpreter.allocateTensors()
                    }
                    resized = true
                }
            } catch (t: Throwable) {
                Log.w(TAG, "Input resize not supported; will pad/truncate if needed", t)
            }
        }

        var usedSeq = sequence
        if (baseT > 0) {
            usedSeq = padOrTruncate(sequence, targetT, inputDim)
            timeSteps = targetT
        } else if (!resized) {
            val shape = interpreter.getInputTensor(0).shape()
            val expectedT = when {
                shape.size == 3 && shape[2] == inputDim -> shape[1]
                shape.size == 2 && shape[1] == inputDim -> shape[0]
                else -> timeSteps
            }
            if (expectedT != timeSteps) {
                usedSeq = padOrTruncate(sequence, expectedT, inputDim)
                timeSteps = expectedT
            }
        }

        val inputBuffer = ByteBuffer.allocateDirect(batch * timeSteps * inputDim * 4).order(ByteOrder.nativeOrder())
        for (t in 0 until timeSteps) {
            for (f in 0 until inputDim) inputBuffer.putFloat(usedSeq[t][f])
        }
        inputBuffer.rewind()

		// Determine output indices and shapes dynamically
		val outCount = interpreter.outputTensorCount
		var ctcIdx = 0
		var catIdx: Int? = null
		var ctcShape = interpreter.getOutputTensor(0).shape()
		for (i in 0 until outCount) {
			val s = interpreter.getOutputTensor(i).shape()
			if (s.any { it == meta.num_ctc }) {
				ctcIdx = i
				ctcShape = s
			} else if (meta.num_cat > 0 && s.any { it == meta.num_cat }) {
				catIdx = i
			}
		}

		// Allocate direct ByteBuffers for outputs
		val outputs: MutableMap<Int, Any> = mutableMapOf()
		val ctcTensor = interpreter.getOutputTensor(ctcIdx)
		val ctcBytes = ctcTensor.numBytes()
		val ctcBuf = ByteBuffer.allocateDirect(ctcBytes).order(ByteOrder.nativeOrder())
		outputs[ctcIdx] = ctcBuf

		var catBuf: ByteBuffer? = null
		if (catIdx != null) {
			val catTensor = interpreter.getOutputTensor(catIdx!!)
			catBuf = ByteBuffer.allocateDirect(catTensor.numBytes()).order(ByteOrder.nativeOrder())
			outputs[catIdx!!] = catBuf
		}

		interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

		// Convert ctcBuf -> [T, C]
		ctcBuf.rewind()
		val ctcFloats = FloatArray(ctcBytes / 4)
		ctcBuf.asFloatBuffer().get(ctcFloats)
        val logProbs2D = reshapeToTimeClasses(ctcFloats, ctcShape, timeSteps, meta.num_ctc)
		val logProbs = Array(1) { logProbs2D }

		var catOut: Array<Array<FloatArray>>? = null
		if (catBuf != null && catIdx != null) {
			catBuf.rewind()
			val s = interpreter.getOutputTensor(catIdx!!).shape()
			val floats = FloatArray(catBuf.capacity() / 4)
			catBuf.asFloatBuffer().get(floats)
			val twoD = reshapeToTimeClasses(floats, s, timeSteps, meta.num_cat)
			catOut = Array(1) { twoD }
		}

		return CtcOutputs(logProbs = logProbs, catLogits = catOut)
	}

    private fun padOrTruncate(seq: Array<FloatArray>, targetT: Int, dim: Int): Array<FloatArray> {
        val out = Array(targetT) { FloatArray(dim) }
        val copyLen = kotlin.math.min(seq.size, targetT)
        for (t in 0 until copyLen) {
            val src = seq[t]
            System.arraycopy(src, 0, out[t], 0, kotlin.math.min(src.size, dim))
        }
        return out
    }

    private fun reshapeToTimeClasses(flat: FloatArray, shape: IntArray, timeSteps: Int, classes: Int): Array<FloatArray> {
        return when (shape.size) {
            3 -> {
                val a = shape[0]; val b = shape[1]; val c = shape[2]
                when {
                    b == timeSteps && c == classes -> to2D(flat, b, c)
                    c == timeSteps && b == classes -> transpose(to2D(flat, b, c))
                    else -> to2D(flat, timeSteps, classes)
                }
            }
            2 -> {
                val a = shape[0]; val b = shape[1]
                when {
                    a == timeSteps && b == classes -> to2D(flat, a, b)
                    b == timeSteps && a == classes -> transpose(to2D(flat, a, b))
                    else -> to2D(flat, timeSteps, classes)
                }
            }
            1 -> to2D(flat, 1, classes)
            else -> to2D(flat, timeSteps, classes)
        }
    }

    private fun to2D(flat: FloatArray, rows: Int, cols: Int): Array<FloatArray> {
        val out = Array(rows) { FloatArray(cols) }
        var idx = 0
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[r][c] = flat[idx++]
            }
        }
        return out
    }

    private fun transpose(arr: Array<FloatArray>): Array<FloatArray> {
        val rows = arr.size
        val cols = if (rows > 0) arr[0].size else 0
        val out = Array(cols) { FloatArray(rows) }
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[c][r] = arr[r][c]
            }
        }
        return out
    }

	fun release() {
		interpreter?.close()
		gpuDelegate?.close()
	}
}


