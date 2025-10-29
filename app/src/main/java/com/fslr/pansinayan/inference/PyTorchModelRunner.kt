package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PyTorchModelRunner(
	private val context: Context,
	private val assetModelPath: String, // e.g., "ctc/sign_transformer_ctc_mobile.ptl" or ".pt"
	private val metadataPath: String
) : ModelRunner {

	companion object {
		private const val TAG = "PyTorchModelRunner"
	}

	override val meta: CtcModelMetadata = ModelMetadataLoader.loadFromAssets(context, metadataPath)

    private val module: Module by lazy {
        val filePath = ensureAssetOnDisk(context, assetModelPath)
        Module.load(filePath)
    }

	override fun run(sequence: Array<FloatArray>): CtcOutputs {
		// Build input tensor [1, T, D] using a DIRECT buffer (required by PyTorch Mobile)
		val t = sequence.size
		val d = if (t > 0) sequence[0].size else meta.input_dim
		val byteBuf = ByteBuffer.allocateDirect(4 * t * d).order(ByteOrder.nativeOrder())
		val floatBuf = byteBuf.asFloatBuffer()
		for (i in 0 until t) floatBuf.put(sequence[i])
		floatBuf.rewind()
		val input = Tensor.fromBlob(floatBuf, longArrayOf(1L, t.toLong(), d.toLong()))

		val iv = module.forward(IValue.from(input))
		return if (iv.isTensor) {
			val out = iv.toTensor()
			val log = reshapeTo2D(out)
			CtcOutputs(logProbs = arrayOf(log), catLogits = null)
		} else if (iv.isTuple) {
			val tuple = iv.toTuple()
			// Expect [ctc_logits, category_logits]
			val ctcTensor = tuple.getOrNull(0)?.toTensor()
			val catTensor = tuple.getOrNull(1)?.toTensor()
			val log = ctcTensor?.let { reshapeTo2D(it) }
			val cat = catTensor?.let { reshapeTo2D(it) }
			CtcOutputs(
				logProbs = arrayOf(log ?: emptyArray()),
				catLogits = cat?.let { arrayOf(it) }
			)
		} else {
			// Fallback: unsupported output type
			Log.w(TAG, "Unexpected model output type: ${iv}")
			CtcOutputs(logProbs = arrayOf(emptyArray()), catLogits = null)
		}
	}

	private fun reshapeTo2D(tensor: Tensor): Array<FloatArray> {
		// Accepts [1, T, C] or [T, C]; returns [T][C]
		val shape = tensor.shape()
		val flat = tensor.dataAsFloatArray
		val (T, C) = when (shape.size) {
			3 -> shape[1].toInt() to shape[2].toInt()
			2 -> shape[0].toInt() to shape[1].toInt()
			else -> 0 to 0
		}
		val out = Array(T) { FloatArray(C) }
		var idx = 0
		for (i in 0 until T) {
			for (j in 0 until C) {
				out[i][j] = flat[idx++]
			}
		}
		return out
	}

	override fun release() {
		// No explicit release needed for PyTorch Mobile; keep for symmetry
	}

	private fun ensureAssetOnDisk(context: Context, assetPath: String): String {
		val outFile = File(context.filesDir, assetPath.substringAfterLast('/'))
		if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
		context.assets.open(assetPath).use { input ->
			outFile.outputStream().use { output ->
				input.copyTo(output)
			}
		}
		return outFile.absolutePath
	}
}


