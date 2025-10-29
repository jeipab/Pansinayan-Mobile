package com.fslr.pansinayan.inference

object PreprocessingUtils {
	fun clamp01(sequence: Array<FloatArray>): Array<FloatArray> {
		for (i in sequence.indices) {
			val row = sequence[i]
			for (j in row.indices) {
				var v = row[j]
				if (v < 0f) v = 0f else if (v > 1f) v = 1f
				row[j] = v
			}
		}
		return sequence
	}

	fun normalize(sequence: Array<FloatArray>, mean: FloatArray?, std: FloatArray?): Array<FloatArray> {
		if (mean == null || std == null) return sequence
		val dim = sequence.firstOrNull()?.size ?: return sequence
		if (mean.size != dim || std.size != dim) return sequence
		for (i in sequence.indices) {
			for (j in 0 until dim) {
				val s = if (std[j] != 0f) std[j] else 1f
				sequence[i][j] = (sequence[i][j] - mean[j]) / s
			}
		}
		return sequence
	}

	fun ensureShape(sequence: Array<FloatArray>, inputDim: Int): Array<FloatArray> {
		return if (sequence.isNotEmpty() && sequence[0].size == inputDim) sequence else Array(sequence.size) { FloatArray(inputDim) }
	}
}


