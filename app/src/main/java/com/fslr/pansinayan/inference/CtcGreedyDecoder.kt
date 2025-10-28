package com.fslr.pansinayan.inference

import kotlin.math.exp

data class DecodedToken(
	val id: Int,
	val startT: Int,
	val endT: Int,
	val confidence: Float
)

object CtcGreedyDecoder {
	/**
	 * Greedy decode from log-probs [T, num_ctc].
	 * Returns list of tokens with python_parity confidence (avg per-frame max prob over the token span)
	 * and uniform timestamps in terms of frame indices.
	 */
	fun decode(logProbs: Array<FloatArray>, blankId: Int): List<DecodedToken> {
		if (logProbs.isEmpty()) return emptyList()
		val T = logProbs.size
		val numCtc = logProbs[0].size
		var prev = -1
		val tokens = mutableListOf<DecodedToken>()
		var curId = -1
		var curStart = -1
		val probsPerFrame = FloatArray(T) { 0f }
		val idsPerFrame = IntArray(T) { -1 }
		for (t in 0 until T) {
			var argmax = 0
			var maxVal = logProbs[t][0]
			for (c in 1 until numCtc) {
				if (logProbs[t][c] > maxVal) { maxVal = logProbs[t][c]; argmax = c }
			}
			idsPerFrame[t] = argmax
			probsPerFrame[t] = exp(maxVal)
		}
		for (t in 0 until T) {
			val id = idsPerFrame[t]
			if (id == prev) continue
			// close previous token
			if (curId != -1 && curId != blankId) {
				val end = t - 1
				val conf = average(probsPerFrame, curStart, end)
				tokens.add(DecodedToken(curId, curStart, end, conf))
			}
			// start new
			curId = id
			curStart = t
			prev = id
		}
		// flush
		if (curId != -1 && curId != blankId) {
			val end = T - 1
			val conf = average(probsPerFrame, curStart, end)
			tokens.add(DecodedToken(curId, curStart, end, conf))
		}
		return tokens
	}

	private fun average(arr: FloatArray, start: Int, end: Int): Float {
		if (end < start) return 0f
		var sum = 0f
		for (i in start..end) sum += arr[i]
		return sum / (end - start + 1)
	}
}


