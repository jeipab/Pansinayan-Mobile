package com.fslr.pansinayan.inference

object CTCDecoder {
	fun greedy(logProbs: Array<FloatArray>, blankId: Int) = CtcGreedyDecoder.decode(logProbs, blankId)
}


