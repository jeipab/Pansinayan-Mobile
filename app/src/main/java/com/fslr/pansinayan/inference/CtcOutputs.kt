package com.fslr.pansinayan.inference

data class CtcOutputs(
	val logProbs: Array<Array<FloatArray>>, // shape: [1][T][num_ctc]
	val catLogits: Array<Array<FloatArray>>? = null // optional: [1][T][num_cat]
)


