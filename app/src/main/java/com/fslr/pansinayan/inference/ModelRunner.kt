package com.fslr.pansinayan.inference

interface ModelRunner {
	val meta: CtcModelMetadata
	fun run(sequence: Array<FloatArray>): CtcOutputs
	fun release()
}


