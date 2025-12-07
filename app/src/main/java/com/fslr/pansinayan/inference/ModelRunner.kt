package com.fslr.pansinayan.inference

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface ModelRunner {
	val meta: CtcModelMetadata
	fun run(sequence: Array<FloatArray>): CtcOutputs
	suspend fun runAsync(sequence: Array<FloatArray>): CtcOutputs = withContext(Dispatchers.Default) {
		run(sequence)
	}
	fun release()
	fun isAvailable(): Boolean = true
}


