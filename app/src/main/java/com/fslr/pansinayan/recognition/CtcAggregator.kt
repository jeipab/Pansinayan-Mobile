package com.fslr.pansinayan.recognition

import com.fslr.pansinayan.inference.DecodedToken
import kotlin.math.max
import kotlin.math.min

data class AggregatedToken(
	val id: Int,
	var startT: Int,
	var endT: Int,
	var confidence: Float
)

class CtcAggregator(private val iouThreshold: Float = 0.5f) {
	private val tokens = mutableListOf<AggregatedToken>()
	private val newlyAdded = mutableListOf<AggregatedToken>()

	fun clear() {
		tokens.clear()
		newlyAdded.clear()
	}

	fun addWindowTokens(windowStart: Int, windowTokens: List<DecodedToken>): List<AggregatedToken> {
		newlyAdded.clear()
		for (wt in windowTokens) {
			val absStart = windowStart + wt.startT
			val absEnd = windowStart + wt.endT
			val candidate = AggregatedToken(wt.id, absStart, absEnd, wt.confidence)
			val matchIdx = findBestMatch(candidate)
			if (matchIdx == -1) {
				tokens.add(candidate)
				newlyAdded.add(candidate)
			} else {
				val existing = tokens[matchIdx]
				if (candidate.confidence > existing.confidence) {
					existing.startT = candidate.startT
					existing.endT = candidate.endT
					existing.confidence = candidate.confidence
					// treat as updated; don't re-emit to avoid duplicates
				}
			}
		}
		// Keep tokens ordered
		tokens.sortBy { it.startT }
		return newlyAdded.toList()
	}

	private fun iou(a: AggregatedToken, b: AggregatedToken): Float {
		val interStart = max(a.startT, b.startT)
		val interEnd = min(a.endT, b.endT)
		val inter = max(0, interEnd - interStart + 1)
		val lenA = a.endT - a.startT + 1
		val lenB = b.endT - b.startT + 1
		val union = lenA + lenB - inter
		if (union <= 0) return 0f
		return inter.toFloat() / union.toFloat()
	}

	private fun findBestMatch(candidate: AggregatedToken): Int {
		var bestIdx = -1
		var bestIou = 0f
		for ((i, t) in tokens.withIndex()) {
			if (t.id != candidate.id) continue
			val v = iou(t, candidate)
			if (v > bestIou) {
				bestIou = v
				bestIdx = i
			}
		}
		return if (bestIou >= iouThreshold) bestIdx else -1
	}

	fun getAll(): List<AggregatedToken> = tokens.toList()
}


