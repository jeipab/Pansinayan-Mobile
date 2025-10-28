package com.fslr.pansinayan.inference

import android.content.Context
import com.google.gson.Gson
import java.io.InputStreamReader

data class CtcModelMetadata(
	val input_dim: Int,
	val num_gloss: Int,
	val blank_id: Int,
	val num_ctc: Int,
	val num_cat: Int,
	val window_size_hint: Int,
	val stride_hint: Int,
	val decode_default: String,
	val model_type: String,
	val labels_file: String,
	val version: String,
	val labels_checksum: String? = null
)

object ModelMetadataLoader {
	fun loadFromAssets(context: Context, assetPath: String): CtcModelMetadata {
		context.assets.open(assetPath).use { input ->
			InputStreamReader(input).use { reader ->
				return Gson().fromJson(reader, CtcModelMetadata::class.java)
			}
		}
	}
}


