package com.fslr.pansinayan

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.fslr.pansinayan.inference.CtcGreedyDecoder
import com.fslr.pansinayan.inference.PyTorchModelRunner
import com.fslr.pansinayan.io.NpyNpzReader
import com.fslr.pansinayan.recognition.CtcAggregator
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

data class GoldenManifest(
    @SerializedName("file_name") val fileName: String,
    val fps: Int? = null,
    val window_size: Int? = null,
    val stride: Int? = null,
    val decode_method: String? = null,
    val expected: ExpectedResult? = null
)

data class ExpectedResult(
    val sequence: List<Int>?,
    val confidences: List<Float>?
)

@RunWith(AndroidJUnit4::class)
class CtcGoldenParityTest {
    private val tolerance = 0.02f

    @Test
    fun runGoldenParity() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val assetManager = context.assets

        val runner = PyTorchModelRunner(
            context = context,
            assetModelPath = "ctc/SignTransformerCtc_best.pt",
            metadataPath = "ctc/SignTransformerCtc_best.model.json"
        )
        val blankId = runner.meta.blank_id

        val manifests = assetManager.list("golden")?.filter { it.endsWith(".json") } ?: emptyList()
        val gson = Gson()
        val reader = NpyNpzReader(context)

        var allPassed = true

        for (mf in manifests) {
            val mfPath = "golden/$mf"
            val mfObj = assetManager.open(mfPath).use { gson.fromJson(it.reader(), GoldenManifest::class.java) }
            val npzPath = "golden/${mfObj.fileName}"
            val X = reader.readXFromNpz(npzPath)

            val windowSize = mfObj.window_size ?: runner.meta.window_size_hint
            val stride = mfObj.stride ?: runner.meta.stride_hint
            val aggregator = CtcAggregator(iouThreshold = 0.5f)

            var start = 0
            while (start < X.size) {
                val end = kotlin.math.min(start + windowSize, X.size)
                val window = java.util.Arrays.copyOfRange(X, start, end)
                val outputs = runner.run(window)
                val logProbs = outputs.logProbs[0]
                val tokens = CtcGreedyDecoder.decode(logProbs, blankId)
                aggregator.addWindowTokens(start, tokens)
                start += stride
            }

            val aggTokens = aggregator.getAll()
            val predSeq = aggTokens.map { it.id }
            val predConf = aggTokens.map { it.confidence }

            val expSeq = mfObj.expected?.sequence
            val expConf = mfObj.expected?.confidences
            if (expSeq != null && expConf != null) {
                val ok = compareSequences(expSeq, predSeq) && compareConf(expConf, predConf, tolerance)
                if (!ok) {
                    allPassed = false
                    println("[GOLDEN FAIL] $mfPath\nexp=$expSeq\npred=$predSeq\nexpC=${format(expConf)}\npredC=${format(predConf)}")
                }
            } else {
                println("[GOLDEN INFO] No expected provided for $mfPath; skipping strict check")
            }
        }

        assertTrue("Golden parity failed; see logs for diffs", allPassed)
        runner.release()
    }

    private fun compareSequences(a: List<Int>, b: List<Int>): Boolean = a == b
    private fun compareConf(a: List<Float>, b: List<Float>, tol: Float): Boolean {
        if (a.size != b.size) return false
        for (i in a.indices) {
            if (kotlin.math.abs(a[i] - b[i]) > tol) return false
        }
        return true
    }
    private fun format(arr: List<Float>): String = arr.joinToString(prefix = "[", postfix = "]") { String.format("%.3f", it) }
}


