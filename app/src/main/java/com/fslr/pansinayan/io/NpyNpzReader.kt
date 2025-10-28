package com.fslr.pansinayan.io

import android.content.Context
import android.util.Log
import java.io.BufferedInputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/**
 * Minimal NPZ/NPY reader for Android assets.
 *
 * Supports reading a single 2D float32 array named "X" from an NPZ file packaged in assets.
 * The NPY header parser supports C-order arrays with dtype '<f4' or '|f4'.
 */
class NpyNpzReader(private val context: Context) {
    companion object {
        private const val TAG = "NpyNpzReader"
    }

    /**
     * Read an NPZ file from assets/golden and extract the entry "X" as [T, 178] FloatArray[].
     * @param assetPath Path under assets, e.g., "golden/continuous_0001_S0_strategy1.npz"
     */
    fun readXFromNpz(assetPath: String): Array<FloatArray> {
        context.assets.open(assetPath).use { inputStream ->
            val zis = ZipInputStream(BufferedInputStream(inputStream))
            var entry: ZipEntry?
            while (true) {
                entry = zis.nextEntry ?: break
                if (!entry.isDirectory && (entry.name == "X.npy" || entry.name.endsWith("/X.npy"))) {
                    val bytes = readEntryFully(zis)
                    return parseNpy2DFloatArray(ByteArrayInputStream(bytes))
                }
            }
        }
        throw IllegalArgumentException("X.npy not found in NPZ: $assetPath")
    }

    private fun readEntryFully(zis: ZipInputStream): ByteArray {
        val buffer = ByteArray(8192)
        val baos = ByteArrayOutputStream()
        while (true) {
            val read = zis.read(buffer)
            if (read == -1) break
            if (read > 0) baos.write(buffer, 0, read)
        }
        return baos.toByteArray()
    }

    /**
     * Parse a NumPy .npy file containing a 2D float32 C-order array.
     * Supports headers like: magic, version, header_len, dict {'descr': '<f4', 'fortran_order': False, 'shape': (T, 178)}
     */
    private fun parseNpy2DFloatArray(input: InputStream): Array<FloatArray> {
        // Read magic: \x93NUMPY (6 bytes total)
        val magic = ByteArray(6)
        if (input.read(magic) != 6 || magic[0] != 0x93.toByte() || String(magic.copyOfRange(1, 6)) != "NUMPY") {
            throw IllegalArgumentException("Invalid NPY magic header")
        }
        // Version
        val ver = ByteArray(2)
        if (input.read(ver) != 2) throw IllegalStateException("Failed to read NPY version")
        val major = ver[0].toInt()
        val minor = ver[1].toInt()

        // Header length
        val headerLen = when (major) {
            1 -> {
                val lenBytes = ByteArray(2)
                if (input.read(lenBytes) != 2) throw IllegalStateException("Failed to read NPY v1 header length")
                ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
            }
            2, 3 -> {
                val lenBytes = ByteArray(4)
                if (input.read(lenBytes) != 4) throw IllegalStateException("Failed to read NPY v$major header length")
                ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).int
            }
            else -> throw IllegalArgumentException("Unsupported NPY version: $major.$minor")
        }

        // Header dict (ASCII, padded to align, ends with newline)
        val headerDictBytes = ByteArray(headerLen)
        var hOff = 0
        while (hOff < headerLen) {
            val r = input.read(headerDictBytes, hOff, headerLen - hOff)
            if (r == -1) throw IllegalStateException("Unexpected EOF reading NPY header")
            hOff += r
        }
        val headerStr = String(headerDictBytes)

        val descr = extractBetween(headerStr, "'descr':", ",")?.trim()?.trim('"', '\'', ' ')
        val fortran = extractBetween(headerStr, "'fortran_order':", ",")?.trim()?.toBoolean() ?: false
        val shapeStr = headerStr.substringAfter("'shape':").substringBefore(")").substringAfter("(")
        val shapeDims = shapeStr.split(",").mapNotNull { it.trim().toIntOrNull() }.toMutableList()
        if (shapeDims.size == 1 && headerStr.contains(",)")) {
            shapeDims.add(1)
        }

        // Determine dtype and byte order
        val byteOrder = when {
            descr?.startsWith(">") == true -> ByteOrder.BIG_ENDIAN
            else -> ByteOrder.LITTLE_ENDIAN
        }
        val bytesPerElem = when {
            descr == null -> 4
            descr.contains("f8") -> 8
            descr.contains("f4") -> 4
            else -> 4
        }
        if (fortran) {
            throw IllegalArgumentException("Fortran-order arrays not supported")
        }
        if (shapeDims.size != 2) {
            throw IllegalArgumentException("Only 2D arrays supported, got shape: $shapeDims")
        }
        val rows = shapeDims[0]
        val cols = shapeDims[1]

        val expectedBytes = rows.toLong() * cols.toLong() * bytesPerElem.toLong()
        if (expectedBytes <= 0L || expectedBytes > Int.MAX_VALUE) {
            throw IllegalArgumentException("Invalid array size: rows=$rows cols=$cols")
        }
        val dataBytes = ByteArray(expectedBytes.toInt())
        var offset = 0
        while (offset < dataBytes.size) {
            val read = input.read(dataBytes, offset, dataBytes.size - offset)
            if (read == -1) throw IllegalStateException("Unexpected EOF: read ${offset} < expected ${dataBytes.size}")
            offset += read
        }
        val bb = ByteBuffer.wrap(dataBytes).order(byteOrder)

        // Read into a temporary 2D float array in row-major order
        val tmp = Array(rows) { FloatArray(cols) }
        if (bytesPerElem == 4) {
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    tmp[r][c] = bb.float
                }
            }
        } else {
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    tmp[r][c] = bb.double.toFloat()
                }
            }
        }

        // Ensure output shape is [T, 178]
        if (cols == 178) {
            return tmp
        } else if (rows == 178) {
            val transposed = Array(cols) { FloatArray(rows) }
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    transposed[c][r] = tmp[r][c]
                }
            }
            return transposed
        } else {
            throw IllegalArgumentException("Unexpected X shape: (${rows}, ${cols}) — expected one dimension to be 178")
        }
    }

    private fun extractBetween(src: String, startKey: String, endKey: String): String? {
        val startIdx = src.indexOf(startKey)
        if (startIdx == -1) return null
        val after = src.substring(startIdx + startKey.length)
        val endIdx = after.indexOf(endKey)
        return if (endIdx == -1) after.trim() else after.substring(0, endIdx)
    }
}


