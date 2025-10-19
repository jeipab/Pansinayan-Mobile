package com.fslr.pansinayan.activities

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.tabs.TabLayout
import com.fslr.pansinayan.adapter.HistoryAdapter
import com.fslr.pansinayan.database.AppDatabase
import com.fslr.pansinayan.database.RecognitionHistory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * Activity for viewing and managing recognition history.
 * 
 * Features:
 * - View history filtered by model (All/Transformer/GRU)
 * - Clear all history
 * - Export history to CSV
 */
class HistoryActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "HistoryActivity"
        private const val CREATE_CSV_REQUEST = 1001
    }

    private lateinit var database: AppDatabase
    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: HistoryAdapter
    private lateinit var tabLayout: TabLayout
    private lateinit var btnClearHistory: Button
    private lateinit var btnDownloadCsv: Button
    private lateinit var toolbar: Toolbar

    private var currentFilter = "All"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        // Initialize views
        toolbar = findViewById(R.id.toolbar)
        tabLayout = findViewById(R.id.tab_layout)
        recyclerView = findViewById(R.id.recycler_view)
        btnClearHistory = findViewById(R.id.btn_clear_history)
        btnDownloadCsv = findViewById(R.id.btn_download_csv)

        // Set up toolbar
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener {
            finish()
        }

        // Initialize database
        database = AppDatabase.getDatabase(this)

        // Set up RecyclerView
        adapter = HistoryAdapter()
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        // Set up tab listener
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> {
                        currentFilter = "All"
                        loadHistory()
                    }
                    1 -> {
                        currentFilter = "Transformer"
                        loadHistoryByModel("Transformer")
                    }
                    2 -> {
                        currentFilter = "GRU"
                        loadHistoryByModel("GRU")
                    }
                }
            }

            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        // Set up button listeners
        btnClearHistory.setOnClickListener {
            showClearHistoryDialog()
        }

        btnDownloadCsv.setOnClickListener {
            exportHistoryToCSV()
        }

        // Load initial history
        loadHistory()
    }

    /**
     * Load all history from database.
     */
    private fun loadHistory() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val history = database.historyDao().getAllHistory()
                withContext(Dispatchers.Main) {
                    adapter.submitList(history)
                    Log.i(TAG, "Loaded ${history.size} history items")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load history", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@HistoryActivity, "Failed to load history", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /**
     * Load history filtered by model.
     */
    private fun loadHistoryByModel(modelName: String) {
        lifecycleScope.launch {
            database.historyDao().getHistoryByModel(modelName).collectLatest { history ->
                adapter.submitList(history)
                Log.i(TAG, "Loaded ${history.size} history items for $modelName")
            }
        }
    }

    /**
     * Show confirmation dialog before clearing history.
     */
    private fun showClearHistoryDialog() {
        AlertDialog.Builder(this)
            .setTitle("Clear History")
            .setMessage("Are you sure you want to delete all recognition history? This action cannot be undone.")
            .setPositiveButton("Clear") { _, _ ->
                clearHistory()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /**
     * Clear all history from database.
     */
    private fun clearHistory() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                database.historyDao().clearAllHistory()
                withContext(Dispatchers.Main) {
                    adapter.submitList(emptyList())
                    Toast.makeText(this@HistoryActivity, "History cleared", Toast.LENGTH_SHORT).show()
                    Log.i(TAG, "History cleared")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear history", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@HistoryActivity, "Failed to clear history", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /**
     * Export history to CSV file.
     */
    private fun exportHistoryToCSV() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val history = database.historyDao().getAllHistory()
                
                if (history.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@HistoryActivity, "No history to export", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                // Build CSV content
                val csvContent = buildCSVContent(history)

                // Launch file picker to save CSV
                withContext(Dispatchers.Main) {
                    val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "text/csv"
                        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                        putExtra(Intent.EXTRA_TITLE, "recognition_history_$timestamp.csv")
                    }
                    startActivityForResult(intent, CREATE_CSV_REQUEST)
                    
                    // Store CSV content temporarily
                    pendingCsvContent = csvContent
                }

            } catch (e: Exception) {
                Log.e(TAG, "Failed to export CSV", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@HistoryActivity, "Failed to export CSV", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private var pendingCsvContent: String? = null

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == CREATE_CSV_REQUEST && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                writeCSVToUri(uri, pendingCsvContent ?: "")
                pendingCsvContent = null
            }
        }
    }

    /**
     * Build CSV content from history list.
     */
    private fun buildCSVContent(history: List<RecognitionHistory>): String {
        val builder = StringBuilder()
        
        // Header
        builder.append("ID,Timestamp,Date,Gloss Label,Category,Occlusion Status,Model Used\n")
        
        // Data rows
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        for (item in history) {
            builder.append("${item.id},")
            builder.append("${item.timestamp},")
            builder.append("${dateFormat.format(Date(item.timestamp))},")
            builder.append("\"${item.glossLabel}\",")
            builder.append("\"${item.categoryLabel}\",")
            builder.append("${item.occlusionStatus},")
            builder.append("${item.modelUsed}\n")
        }
        
        return builder.toString()
    }

    /**
     * Write CSV content to URI.
     */
    private fun writeCSVToUri(uri: Uri, content: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                contentResolver.openOutputStream(uri)?.use { outputStream ->
                    outputStream.write(content.toByteArray())
                }
                
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@HistoryActivity, "CSV exported successfully", Toast.LENGTH_LONG).show()
                    Log.i(TAG, "CSV exported to: $uri")
                }
            } catch (e: IOException) {
                Log.e(TAG, "Failed to write CSV", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@HistoryActivity, "Failed to write CSV file", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
}

