package com.fslr.pansinayan.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

/**
 * Home screen activity - entry point of the app.
 * 
 * Provides navigation to:
 * - Main Recognition Activity
 * - History Activity
 */
class HomeActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        // Set up button navigation
        findViewById<Button>(R.id.btn_start_recognition).setOnClickListener {
            // Navigate to main recognition screen
            startActivity(Intent(this, MainActivity::class.java))
        }

        findViewById<Button>(R.id.btn_view_history).setOnClickListener {
            // Navigate to history screen
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }
}

