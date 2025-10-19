package com.fslr.pansinayan.adapter

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.fslr.pansinayan.R
import com.fslr.pansinayan.database.RecognitionHistory
import java.text.SimpleDateFormat
import java.util.*

/**
 * RecyclerView adapter for displaying recognition history.
 */
class HistoryAdapter : RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    private var historyList: List<RecognitionHistory> = emptyList()
    private val dateFormat = SimpleDateFormat("MMM dd, HH:mm:ss", Locale.getDefault())

    /**
     * Update the history list and refresh the view.
     */
    fun submitList(newList: List<RecognitionHistory>) {
        historyList = newList
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HistoryViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return HistoryViewHolder(view)
    }

    override fun onBindViewHolder(holder: HistoryViewHolder, position: Int) {
        holder.bind(historyList[position])
    }

    override fun getItemCount(): Int = historyList.size

    /**
     * ViewHolder for history items.
     */
    inner class HistoryViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvGlossLabel: TextView = itemView.findViewById(R.id.tv_gloss_label)
        private val tvCategoryLabel: TextView = itemView.findViewById(R.id.tv_category_label)
        private val tvTimestamp: TextView = itemView.findViewById(R.id.tv_timestamp)
        private val tvModel: TextView = itemView.findViewById(R.id.tv_model)
        private val tvOcclusion: TextView = itemView.findViewById(R.id.tv_occlusion)

        fun bind(history: RecognitionHistory) {
            tvGlossLabel.text = history.glossLabel
            tvCategoryLabel.text = "Category: ${history.categoryLabel}"
            tvTimestamp.text = dateFormat.format(Date(history.timestamp))
            tvModel.text = history.modelUsed

            // Set occlusion status with color
            tvOcclusion.text = history.occlusionStatus
            tvOcclusion.setTextColor(
                if (history.occlusionStatus == "Occluded") Color.RED else Color.GREEN
            )
        }
    }
}

