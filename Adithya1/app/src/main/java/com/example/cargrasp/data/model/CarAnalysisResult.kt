package com.example.cargrasp.data.model

import com.google.gson.annotations.SerializedName

/**
 * Data class representing the structured car recognition result
 * adhering strictly to the Car Vision AI Technical Blueprint (Slide 8).
 */
data class CarAnalysisResult(
    @SerializedName("detected", alternate = ["carDetected"])
    val detected: Boolean = false,

    @SerializedName("make")
    val make: String = "Unknown",

    @SerializedName("model")
    val model: String = "Unknown",

    @SerializedName("year", alternate = ["estimatedYearRange"])
    val year: String = "Unknown",

    @SerializedName("color", alternate = ["colour"])
    val color: String = "Unknown",

    @SerializedName("color_hex", alternate = ["hexColor"])
    val colorHex: String = "#C0C0C0",

    @SerializedName("body_type", alternate = ["bodyType"])
    val bodyType: String = "Unknown",

    @SerializedName("confidence")
    val confidence: Any = 95, // Supports both integer (98) or String ("High (98%)")

    @SerializedName("additional_details", alternate = ["notes"])
    val additionalDetails: String = "",

    @SerializedName("source")
    val source: String = "Google Gemini 1.5 Flash Vision"
) {
    val carDetected: Boolean get() = detected
    val colour: String get() = color
    val estimatedYearRange: String get() = year
    val notes: String get() = additionalDetails

    /**
     * Checks whether a car was successfully recognized with reasonable certainty.
     */
    fun isRecognized(): Boolean = detected && !make.equals("Unknown", ignoreCase = true) && !make.equals("Not Detected", ignoreCase = true)

    /**
     * Formats car title as "[Make] [Model]" or fallback if not detected.
     */
    fun getFullTitle(): String {
        return if (detected) {
            when {
                make.isNotBlank() && model.isNotBlank() && !make.equals("Unknown", true) && !model.equals("Unknown", true) -> "$make $model"
                make.isNotBlank() && !make.equals("Unknown", true) -> make
                else -> "Vehicle Detected"
            }
        } else {
            "No Automobile Recognized"
        }
    }

    /**
     * Formats subtitle as "[Year] • [Body Type]"
     */
    fun getSubtitle(): String {
        return if (year.isNotBlank() && year != "Unknown" && year != "N/A") {
            "$year • $bodyType"
        } else {
            bodyType
        }
    }

    /**
     * Gets integer confidence percentage (0-100)
     */
    fun getConfidenceInt(): Int {
        return when (confidence) {
            is Number -> confidence.toInt()
            is String -> {
                val match = Regex("""(\d+)%?""").find(confidence)
                match?.groupValues?.get(1)?.toIntOrNull() ?: if (isRecognized()) 95 else 15
            }
            else -> if (isRecognized()) 95 else 15
        }
    }
}
