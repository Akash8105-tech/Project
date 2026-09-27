package com.example.cargrasp.data.repository

import android.graphics.Bitmap
import com.example.cargrasp.data.model.CarAnalysisResult
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content
import com.google.ai.client.generativeai.type.generationConfig
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.regex.Pattern

/**
 * Repository responsible for communicating with Google Gemini Vision API
 * to analyze car images and return structured car specifications.
 */
class GeminiCarRepository(
    private val apiKeyStorage: ApiKeyStorage
) {
    private val gson = Gson()

    /**
     * Sends the bitmap to Gemini Vision API and returns parsed CarAnalysisResult.
     */
    suspend fun analyzeCarImage(bitmap: Bitmap): Result<CarAnalysisResult> = withContext(Dispatchers.IO) {
        try {
            val apiKey = apiKeyStorage.getApiKey()
            if (apiKey.isBlank()) {
                return@withContext Result.failure(
                    IllegalStateException("Gemini API key is not configured. Please add your GEMINI_API_KEY to local.properties or enter it in app settings.")
                )
            }

            // Configure Gemini 1.5 Flash for fast multimodal vision recognition with JSON mode
            val generativeModel = GenerativeModel(
                modelName = "gemini-1.5-flash",
                apiKey = apiKey,
                generationConfig = generationConfig {
                    temperature = 0.2f
                    topK = 32
                    topP = 0.95f
                    responseMimeType = "application/json"
                }
            )

            val prompt = """
                You are an expert automotive image analyzer.
                Analyze this photo to identify if a car/automobile is present.
                If multiple vehicles are present, focus on the most prominent or centered car in the foreground.

                Return a JSON response adhering EXACTLY to this schema:
                {
                  "carDetected": boolean,
                  "make": string,
                  "model": string,
                  "colour": string,
                  "confidence": string,
                  "bodyType": string,
                  "estimatedYearRange": string,
                  "notes": string
                }

                Detailed Instructions:
                1. "carDetected": true if a recognizable car/automobile (sedan, SUV, truck, coupe, hatchback, van, supercar, etc.) is visible; false if no car is present or the object is unrelated.
                2. "make": The exact manufacturer / brand name (e.g. "Toyota", "BMW", "Honda", "Tesla", "Porsche", "Mercedes-Benz", "Hyundai", "Ford", "Tata", "Audi"). If car is not detected, set to "Not Detected". If uncertain, set to "Unknown/Uncertain".
                3. "model": The specific vehicle model name (e.g. "Corolla", "3 Series", "Civic", "Model Y", "911 Carrera", "Mustang", "Creta", "Nexon"). If unable to identify exact model, write "Unknown/Uncertain" rather than guessing.
                4. "colour": The primary exterior paint color (e.g. "Metallic Blue", "Pearl White", "Gloss Black", "Silver", "Racing Red", "Matte Grey").
                5. "confidence": One of ["High", "Medium", "Low", "Uncertain"]. High = clear badge/grille/taillight visible; Medium = general shape and body recognizable; Low/Uncertain = obscured or partial.
                6. "bodyType": One of ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Truck / Pickup", "Van / Minivan", "Supercar", "Wagon", "Unknown"].
                7. "estimatedYearRange": Approximate generation / model year (e.g. "2020-2024", "2015-2019", "Classic", or "Unknown").
                8. "notes": A concise 1-2 sentence explanation of distinctive styling cues recognized (e.g., signature grille, headlight daytime running lights, badges, body lines) or why identification was limited.
                
                CRITICAL:
                - Do NOT invent or hallucinate information. If the exact model cannot be determined, set "model": "Unknown/Uncertain".
                - If the image contains no car, set "carDetected": false, "make": "Not Detected", "model": "Not Detected", "colour": "Not Detected", "confidence": "Uncertain", "notes": "No automobile was recognized in this photo.".
            """.trimIndent()

            android.util.Log.d("CarGrasp", "[DEBUG] Preparing Gemini Vision Call: Bitmap=${bitmap.width}x${bitmap.height}, Bytes=${bitmap.byteCount}, Hash=${bitmap.hashCode()}")

            val inputContent = content {
                image(bitmap)
                text(prompt)
            }

            // Execute multimodal Gemini request
            val response = generativeModel.generateContent(inputContent)
            val responseText = response.text ?: ""

            if (responseText.isBlank()) {
                return@withContext Result.failure(
                    RuntimeException("Received an empty response from Gemini AI. Please try again with a clearer photo.")
                )
            }

            // Parse structured JSON response
            val parsedResult = parseCarResponse(responseText)
            android.util.Log.d("CarGrasp", "[DEBUG] Gemini AI Recognized: ${parsedResult.make} ${parsedResult.model} (${parsedResult.colour}) | Source=${parsedResult.source}")
            Result.success(parsedResult)

        } catch (e: Exception) {
            val friendlyMessage = mapExceptionToUserFriendlyMessage(e)
            android.util.Log.e("CarGrasp", "[DEBUG] Gemini AI Error: $friendlyMessage", e)
            Result.failure(Exception(friendlyMessage, e))
        }
    }

    /**
     * Parses the JSON output from Gemini safely, handling code blocks or malformed characters.
     */
    private fun parseCarResponse(rawResponse: String): CarAnalysisResult {
        return try {
            // Clean up possible markdown wrappers like ```json ... ```
            val cleanedJson = extractJsonString(rawResponse)
            
            val jsonObject = JsonParser.parseString(cleanedJson).asJsonObject

            val carDetected = jsonObject.get("carDetected")?.asBoolean ?: false
            val make = jsonObject.get("make")?.asString?.trim() ?: "Unknown"
            val model = jsonObject.get("model")?.asString?.trim() ?: "Unknown"
            val colour = jsonObject.get("colour")?.asString?.trim() ?: "Unknown"
            val confidence = jsonObject.get("confidence")?.asString?.trim() ?: "Uncertain"
            val bodyType = jsonObject.get("bodyType")?.asString?.trim() ?: "Unknown"
            val estimatedYearRange = jsonObject.get("estimatedYearRange")?.asString?.trim() ?: "Unknown"
            val notes = jsonObject.get("notes")?.asString?.trim() ?: ""

            CarAnalysisResult(
                detected = carDetected,
                make = make,
                model = model,
                color = colour,
                confidence = confidence,
                bodyType = bodyType,
                year = estimatedYearRange,
                additionalDetails = notes,
                source = "Google Gemini 1.5 Flash Vision"
            )
        } catch (e: Exception) {
            // If strict JSON parsing failed, try heuristic fallback or fallback result
            tryFallbackRegexParsing(rawResponse)
        }
    }

    /**
     * Extracts pure JSON from a string that might contain markdown backticks.
     */
    private fun extractJsonString(input: String): String {
        var clean = input.trim()
        if (clean.startsWith("```json")) {
            clean = clean.removePrefix("```json")
        } else if (clean.startsWith("```")) {
            clean = clean.removePrefix("```")
        }
        if (clean.endsWith("```")) {
            clean = clean.removeSuffix("```")
        }
        clean = clean.trim()

        // Find outer curly braces if extra text exists
        val firstBrace = clean.indexOf('{')
        val lastBrace = clean.lastIndexOf('}')
        if (firstBrace != -1 && lastBrace != -1 && lastBrace > firstBrace) {
            clean = clean.substring(firstBrace, lastBrace + 1)
        }
        return clean
    }

    /**
     * Fallback parser using regex if JSON structure is slightly malformed.
     */
    private fun tryFallbackRegexParsing(raw: String): CarAnalysisResult {
        fun extractKey(key: String): String? {
            val pattern = Pattern.compile("\"$key\"\\s*:\\s*\"([^\"]+)\"", Pattern.CASE_INSENSITIVE)
            val matcher = pattern.matcher(raw)
            return if (matcher.find()) matcher.group(1) else null
        }

        val carDetectedPattern = Pattern.compile("\"carDetected\"\\s*:\\s*(true|false)", Pattern.CASE_INSENSITIVE)
        val carDetectedMatcher = carDetectedPattern.matcher(raw)
        val carDetected = if (carDetectedMatcher.find()) {
            carDetectedMatcher.group(1).equals("true", ignoreCase = true)
        } else {
            raw.contains("make", ignoreCase = true) && !raw.contains("not detected", ignoreCase = true)
        }

        return CarAnalysisResult(
            carDetected = carDetected,
            make = extractKey("make") ?: if (carDetected) "Vehicle Detected" else "Not Detected",
            model = extractKey("model") ?: "Unknown",
            colour = extractKey("colour") ?: "Unknown",
            confidence = extractKey("confidence") ?: "Medium",
            bodyType = extractKey("bodyType") ?: "Unknown",
            estimatedYearRange = extractKey("estimatedYearRange") ?: "Unknown",
            notes = extractKey("notes") ?: raw.take(150)
        )
    }

    /**
     * Translates raw SDK/Network exceptions into clean, helpful user messages.
     */
    private fun mapExceptionToUserFriendlyMessage(e: Throwable): String {
        val msg = e.message.orEmpty()
        return when {
            msg.contains("API_KEY_INVALID", true) || msg.contains("API key not valid", true) ->
                "Invalid Gemini API key. Please check your API key in local.properties or settings."
            msg.contains("RESOURCE_EXHAUSTED", true) || msg.contains("429", true) || msg.contains("quota", true) ->
                "Gemini API rate limit or quota exceeded. Please wait a few seconds and retry."
            msg.contains("Unable to resolve host", true) || msg.contains("ConnectException", true) || msg.contains("UnknownHostException", true) ->
                "Network error: Unable to reach Gemini AI. Please check your internet connection."
            msg.contains("PERMISSION_DENIED", true) ->
                "Permission denied: Please ensure the Generative Language API is enabled in your Google AI Studio project."
            msg.isNotBlank() -> msg
            else -> "An unexpected error occurred while analyzing the image."
        }
    }
}
