package it.unisa.icaro.presentation

import android.Manifest
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.RingtoneManager
import android.os.VibrationEffect
import android.os.VibratorManager
import android.util.Log
import androidx.annotation.RequiresPermission
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import kotlinx.coroutines.*
import java.net.HttpURLConnection
import java.net.URL

class FirebaseNotificationService : FirebaseMessagingService() {

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(serviceJob + Dispatchers.Default)

    // List of target IPs for redundancy
    private val targetIps = listOf("192.168.1.15", "192.168.27.66")

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)

        Log.d("FIREBASE_DEBUG", "Message received from: ${message.from}")
        Log.d("FIREBASE_DEBUG", "Data payload: ${message.data}")

        notifyUser()

        val title = message.notification?.title?.trim().orEmpty()
        val isStartMonitoring = title.equals("Start monitoring", ignoreCase = true)

        if (!isStartMonitoring) {
            val intent = Intent("FALL_EVENT")
            intent.setPackage(packageName)
            sendBroadcast(intent)
        }

        serviceScope.launch {
            // Increased timeout to allow sensor to wake up (60 seconds)
            measureAndSendHeartRateAndAccel(timeoutMs = 60000L)
        }
    }

    @RequiresPermission(Manifest.permission.VIBRATE)
    private fun notifyUser() {
        try {
            // 1. Vibrate
            val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            val vibrator = vm.defaultVibrator

            if (vibrator.hasVibrator()) {
                vibrator.vibrate(VibrationEffect.createOneShot(250, VibrationEffect.DEFAULT_AMPLITUDE))
            }

            // 2. Sound
            // Try Alarm sound first (more intrusive), then Notification
            var notificationUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            if (notificationUri == null) {
                 notificationUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            }
            
            val ringtone = RingtoneManager.getRingtone(applicationContext, notificationUri)
            
            // Set attributes to ensure it plays loudly
             ringtone.audioAttributes = android.media.AudioAttributes.Builder()
                .setUsage(android.media.AudioAttributes.USAGE_ALARM)
                .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()
                
            ringtone.play()

        } catch (e: Exception) {
            Log.w("ICARO_SERVICE", "Notification error (vibration/sound)", e)
        }
    }

    private suspend fun measureAndSendHeartRateAndAccel(timeoutMs: Long) {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        val heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        
        if (heartRateSensor == null || accelSensor == null) {
            Log.e("ICARO_SERVICE", "Sensors not found")
             sendHeartRateToApi(0.0)
             sendAccelToApi(0.0, 0.0, 0.0)
             return
        }

        val hrReadings = mutableListOf<Float>()
        val axReadings = mutableListOf<Float>()
        val ayReadings = mutableListOf<Float>()
        val azReadings = mutableListOf<Float>()

        // Used to signal when the first valid heart rate is received
        val hrReceived = CompletableDeferred<Unit>()

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_HEART_RATE -> {
                        if (event.values.isNotEmpty()) {
                            val v = event.values[0]
                            if (v > 0f) {
                                hrReadings.add(v)
                                
                                // Broadcast live heart rate
                                val intent = Intent("HEART_RATE_UPDATE")
                                intent.putExtra("heart_rate", v)
                                intent.setPackage(packageName)
                                sendBroadcast(intent)
                                
                                // Complete the deferred value to signal we have data
                                if (hrReceived.isActive) {
                                    hrReceived.complete(Unit)
                                }
                            }
                        }
                    }
                    Sensor.TYPE_ACCELEROMETER -> {
                        if (event.values.size >= 3) {
                            axReadings.add(event.values[0])
                            ayReadings.add(event.values[1])
                            azReadings.add(event.values[2])
                        }
                    }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        try {
            sensorManager.registerListener(listener, heartRateSensor, SensorManager.SENSOR_DELAY_NORMAL)
            sensorManager.registerListener(listener, accelSensor, SensorManager.SENSOR_DELAY_GAME)

            Log.d("ICARO_SERVICE", "Waiting for Heart Rate data (max ${timeoutMs}ms)...")
            
            // Wait until we get at least one valid HR reading or timeout
            withTimeoutOrNull(timeoutMs) {
                hrReceived.await()
            }
            
            if (hrReadings.isNotEmpty()) {
                Log.d("ICARO_SERVICE", "Heart Rate detected!")
            } else {
                 Log.w("ICARO_SERVICE", "Heart Rate sensor timed out.")
            }

        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "Sensor collection error", e)
        } finally {
            sensorManager.unregisterListener(listener)
        }

        val meanHr: Double = hrReadings.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val meanAx: Double = axReadings.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val meanAy: Double = ayReadings.takeIf { it.isNotEmpty() }?.average() ?: 0.0
        val meanAz: Double = azReadings.takeIf { it.isNotEmpty() }?.average() ?: 0.0

        sendHeartRateToApi(meanHr)
        sendAccelToApi(meanAx, meanAy, meanAz)
    }

    private suspend fun sendHeartRateToApi(meanHeartbeat: Double) = withContext(Dispatchers.IO) {
        val path = "/api/v1/measure/${meanHeartbeat.toInt()}"
        sendToAllIps(path, "ICARO_HEART")
    }

    private suspend fun sendAccelToApi(x: Double, y: Double, z: Double) = withContext(Dispatchers.IO) {
        val fx = String.format(java.util.Locale.US, "%.3f", x)
        val fy = String.format(java.util.Locale.US, "%.3f", y)
        val fz = String.format(java.util.Locale.US, "%.3f", z)

        val path = "/api/v1/monitor/$fx-$fy-$fz"
        sendToAllIps(path, "ICARO_ACCEL")
    }

    // Try sending to all configured IPs
    private fun sendToAllIps(path: String, tag: String) {
        for (ip in targetIps) {
            val url = "http://$ip:8000$path"
            try {
                doPost(url, tag)
            } catch (e: Exception) {
                // Log but continue to next IP
                Log.e("ICARO_SERVICE", "[$tag] Failed to send to $ip: ${e.message}")
            }
        }
    }

    private fun doPost(url: String, tag: String) {
        var connection: HttpURLConnection? = null
        try {
            val urlObj = URL(url)
            connection = urlObj.openConnection() as HttpURLConnection
            connection.requestMethod = "POST"
            // Set 30 seconds timeout
            connection.connectTimeout = 30000
            connection.readTimeout = 30000
            
            // Send empty body for POST as we are passing data in URL parameters
            connection.doOutput = true
            connection.setFixedLengthStreamingMode(0)

            val responseCode = connection.responseCode
            val inputStream = if (responseCode in 200..299) connection.inputStream else connection.errorStream
            val body = inputStream?.bufferedReader()?.use { it.readText() } ?: ""

            if (responseCode in 200..299) {
                Log.d("ICARO_SERVICE", "[$tag] Success ($responseCode) -> $url: $body")
            } else {
                Log.e("ICARO_SERVICE", "[$tag] Failed ($responseCode) -> $url: $body")
            }
        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "[$tag] Network Error: $url", e)
        } finally {
            connection?.disconnect()
        }
    }

    override fun onNewToken(token: String) {}

    override fun onDestroy() {
        super.onDestroy()
        serviceJob.cancel()
    }
}
