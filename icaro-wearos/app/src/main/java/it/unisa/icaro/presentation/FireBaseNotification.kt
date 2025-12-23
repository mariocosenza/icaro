package it.unisa.icaro.presentation

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import kotlinx.coroutines.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

class FirebaseNotificationService : FirebaseMessagingService() {

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(serviceJob + Dispatchers.Default)

    private val httpClient: OkHttpClient = OkHttpClient.Builder()
        .callTimeout(5, TimeUnit.SECONDS)
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)

        Log.d("FIREBASE_DEBUG", "Message received from: ${message.from}")
        Log.d("FIREBASE_DEBUG", "Data payload: ${message.data}")

        vibrateOnce()

        val title = message.notification?.title?.trim().orEmpty()
        val isStartMonitoring = title.equals("Start monitoring", ignoreCase = true)

        if (!isStartMonitoring) {
            val intent = Intent("FALL_EVENT")
            intent.setPackage(packageName)
            sendBroadcast(intent)
        }

        serviceScope.launch {
            measureAndSendHeartRateAndAccel(windowMs = 2000L)
        }
    }

    private fun vibrateOnce() {
        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vm.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            if (!vibrator.hasVibrator()) return

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(250, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(250)
            }
        } catch (e: Exception) {
            Log.w("ICARO_SERVICE", "Vibration error", e)
        }
    }

    private suspend fun measureAndSendHeartRateAndAccel(windowMs: Long) {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        val heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        if (heartRateSensor == null) {
            Log.e("ICARO_SERVICE", "No Heart Rate Sensor found")
            return
        }

        val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (accelSensor == null) {
            Log.e("ICARO_SERVICE", "No Accelerometer Sensor found")
            return
        }

        val hrReadings = mutableListOf<Float>()
        val axReadings = mutableListOf<Float>()
        val ayReadings = mutableListOf<Float>()
        val azReadings = mutableListOf<Float>()

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_HEART_RATE -> {
                        if (event.values.isNotEmpty()) {
                            val v = event.values[0]
                            if (v > 0f) hrReadings.add(v)
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

            delay(windowMs)
        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "Sensor collection error", e)
            return
        } finally {
            sensorManager.unregisterListener(listener)
        }

        val meanHr: Double? = hrReadings.takeIf { it.isNotEmpty() }?.average()
        val meanAx: Double? = axReadings.takeIf { it.isNotEmpty() }?.average()
        val meanAy: Double? = ayReadings.takeIf { it.isNotEmpty() }?.average()
        val meanAz: Double? = azReadings.takeIf { it.isNotEmpty() }?.average()

        if (meanHr != null) sendHeartRateToApi(meanHr) else Log.w("ICARO_SERVICE", "No heart rate data collected")
        if (meanAx != null && meanAy != null && meanAz != null) {
            sendAccelToApi(meanAx, meanAy, meanAz)
        } else {
            Log.w("ICARO_SERVICE", "No accelerometer data collected")
        }
    }

    private suspend fun sendHeartRateToApi(meanHeartbeat: Double) = withContext(Dispatchers.IO) {
        val targetUrl = "http://192.168.1.15:8000/api/v1/measure/${meanHeartbeat.toInt()}"
        doGet(targetUrl, tag = "ICARO_HEART")
    }

    private suspend fun sendAccelToApi(x: Double, y: Double, z: Double) = withContext(Dispatchers.IO) {
        val fx = String.format(java.util.Locale.US, "%.3f", x)
        val fy = String.format(java.util.Locale.US, "%.3f", y)
        val fz = String.format(java.util.Locale.US, "%.3f", z)

        val targetUrl = "http://192.168.1.15:8000/api/v1/monitor/$fx-$fy-$fz"
        doGet(targetUrl, tag = "ICARO_ACCEL")
    }

    private fun doGet(url: String, tag: String) {
        try {
            val request = Request.Builder().url(url).get().build()
            httpClient.newCall(request).execute().use { response ->
                val body = response.body?.string().orEmpty()
                if (response.isSuccessful) {
                    Log.d("ICARO_SERVICE", "[$tag] Success (${response.code}): $body")
                } else {
                    Log.e("ICARO_SERVICE", "[$tag] Failed (${response.code}): $body")
                }
            }
        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "[$tag] Network Error: $url", e)
        }
    }

    override fun onNewToken(token: String) {}

    override fun onDestroy() {
        super.onDestroy()
        serviceJob.cancel()
    }
}
