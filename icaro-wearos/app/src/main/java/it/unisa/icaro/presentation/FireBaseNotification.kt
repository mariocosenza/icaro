package it.unisa.icaro.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class FirebaseNotificationService : FirebaseMessagingService() {

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        if (message.from?.endsWith("/topic/fall") == true) {
            measureAndSendHeartRate()
        }
    }

    private fun measureAndSendHeartRate() {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)

        if (heartRateSensor == null) {
            Log.e("ICARO_SERVICE", "No Heart Rate Sensor found on this device.")
            return
        }

        val readings = mutableListOf<Float>()

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event != null && event.values.isNotEmpty()) {
                    val value = event.values[0]
                    readings.add(value)
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        sensorManager.registerListener(listener, heartRateSensor, SensorManager.SENSOR_DELAY_NORMAL)

        try {
            Thread.sleep(2000)
        } catch (e: InterruptedException) {
            Log.e("ICARO_SERVICE", "Sleep interrupted", e)
        }

        sensorManager.unregisterListener(listener)

        if (readings.isNotEmpty()) {
            val meanValue = readings.average()
            sendHeartRateToApi(meanValue)
        } else {
            Log.w("ICARO_SERVICE", "No heart rate data collected in 2 seconds.")
        }
    }

    private fun sendHeartRateToApi(meanHeartbeat: Double) {
        try {
            val targetUrl = "http://192.168.1.15:8000/api/v1/mesure/${meanHeartbeat.toInt()}"
            val response = khttp.post(url = targetUrl)

            if (response.statusCode == 200) {
                Log.d("ICARO_SERVICE", "Success: ${response.text}")
            } else {
                Log.e("ICARO_SERVICE", "Failed: ${response.statusCode}")
            }
        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "Network Error", e)
        }
    }

    override fun onNewToken(token: String) {}
}